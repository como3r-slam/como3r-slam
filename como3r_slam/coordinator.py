"""Central coordinator process for multi-agent CoMo3R-SLAM.

Responsibilities:
- Maintain the shared ASMK retrieval database (cross-agent place recognition).
- For each new keyframe message from an agent, query the shared DB for
  candidates from OTHER agents and attempt to add an inter-agent factor.
- Always also add the sequential intra-agent edge (prev KF of same agent ->
  new KF) into the global graph so the global graph stays connected.
- Periodically run Level-2 Gauss-Newton on Sim(3) over all agents' poses.
- Notify agents to pull updated global poses back into their local stores.
"""
import math
import time
from typing import Dict, List

import torch

from como3r_slam.config import config, set_global_config
from como3r_slam.multi_agent import (
    GlobalFactorGraph,
    SharedRetrievalDatabase,
    snap_agent_chain,
)


def _publish_edges_for_viz(global_keyframes, global_graph):
    """Snapshot current global factor-graph edges into the shared lists
    consumed by the multi-agent visualizer."""
    ii = global_graph.graph.ii.cpu().tolist()
    jj = global_graph.graph.jj.cpu().tolist()
    global_keyframes.global_edges_ii[:] = ii
    global_keyframes.global_edges_jj[:] = jj


def _run_global_solve_safely(
    global_graph, kfs, use_calib, max_iters, max_jump_m, label
):
    """Run solve_global with NaN/divergence guard.

    Snapshots the full Sim(3) pose buffer for the active KF range, runs the
    CUDA Gauss-Newton, and verifies that the result is finite and that no
    pose moved more than `max_jump_m` meters. On failure, restores the
    snapshot so a single bad solve cannot poison subsequent iterations
    (one NaN pose feeds NaN into every later kernel via the Hessian build).

    Returns (ok, max_delta, mean_delta, moved, pre, post). On revert,
    post == pre and ok == False.
    """
    with kfs.lock:
        N = len(kfs)
        snapshot = kfs.T_WC[:N].clone()
        pre = snapshot[:, 0, :3].detach().cpu().numpy().copy()

    try:
        global_graph.solve_global(use_calib, max_iters=max_iters)
    except Exception as e:
        with kfs.lock:
            kfs.T_WC[:N] = snapshot
        print(f"[coordinator] {label}: solve raised, reverted ({e})")
        return False, float("nan"), float("nan"), 0, pre, pre

    with kfs.lock:
        post_full = kfs.T_WC[:N].clone()

    finite_ok = bool(torch.isfinite(post_full).all().item())
    post = post_full[:, 0, :3].detach().cpu().numpy().copy()
    deltas = ((post - pre) ** 2).sum(axis=1) ** 0.5
    max_d = float(deltas.max()) if deltas.size else 0.0
    mean_d = float(deltas.mean()) if deltas.size else 0.0
    over_cap = (
        deltas.size > 0
        and math.isfinite(max_jump_m)
        and max_d > max_jump_m
    )

    if (not finite_ok) or over_cap:
        reason = (
            "NaN/Inf in poses"
            if not finite_ok
            else f"max_delta={max_d:.1f}m > {max_jump_m:.1f}m cap"
        )
        with kfs.lock:
            kfs.T_WC[:N] = snapshot
        print(
            f"[coordinator] {label}: divergence guard tripped ({reason}); "
            f"reverted {N} poses to pre-solve snapshot"
        )
        return False, float("nan"), float("nan"), 0, pre, pre

    moved = int((deltas >= 0.02).sum())
    return True, max_d, mean_d, moved, pre, post


def run_coordinator(
    cfg,
    model,
    global_keyframes,    # MultiAgentKeyframes
    agent_to_coord_qs,   # list[mp.Queue]
    coord_to_agent_qs,   # list[mp.Queue]
    K,
    num_agents,
    device="cuda:0",
):
    set_global_config(cfg)
    torch.set_grad_enabled(False)

    shared_retrieval = SharedRetrievalDatabase(model, device=device)
    global_graph = GlobalFactorGraph(
        model, global_keyframes, K, device, num_agents
    )

    last_global_kf_per_agent: Dict[int, int] = {}
    agents_done = [False] * num_agents

    # Track agents' Sim(3) alignment components via union-find. Two agents
    # share a component iff their chains have been snapped to a common
    # frame. Agent 0 is the gauge anchor; the "global" component is the one
    # whose root is 0. When a cross-agent edge connects two agents:
    #   - same component   -> no snap (already aligned)
    #   - one is global    -> snap the OTHER agent's whole component into
    #                          the global frame (rigid Sim(3) over all KFs
    #                          of every agent in that component)
    #   - neither is global-> snap the cand side's component into agent_id's
    #                          component's frame; they remain non-global
    #                          but share a frame. When either later
    #                          connects to global, the snap drags the whole
    #                          merged block as one rigid body. This fixes
    #                          the 3+ agent case where an early (1,2) snap
    #                          used to permanently lock agent 2 into agent
    #                          1's *pre-global* frame.
    agent_component_parent: List[int] = list(range(num_agents))

    def _find(a: int) -> int:
        while agent_component_parent[a] != a:
            agent_component_parent[a] = agent_component_parent[
                agent_component_parent[a]
            ]
            a = agent_component_parent[a]
        return a

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        # Preserve component containing agent 0 (the global gauge) as root
        # so `_find(x) == 0` continues to mean "x is in the global frame".
        if rb == 0:
            ra, rb = rb, ra
        agent_component_parent[rb] = ra

    def _component_members(a: int) -> List[int]:
        root = _find(a)
        return [
            x for x in range(num_agents) if _find(x) == root
        ]

    poll_sleep = float(cfg.get("multi_agent", {}).get("poll_sleep", 0.01))
    inter_threshold = float(
        cfg.get("multi_agent", {}).get("inter_agent_min_match_frac", 0.1)
    )
    global_opt_interval = int(
        cfg.get("multi_agent", {}).get("global_opt_interval", 1)
    )
    # Sanity cap on per-KF translation change inside one global solve.
    # If exceeded (or any pose goes NaN/Inf), the divergence guard reverts the
    # solve so a single bad LLT step cannot corrupt every later iteration.
    # Set to "inf" / 0 / a negative value in YAML to disable.
    _raw_jump_cap = cfg.get("multi_agent", {}).get(
        "global_opt_max_translation_jump", 100.0
    )
    try:
        global_opt_max_jump = float(_raw_jump_cap)
    except (TypeError, ValueError):
        global_opt_max_jump = 100.0
    if global_opt_max_jump <= 0.0:
        global_opt_max_jump = math.inf

    inter_edges_at_last_solve = 0
    new_kf_msgs_pending: List[dict] = []

    # Diagnostic counters
    stats = {
        "kf_msgs_processed": 0,
        "retrieval_queries": 0,
        "cross_agent_candidates_total": 0,
        "inter_agent_attempts": 0,
        "inter_agent_accepted": 0,
        "global_solves": 0,
        "global_solves_reverted": 0,
    }

    def _process_message(msg: dict) -> bool:
        """Returns True if a new inter-agent edge was added by this message."""
        agent_id = msg["agent_id"]
        global_idx = msg["global_idx"]
        is_first = msg.get("is_first_kf_of_agent", False)

        stats["kf_msgs_processed"] += 1

        # The keyframe is already in the global shared store; read it back.
        with global_keyframes.keyframes.lock:
            frame = global_keyframes.keyframes[global_idx]

        # Intra-agent sequential edge in the global graph (skips for first KF
        # of an agent or until that agent has 2+ KFs in the global store).
        prev = last_global_kf_per_agent.get(agent_id)
        if prev is not None and prev != global_idx:
            added_intra = global_graph.add_intra_agent_factor(
                prev, global_idx, config["local_opt"]["min_match_frac"]
            )
            print(
                f"[coordinator] intra-agent edge agent={agent_id} "
                f"{prev}->{global_idx} added={added_intra}"
            )
        last_global_kf_per_agent[agent_id] = global_idx

        # Cross-agent retrieval
        added_inter = False
        stats["retrieval_queries"] += 1
        try:
            cross_candidates = shared_retrieval.update_and_query(
                frame,
                agent_id,
                global_idx,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
        except Exception as e:
            print(f"[coordinator] retrieval failed for agent {agent_id} kf {global_idx}: {e}")
            cross_candidates = []

        stats["cross_agent_candidates_total"] += len(cross_candidates)
        if cross_candidates:
            print(
                f"[coordinator] cross-agent candidates for agent {agent_id} "
                f"global_kf={global_idx}: {cross_candidates}"
            )
        for cand in cross_candidates:
            stats["inter_agent_attempts"] += 1
            ok = global_graph.add_inter_agent_factor(
                global_idx, cand, inter_threshold
            )
            if ok:
                added_inter = True
                stats["inter_agent_accepted"] += 1
                print(
                    f"[coordinator] inter-agent edge accepted: "
                    f"global_kf {global_idx} <-> {cand}"
                )
                # Rigid Sim(3) snap the first time two alignment components
                # are connected. This breaks the degenerate "both chains
                # overlap at the origin" initialization that makes the
                # Hessian rank-deficient and silently makes the CUDA LLT
                # solve a no-op (dx=0). Anchor is the side already in the
                # more-globally-grounded component, so the snap always
                # composes correctly through to agent 0's frame.
                with global_keyframes.keyframes.lock:
                    cand_agent = int(global_keyframes.keyframes.agent_id[cand])
                if agent_id != cand_agent:
                    ra = _find(agent_id)
                    rb = _find(cand_agent)
                    if ra != rb:
                        # Pick anchor: prefer the side whose component
                        # contains agent 0. If neither does, the choice is
                        # arbitrary and we anchor on agent_id; the merged
                        # block will be snapped to global later when this
                        # component first connects to agent 0's.
                        if rb == 0:
                            anchor_a, moving_a = cand_agent, agent_id
                            anchor_idx, moving_idx = cand, global_idx
                        else:
                            anchor_a, moving_a = agent_id, cand_agent
                            anchor_idx, moving_idx = global_idx, cand
                        moving_block = _component_members(moving_a)
                        a_ag, m_ag, n_corr = snap_agent_chain(
                            global_keyframes,
                            anchor_idx,
                            moving_idx,
                            global_graph=global_graph,
                            moving_agent_ids=moving_block,
                        )
                        _union(anchor_a, moving_a)
                        snap_kind = (
                            f"Umeyama Sim(3) over {n_corr} correspondences"
                            if n_corr > 0
                            else "crude T_a = T_b fallback (too few correspondences)"
                        )
                        into_global = _find(anchor_a) == 0
                        scope = (
                            f"moving block agents={moving_block} -> "
                            f"{'GLOBAL' if into_global else 'non-global'} frame "
                            f"(anchor component root={_find(anchor_a)})"
                        )
                        print(
                            f"[coordinator] chain snap: anchor agent {a_ag} "
                            f"kf={anchor_idx} <- moving agent {m_ag} kf={moving_idx} "
                            f"({snap_kind}); {scope}"
                        )
            else:
                print(
                    f"[coordinator] inter-agent edge REJECTED: "
                    f"global_kf {global_idx} <-> {cand} "
                    f"(match_frac below {inter_threshold:.3f})"
                )

        return added_inter

    print(f"[coordinator] running with {num_agents} agents")
    while True:
        progressed = False
        for agent_id in range(num_agents):
            q = agent_to_coord_qs[agent_id]
            while True:
                try:
                    msg = q.get_nowait()
                except Exception:
                    break
                progressed = True
                if msg.get("type") == "new_kf":
                    new_kf_msgs_pending.append(msg)
                elif msg.get("type") == "agent_done":
                    agents_done[msg["agent_id"]] = True
                    print(
                        f"[coordinator] agent {msg['agent_id']} finished its stream"
                    )

        # Process pending new-KF messages in global-index order so that the
        # retrieval DB inserts in the same order edges are added.
        new_kf_msgs_pending.sort(key=lambda m: m["global_idx"])
        new_inter_edges_this_pass = False
        while new_kf_msgs_pending:
            msg = new_kf_msgs_pending.pop(0)
            if _process_message(msg):
                new_inter_edges_this_pass = True

        n_inter = global_graph.num_inter_agent_edges()
        should_solve_global = (
            new_inter_edges_this_pass
            and (n_inter - inter_edges_at_last_solve) >= global_opt_interval
        )
        if should_solve_global:
            total_edges = global_graph.graph.ii.numel()
            print(
                f"[coordinator] running global Sim(3) Gauss-Newton: "
                f"inter-agent edges = {n_inter}, total edges = {total_edges}, "
                f"total KFs = {len(global_keyframes)}, "
                f"GPU res = {torch.cuda.memory_reserved() / (1 << 30):.1f}G "
                f"(graph {global_graph.graph.edge_memory_bytes() / (1 << 30):.1f}G)"
            )
            kfs = global_keyframes.keyframes
            ok, max_delta, mean_delta, moved, pre, post = (
                _run_global_solve_safely(
                    global_graph,
                    kfs,
                    config["use_calib"],
                    int(
                        cfg.get("multi_agent", {}).get(
                            "global_opt_max_iters", 10
                        )
                    ),
                    global_opt_max_jump,
                    "global GN",
                )
            )
            inter_edges_at_last_solve = n_inter
            stats["global_solves"] += 1
            if not ok:
                stats["global_solves_reverted"] += 1
                # Edges added in this pass remain in the graph; still publish
                # so viz reflects the latest connectivity. Skip pose_update
                # because poses didn't actually change (we reverted).
                _publish_edges_for_viz(global_keyframes, global_graph)
            else:
                N = pre.shape[0]
                print(
                    f"[coordinator] global GN finished: max_delta={max_delta:.3f}m "
                    f"mean_delta={mean_delta:.3f}m  KFs moved >0.02m: {moved}/{N}"
                )
                if moved > 0:
                    deltas = ((post - pre) ** 2).sum(axis=1) ** 0.5
                    for i in range(N):
                        if deltas[i] >= 0.02:
                            print(
                                f"  KF{i:2d} "
                                f"({pre[i,0]:+.2f},{pre[i,1]:+.2f},{pre[i,2]:+.2f}) -> "
                                f"({post[i,0]:+.2f},{post[i,1]:+.2f},{post[i,2]:+.2f}) "
                                f"d={deltas[i]:.3f}m"
                            )
                _publish_edges_for_viz(global_keyframes, global_graph)
                for q in coord_to_agent_qs:
                    q.put({"type": "pose_update"})
        elif new_inter_edges_this_pass or (
            len(global_keyframes) > 0
            and len(global_keyframes.global_edges_ii) == 0
        ):
            # Even when no global solve fires, surface the current edge set
            # so the visualizer can draw incoming intra-agent edges live.
            _publish_edges_for_viz(global_keyframes, global_graph)

        if all(agents_done) and not new_kf_msgs_pending:
            # All input streams exhausted; do a thorough final cleanup solve.
            # In-stream solves use a small iter budget so they don't stall the
            # pipeline; the final pass is allowed to chain several full GN
            # solves until the pose updates fall below `final_opt_delta_stop`.
            if global_graph.num_inter_agent_edges() > 0:
                final_iters = int(
                    cfg.get("multi_agent", {}).get("final_opt_max_iters", 100)
                )
                final_passes = int(
                    cfg.get("multi_agent", {}).get("final_opt_passes", 5)
                )
                final_stop = float(
                    cfg.get("multi_agent", {}).get("final_opt_delta_stop", 0.005)
                )
                print(
                    f"[coordinator] final global Sim(3) cleanup: up to "
                    f"{final_passes} passes of {final_iters} iters, "
                    f"stop when max_delta < {final_stop:.4f}m  "
                    f"(inter-agent edges={global_graph.num_inter_agent_edges()}, "
                    f"total edges={global_graph.graph.ii.numel()}, "
                    f"KFs={len(global_keyframes)})"
                )
                kfs = global_keyframes.keyframes
                for p in range(final_passes):
                    ok, max_d, mean_d, moved, pre, post = (
                        _run_global_solve_safely(
                            global_graph,
                            kfs,
                            config["use_calib"],
                            final_iters,
                            global_opt_max_jump,
                            f"final pass {p+1}/{final_passes}",
                        )
                    )
                    stats["global_solves"] += 1
                    if not ok:
                        stats["global_solves_reverted"] += 1
                        # Bad pass; bail out of the cleanup loop rather than
                        # keep firing solves on a divergent configuration.
                        break
                    N = pre.shape[0]
                    print(
                        f"  [final pass {p+1}/{final_passes}] "
                        f"max_delta={max_d:.4f}m mean_delta={mean_d:.4f}m  "
                        f"KFs moved >0.02m: {moved}/{N}"
                    )
                    _publish_edges_for_viz(global_keyframes, global_graph)
                    for q in coord_to_agent_qs:
                        q.put({"type": "pose_update"})
                    if max_d < final_stop:
                        print(
                            f"  converged: max_delta {max_d:.4f}m < {final_stop:.4f}m "
                            f"after pass {p+1}"
                        )
                        break
                else:
                    print(
                        f"  exhausted {final_passes} passes without converging "
                        f"below {final_stop:.4f}m"
                    )

                # After pose convergence, run K rounds of (depth, pose) to let
                # each side react to the other: depth_refinement rewrites X
                # along frozen rays, then a short pose pass absorbs the change
                # so poses stay self-consistent with the updated depths.
                # Gated by depth_refinement.enabled; skipped entirely if off.
                if cfg.get("depth_refinement", {}).get("enabled", False):
                    alt_rounds = int(
                        cfg.get("multi_agent", {}).get("final_alt_rounds", 2)
                    )
                    alt_pose_iters = int(
                        cfg.get("multi_agent", {}).get("final_alt_pose_iters", 30)
                    )
                    print(
                        f"[coordinator] final (depth, pose) alternation: "
                        f"{alt_rounds} rounds, pose_iters/round={alt_pose_iters}"
                    )
                    for r in range(alt_rounds):
                        try:
                            global_graph.graph.refine_depths()
                        except Exception as e:
                            print(
                                f"[coordinator] alt round {r+1}/{alt_rounds} "
                                f"depth refine failed: {e}"
                            )
                            break
                        ok, max_d, mean_d, moved, pre, post = (
                            _run_global_solve_safely(
                                global_graph,
                                kfs,
                                config["use_calib"],
                                alt_pose_iters,
                                global_opt_max_jump,
                                f"alt round {r+1}/{alt_rounds}",
                            )
                        )
                        stats["global_solves"] += 1
                        if not ok:
                            stats["global_solves_reverted"] += 1
                            break
                        N = pre.shape[0]
                        print(
                            f"  [alt round {r+1}/{alt_rounds}] "
                            f"pose max_delta={max_d:.4f}m "
                            f"mean_delta={mean_d:.4f}m  "
                            f"KFs moved >0.02m: {moved}/{N}"
                        )
                        _publish_edges_for_viz(global_keyframes, global_graph)
                        for q in coord_to_agent_qs:
                            q.put({"type": "pose_update"})

            print(
                "[coordinator] === diagnostic summary ===\n"
                f"  KF messages processed   : {stats['kf_msgs_processed']}\n"
                f"  retrieval queries       : {stats['retrieval_queries']}\n"
                f"  cross-agent candidates  : {stats['cross_agent_candidates_total']}\n"
                f"  inter-agent attempts    : {stats['inter_agent_attempts']}\n"
                f"  inter-agent accepted    : {stats['inter_agent_accepted']}\n"
                f"  inter-agent edges total : {global_graph.num_inter_agent_edges()}\n"
                f"  total graph edges       : {global_graph.graph.ii.numel()}\n"
                f"  global solves run       : {stats['global_solves']}\n"
                f"  global solves reverted  : {stats['global_solves_reverted']}\n"
                f"  retrieval DB size       : {len(shared_retrieval.agent_id_per_entry)}\n"
                "==========================="
            )
            for q in coord_to_agent_qs:
                q.put({"type": "terminate"})
            print("[coordinator] all agents done, exiting")
            return

        if not progressed:
            time.sleep(poll_sleep)
