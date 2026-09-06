"""Per-agent process for multi-agent CoMo3R-SLAM.

Each agent owns its own local SharedKeyframes + FactorGraph (Level-1 / fast,
high-frequency). When a new keyframe is added, the agent also appends a copy
into the globally shared MultiAgentKeyframes and notifies the coordinator,
which runs Level-2 (slow, low-frequency) cross-agent optimization.

After a global solve, the coordinator notifies agents to pull updated poses
from the global store back into their local store so subsequent tracking is
expressed in the global frame.
"""
import pathlib
import time

import lietorch
import numpy as np
import torch

from como3r_slam.config import config, set_global_config
from como3r_slam.dataloader import Intrinsics, load_dataset
from como3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from como3r_slam.global_opt import FactorGraph
from como3r_slam.mast3r_utils import (
    load_retriever,
    mast3r_inference_mono,
)
from como3r_slam.tracker import FrameTracker


def _relocalization(frame, keyframes, factor_graph, retrieval_database):
    """Per-agent relocalization against this agent's own keyframes."""
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful = False
        if kf_idx:
            keyframes.append(frame, agent_id=-1)  # local store; agent_id unused here
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                successful = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
        if successful:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful


def _propagate_global_to_local(
    agent_id, global_keyframes, local_keyframes, local_to_global
):
    """Copy globally optimized T_WC poses from global store into local store.

    `local_to_global[i]` is the global KF index for local KF index `i`.
    Called after the coordinator runs a global Sim(3) Gauss-Newton.
    """
    if not local_to_global:
        return
    with global_keyframes.keyframes.lock, local_keyframes.lock:
        for local_idx, global_idx in enumerate(local_to_global):
            if local_idx >= len(local_keyframes):
                break
            local_keyframes.T_WC[local_idx] = (
                global_keyframes.keyframes.T_WC[global_idx].clone()
            )


def _copy_local_kf_to_global(local_keyframes, local_idx, global_keyframes, agent_id):
    """Append the keyframe at local index into the global store, return global idx."""
    with local_keyframes.lock:
        kf = local_keyframes[local_idx]
        # Detach references so the global store gets its own copies.
        kf.img = kf.img.clone()
        kf.uimg = kf.uimg.clone()
        kf.img_shape = kf.img_shape.clone()
        kf.img_true_shape = kf.img_true_shape.clone()
        kf.T_WC = lietorch.Sim3(kf.T_WC.data.clone())
        kf.X_canon = kf.X_canon.clone()
        # Carry the local store's pristine MASt3R depth snapshot over to the
        # global store. By the time copy happens, kf.X_canon has already been
        # refined by the local FactorGraph's GN pass, so re-snapshotting from
        # X_canon at the global side would lock in the (already-refined) state
        # instead of MASt3R's original smooth prediction.
        kf.depth_init = kf.depth_init.clone() if kf.depth_init is not None else None
        kf.C = kf.C.clone()
        kf.feat = kf.feat.clone()
        kf.pos = kf.pos.clone()
    return global_keyframes.append(kf, agent_id=agent_id)


def run_agent(
    agent_id,
    cfg,
    model,
    dataset_path,
    calib_intrinsics,  # dict with width/height/calibration; may be None
    global_keyframes,  # MultiAgentKeyframes
    shared_states,     # SharedStates allocated in the main process (for viz access)
    agent_to_coord_q,
    coord_to_agent_q,
    agent0_inited=None,
    device="cuda:0",
    max_frames=-1,
    per_frame_save_path=None,
):
    set_global_config(cfg)
    torch.set_grad_enabled(False)

    import torch.multiprocessing as mp

    manager = mp.Manager()

    dataset = load_dataset(dataset_path)
    dataset.subsample(config["dataset"]["subsample"])
    # Raw image (W, H) drive the cv2 undistort remap; using the YAML's W/H
    # (which may be from a different sensor) causes remap output to mismatch
    # `get_img_shape()` and breaks the SharedKeyframes buffer allocation.
    raw_h, raw_w = dataset.get_img_shape()[1]

    K = None
    if config["use_calib"]:
        if calib_intrinsics is None:
            raise RuntimeError(
                f"agent {agent_id}: use_calib=True but no intrinsics supplied"
            )
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            raw_w,
            raw_h,
            calib_intrinsics["calibration"],
        )
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
    h, w = dataset.get_img_shape()[0]

    # The local store never leaves this process, so it can grow on demand: how
    # many keyframes an agent ends up with is not knowable up front. The config
    # value is only the initial allocation.
    local_keyframes = SharedKeyframes(
        manager,
        h,
        w,
        buffer=int(config.get("multi_agent", {}).get("local_kf_buffer", 256)),
        device=device,
        can_grow=True,
        name=f"agent{agent_id}-local",
    )
    if K is not None:
        local_keyframes.set_intrinsics(K)
    # External shared states (created in main) so the multi-agent visualizer
    # can read the current frame for every agent.
    local_states = shared_states
    local_states.set_mode(Mode.INIT)

    tracker = FrameTracker(model, local_keyframes, device)
    local_graph = FactorGraph(model, local_keyframes, K, device)
    local_retrieval = load_retriever(model, device=device)

    # local index -> global index
    local_to_global: list = []

    # Per-frame trajectory records for the "full" (non-KF) trajectory. Each
    # entry is (frame_id, anchor_global_kf_idx, T_rel_data_8). T_rel is the
    # frame's pose relative to its anchor KF at tracking time, which is
    # frame-invariant so the main process can compose it with the anchor's
    # FINAL global pose (post all Level-2 solves) at save time.
    per_frame_records: list = []

    def _drain_coordinator():
        """Process any pose-update messages sent by the coordinator."""
        while True:
            try:
                msg = coord_to_agent_q.get_nowait()
            except Exception:
                break
            if msg is None:
                return "terminate"
            if msg.get("type") == "pose_update":
                _propagate_global_to_local(
                    agent_id, global_keyframes, local_keyframes, local_to_global
                )
            elif msg.get("type") == "terminate":
                return "terminate"
        return None

    if agent0_inited is not None and agent_id != 0:
        # Wait for agent 0 to claim global KF index 0 (gauge anchor).
        agent0_inited.wait()

    fps_timer = time.time()
    n = len(dataset)
    if max_frames > 0:
        n = min(n, max_frames)
    i = 0
    while i < n:
        if _drain_coordinator() == "terminate":
            break

        timestamp, img = dataset[i]
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if len(local_keyframes) == 0
            else local_states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        mode = local_states.get_mode()

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            local_keyframes.append(frame, agent_id=agent_id)
            local_kf_idx = len(local_keyframes) - 1
            # Register the init KF in the retrieval DB so its internal
            # kf_counter stays aligned with local_keyframes indices. Without
            # this, every later retrieval result is local_kf_idx - 1, which
            # makes reloc's strict factor-graph edges match against the wrong
            # keyframes and reloc gets stuck.
            local_retrieval.update(
                frame,
                add_after_query=True,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            global_idx = _copy_local_kf_to_global(
                local_keyframes, local_kf_idx, global_keyframes, agent_id
            )
            local_to_global.append(global_idx)
            agent_to_coord_q.put(
                {
                    "type": "new_kf",
                    "agent_id": agent_id,
                    "global_idx": global_idx,
                    "is_first_kf_of_agent": True,
                }
            )
            if agent_id == 0 and agent0_inited is not None:
                agent0_inited.set()
            local_states.set_mode(Mode.TRACKING)
            local_states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, _match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                local_states.set_mode(Mode.RELOC)
            local_states.set_frame(frame)

            # Record per-frame pose relative to current anchor KF (the last
            # local KF used by the tracker). Only valid if tracker did not
            # request reloc; skipped frames have no usable pose.
            if not try_reloc and not add_new_kf and len(local_keyframes) > 0:
                anchor_local_idx = len(local_keyframes) - 1
                anchor_global_idx = local_to_global[anchor_local_idx]
                with local_keyframes.lock:
                    T_anchor = lietorch.Sim3(
                        local_keyframes.T_WC[anchor_local_idx].clone()
                    )
                T_rel = T_anchor.inv() * frame.T_WC
                per_frame_records.append(
                    (
                        int(frame.frame_id),
                        int(anchor_global_idx),
                        T_rel.data.detach().cpu().numpy().reshape(-1).astype(np.float32),
                    )
                )

            if add_new_kf:
                local_keyframes.append(frame, agent_id=agent_id)
                local_kf_idx = len(local_keyframes) - 1

                # Level-1 (intra-agent): sequential edge + local retrieval
                local_graph.add_factors(
                    [local_kf_idx - 1],
                    [local_kf_idx],
                    config["local_opt"]["min_match_frac"],
                )
                loop_inds = local_retrieval.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                loop_inds = [int(x) for x in loop_inds if x != local_kf_idx - 1 and x != local_kf_idx]
                if loop_inds:
                    local_graph.add_factors(
                        loop_inds,
                        [local_kf_idx] * len(loop_inds),
                        config["local_opt"]["min_match_frac"],
                    )

                if config["use_calib"]:
                    local_graph.solve_GN_calib()
                else:
                    local_graph.solve_GN_rays()

                global_idx = _copy_local_kf_to_global(
                    local_keyframes, local_kf_idx, global_keyframes, agent_id
                )
                local_to_global.append(global_idx)

                agent_to_coord_q.put(
                    {
                        "type": "new_kf",
                        "agent_id": agent_id,
                        "global_idx": global_idx,
                        "is_first_kf_of_agent": False,
                    }
                )

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            local_states.set_frame(frame)
            success = _relocalization(
                frame, local_keyframes, local_graph, local_retrieval
            )
            if success:
                local_states.set_mode(Mode.TRACKING)
                # We popped/relocalized; treat last local KF as new global KF too.
                local_kf_idx = len(local_keyframes) - 1
                global_idx = _copy_local_kf_to_global(
                    local_keyframes, local_kf_idx, global_keyframes, agent_id
                )
                local_to_global.append(global_idx)
                agent_to_coord_q.put(
                    {
                        "type": "new_kf",
                        "agent_id": agent_id,
                        "global_idx": global_idx,
                        "is_first_kf_of_agent": False,
                    }
                )
        else:
            raise RuntimeError(f"agent {agent_id}: invalid mode {mode}")

        if i % 30 == 0 and i > 0:
            fps = i / max(time.time() - fps_timer, 1e-6)
            # GPU accounting: `res` is what this process holds from the driver,
            # which is what actually competes with the other processes on the
            # card. Watch it to size the run before it OOMs.
            res = torch.cuda.memory_reserved() / (1 << 30)
            store_gb = (
                len(local_keyframes) * local_keyframes.slot_bytes() / (1 << 30)
            )
            graph_gb = local_graph.edge_memory_bytes() / (1 << 30)
            print(
                f"[agent {agent_id}] frame {i}/{n}  FPS={fps:.2f}  "
                f"KFs={len(local_keyframes)}  GPU res={res:.1f}G "
                f"(kf store {store_gb:.1f}G, graph {graph_gb:.1f}G)"
            )

        i += 1

    # Signal the coordinator that this agent has finished its stream.
    agent_to_coord_q.put({"type": "agent_done", "agent_id": agent_id})
    print(f"[agent {agent_id}] done ({len(local_keyframes)} keyframes)")

    # Final pose pull-back so saved trajectories reflect the latest global solve.
    while True:
        msg = _drain_coordinator()
        if msg == "terminate":
            break
        # Wait for an explicit terminate from the coordinator.
        time.sleep(0.05)

    # Persist per-frame relative-pose records so main can compose them with
    # the final global KF poses for ATE on the full trajectory. A ``.ready``
    # sentinel is touched AFTER np.savez closes the file, so the multi-agent
    # visualizer (running concurrently) can poll for sentinels and avoid
    # racing a half-written npz.
    if per_frame_save_path is not None:
        out = pathlib.Path(per_frame_save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if per_frame_records:
            frame_ids = np.array([r[0] for r in per_frame_records], dtype=np.int64)
            anchor_idx = np.array([r[1] for r in per_frame_records], dtype=np.int64)
            T_rel = np.stack([r[2] for r in per_frame_records], axis=0)  # (M, 8)
        else:
            frame_ids = np.zeros((0,), dtype=np.int64)
            anchor_idx = np.zeros((0,), dtype=np.int64)
            T_rel = np.zeros((0, 8), dtype=np.float32)
        np.savez(out, frame_ids=frame_ids, anchor_global_idx=anchor_idx, T_rel=T_rel)
        (out.parent / f"{out.stem}.ready").touch()
        print(f"[agent {agent_id}] saved {len(per_frame_records)} per-frame records -> {out}")
