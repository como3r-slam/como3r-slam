"""
Multi-agent extension for CoMo3R-SLAM.

Contains three coordinator-side abstractions:
- MultiAgentKeyframes: wraps a single SharedKeyframes that stores the union
  of keyframes from every agent in one global index space. Each entry is
  tagged with its agent_id.
- SharedRetrievalDatabase: a coordinator-owned ASMK database that filters
  query results to keyframes from OTHER agents (cross-agent loop closure).
- GlobalFactorGraph: a thin wrapper around the existing FactorGraph that
  tracks which edges are intra- vs inter-agent and runs the Level-2 GN.

The CUDA Gauss-Newton solver is edge-topology agnostic: an inter-agent edge
contributes the same 7x7 Hessian blocks as any other Sim(3) edge, so we
reuse FactorGraph as-is here.
"""
from typing import List, Optional, Tuple

import lietorch
import torch

from como3r_slam.frame import Frame, SharedKeyframes
from como3r_slam.global_opt import FactorGraph
from como3r_slam.mast3r_utils import load_retriever


def _umeyama_sim3(X: torch.Tensor, Y: torch.Tensor):
    """Closed-form Sim(3) alignment via Umeyama 1991.

    Find (s, R, t) minimizing  sum_k || X[k] - (s R Y[k] + t) ||^2.
    Returns (s: scalar, R: (3,3), t: (3,)) all on X.device.
    """
    assert X.shape == Y.shape and X.shape[-1] == 3 and X.dim() == 2
    n = X.shape[0]
    mean_X = X.mean(dim=0)
    mean_Y = Y.mean(dim=0)
    Xc = X - mean_X
    Yc = Y - mean_Y
    sigma_Y = (Yc * Yc).sum() / n
    cov = (Xc.T @ Yc) / n
    # SVD
    U, D, Vt = torch.linalg.svd(cov)
    # Ensure proper rotation (det = +1) via Umeyama's S diagonal.
    det_uvt = torch.linalg.det(U @ Vt)
    S_diag = torch.tensor([1.0, 1.0, 1.0], device=X.device, dtype=X.dtype)
    if det_uvt < 0:
        S_diag[2] = -1.0
    R = U @ torch.diag(S_diag) @ Vt
    s = (D * S_diag).sum() / (sigma_Y + 1e-12)
    t = mean_X - s * (R @ mean_Y)
    return s, R, t


def _rotmat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """Convert (3,3) rotation matrix to xyzw quaternion. Uses scipy for
    numerical robustness (Shepperd's algorithm with branch selection)."""
    from scipy.spatial.transform import Rotation as _Rsc

    R_np = R.detach().cpu().numpy()
    q = _Rsc.from_matrix(R_np).as_quat()  # xyzw
    return torch.from_numpy(q).to(R.device).to(R.dtype)


def _sim3_from_components(s, R, t, device, dtype) -> "lietorch.Sim3":
    """Build a lietorch.Sim3 from (scale, R, t)."""
    q = _rotmat_to_quat_xyzw(R)
    data = torch.cat(
        [
            t.reshape(3).to(device=device, dtype=dtype),
            q.reshape(4).to(device=device, dtype=dtype),
            s.reshape(1).to(device=device, dtype=dtype),
        ],
        dim=-1,
    ).unsqueeze(0)  # (1, 8)
    return lietorch.Sim3(data)


def snap_agent_chain(
    global_keyframes,
    anchor_global_idx: int,
    moving_global_idx: int,
    global_graph=None,
    q_thresh: float = 1.5,
    min_pts: int = 64,
    moving_agent_ids: Optional[List[int]] = None,
) -> tuple:
    """Procrustes-style Sim(3) snap of one agent's whole chain into the
    other's frame, using MASt3R pixel correspondences from the cross-agent
    edge that just connected them.

    Why this is needed: each agent's local chain is built in its own
    arbitrary frame (first KF at Identity). Without this snap, the global
    GN sees a degenerate initialization where intra-agent residuals are
    near zero and the inter-agent residual is also near zero by
    coincidence -- the Hessian becomes ill-conditioned and the CUDA
    Cholesky solve silently returns dx=0. By transforming the moving
    agent's chain into the anchor's frame via the actual measured
    correspondences (instead of just identity), we put the system in a
    well-conditioned initial state.

    If `global_graph` is provided (default for multi-agent), we use the
    correspondences from the just-added inter-agent edge between the two
    KFs and solve Umeyama Sim(3) over those matched 3D points. If the
    edge isn't found or has too few high-Q correspondences (< `min_pts`),
    we fall back to the crude "pose A == pose B" snap.

    `moving_agent_ids`: when set, applies the computed Sim(3) to ALL KFs
    of ALL listed agents as one rigid block. This is required when the
    "moving" side belongs to a multi-agent alignment component that
    shares a common frame -- e.g. when agents 1 and 2 were already
    snapped together before either touched the global frame, and now
    the (0,1) edge needs to drag agent 2 along with agent 1. Defaults
    to `[moving_agent]` (i.e. the agent owning `moving_global_idx`),
    matching the 2-agent behavior.

    Returns (anchor_agent, moving_agent, n_correspondences_used_or_0).
    """
    kfs = global_keyframes.keyframes
    with kfs.lock:
        anchor_agent = int(kfs.agent_id[anchor_global_idx])
        moving_agent = int(kfs.agent_id[moving_global_idx])
        T_anchor = lietorch.Sim3(kfs.T_WC[anchor_global_idx].clone())
        T_moving = lietorch.Sim3(kfs.T_WC[moving_global_idx].clone())
        X_canon_anchor = kfs.X[anchor_global_idx].clone()  # (HW, 3)
        X_canon_moving = kfs.X[moving_global_idx].clone()

    align_S = None
    n_used = 0

    if global_graph is not None:
        graph = global_graph.graph
        # Locate the inter-agent edge that connects these two KFs. We
        # search backwards so we hit the most recently added one.
        ii_cpu = graph.ii.cpu().tolist()
        jj_cpu = graph.jj.cpu().tolist()
        edge_idx = None
        anchor_is_i = True
        for k in range(len(ii_cpu) - 1, -1, -1):
            i_k, j_k = ii_cpu[k], jj_cpu[k]
            if i_k == anchor_global_idx and j_k == moving_global_idx:
                edge_idx, anchor_is_i = k, True
                break
            if i_k == moving_global_idx and j_k == anchor_global_idx:
                edge_idx, anchor_is_i = k, False
                break

        if edge_idx is not None:
            # Pick the i→j direction that maps anchor→moving so that
            # idx[p in anchor] = q in moving.
            if anchor_is_i:
                idx_a2m = graph.idx_ii2jj[edge_idx].reshape(-1).long()
                valid = graph.valid_match_j[edge_idx].reshape(-1).bool()
                Q = graph.Q_ii2jj[edge_idx].reshape(-1)
            else:
                idx_a2m = graph.idx_jj2ii[edge_idx].reshape(-1).long()
                valid = graph.valid_match_i[edge_idx].reshape(-1).bool()
                Q = graph.Q_jj2ii[edge_idx].reshape(-1)
            mask = valid & (Q > q_thresh)
            n_used = int(mask.sum().item())

            if n_used >= min_pts:
                p_idx = mask.nonzero(as_tuple=True)[0]
                q_idx = idx_a2m[p_idx]
                X_a_local = X_canon_anchor[p_idx]
                X_m_local = X_canon_moving[q_idx]

                # Lift to world frame using current poses.
                M_a = T_anchor.matrix().squeeze(0)
                M_m = T_moving.matrix().squeeze(0)
                ones = torch.ones_like(X_a_local[..., :1])
                X_a_w = (torch.cat([X_a_local, ones], dim=-1) @ M_a.T)[..., :3]
                X_m_w = (torch.cat([X_m_local, ones], dim=-1) @ M_m.T)[..., :3]

                # Umeyama: find S so that S * X_m_w ≈ X_a_w.
                s, R, t = _umeyama_sim3(X_a_w, X_m_w)
                align_S = _sim3_from_components(
                    s, R, t, device=kfs.T_WC.device, dtype=kfs.T_WC.dtype
                )

    if align_S is None:
        # Fallback: snap moving KF's pose to equal anchor KF's pose.
        align_S = T_anchor * T_moving.inv()

    if moving_agent_ids is None:
        moving_agent_ids = [moving_agent]
    moving_indices: List[int] = []
    for aid in moving_agent_ids:
        moving_indices.extend(global_keyframes.get_local_to_global_map(aid))
    with kfs.lock:
        for idx in moving_indices:
            T_curr = lietorch.Sim3(kfs.T_WC[idx].clone())
            T_new = align_S * T_curr
            kfs.T_WC[idx] = T_new.data
    return anchor_agent, moving_agent, n_used


class MultiAgentKeyframes:
    """Unified keyframe store across all agents.

    The underlying SharedKeyframes holds a flat list of keyframes; the
    FactorGraph sees a single integer index space and does not need to
    know which agent produced any given keyframe.
    """

    def __init__(self, manager, h, w, num_agents, buffer=512, device="cuda"):
        # Cross-process store: agents, coordinator and the visualizer all hold
        # the tensors allocated here, so it cannot grow later -- `buffer` must
        # cover the total keyframe count across all agents.
        self.keyframes = SharedKeyframes(
            manager,
            h,
            w,
            buffer=buffer,
            device=device,
            name="global",
            buffer_hint=(
                "`multi_agent.global_kf_buffer` in the config YAML "
                "(or pass --global-kf-buffer N)"
            ),
        )
        self.num_agents = num_agents
        # number of keyframes contributed per agent (for fast counts)
        self.agent_kf_counts = manager.list([0] * num_agents)
        self.lock = manager.RLock()
        # Visualization-facing snapshot of the coordinator's global factor graph
        # edges (in global keyframe-index space). Written by the coordinator
        # after each Level-2 solve and read by the multi-agent visualizer.
        self.global_edges_ii = manager.list()
        self.global_edges_jj = manager.list()

    def append(self, frame: Frame, agent_id: int) -> int:
        """Append a keyframe tagged with its source agent. Returns global idx."""
        with self.lock:
            global_idx = self.keyframes.append(frame, agent_id=agent_id)
            self.agent_kf_counts[agent_id] = int(self.agent_kf_counts[agent_id]) + 1
            return global_idx

    def get_local_to_global_map(self, agent_id: int) -> List[int]:
        """Indices in the global store belonging to `agent_id`, in append order."""
        return self.keyframes.get_agent_keyframe_indices(agent_id)

    def __len__(self):
        return len(self.keyframes)


class SharedRetrievalDatabase:
    """ASMK retrieval database used by the coordinator for cross-agent matches.

    Each entry is tagged with an agent_id at insertion time so that queries
    can be filtered to candidates from OTHER agents (cross-agent loop closures
    only). Intra-agent loop closures are handled by each agent's local DB.

    The retrieval DB returns insertion-ordered IVF indices, which need not
    match global keyframe indices (the coordinator may process keyframes
    out of order). We keep an explicit mapping from DB position to global
    keyframe index.
    """

    def __init__(self, model, device="cuda"):
        self.retrieval_db = load_retriever(model, device=device)
        self.agent_id_per_entry: List[int] = []
        self.global_idx_per_entry: List[int] = []

    def update_and_query(
        self,
        frame: Frame,
        agent_id: int,
        global_idx: int,
        k: int,
        min_thresh: float,
    ) -> List[int]:
        """Add the frame's features and return cross-agent global KF indices."""
        db_size_before = len(self.agent_id_per_entry)
        # Histogram of agents already in the DB (helps diagnose "no cross-agent
        # candidates because the DB still only contains my own agent").
        agent_counts = {}
        for aid in self.agent_id_per_entry:
            agent_counts[aid] = agent_counts.get(aid, 0) + 1

        candidates = self.retrieval_db.update(
            frame, add_after_query=True, k=k, min_thresh=min_thresh
        )
        raw_agents = [int(self.agent_id_per_entry[idx]) for idx in candidates]
        cross_global = [
            self.global_idx_per_entry[idx]
            for idx in candidates
            if self.agent_id_per_entry[idx] != agent_id
        ]
        same_agent_dropped = len(candidates) - len(cross_global)
        print(
            f"[coordinator] retrieval agent={agent_id} global_kf={global_idx} "
            f"db_size_before={db_size_before} db_agents={agent_counts} "
            f"raw_candidates={list(candidates)} raw_candidate_agents={raw_agents} "
            f"cross_agent={len(cross_global)} same_agent_dropped={same_agent_dropped} "
            f"(k={k}, min_thresh={min_thresh})"
        )

        self.agent_id_per_entry.append(agent_id)
        self.global_idx_per_entry.append(global_idx)
        return cross_global


class GlobalFactorGraph:
    """Coordinator-side factor graph over all agents.

    Holds the global FactorGraph (operating on MultiAgentKeyframes' underlying
    SharedKeyframes) plus bookkeeping of which edges crossed agent boundaries.
    """

    def __init__(self, model, global_keyframes: MultiAgentKeyframes, K, device, num_agents):
        self.global_keyframes = global_keyframes
        self.graph = FactorGraph(model, global_keyframes.keyframes, K, device)
        self.inter_agent_edges: List[Tuple[int, int]] = []
        self.num_agents = num_agents

    def add_intra_agent_factor(self, prev_global_idx: int, new_global_idx: int, min_match_frac: float) -> bool:
        """Sequential edge between two consecutive keyframes of the same agent."""
        return bool(
            self.graph.add_factors(
                [prev_global_idx], [new_global_idx], min_match_frac
            )
        )

    def add_intra_agent_loop(self, new_global_idx: int, neighbor_global_indices: List[int], min_match_frac: float) -> bool:
        """Intra-agent loop-closure edges (multiple neighbors → new KF)."""
        if not neighbor_global_indices:
            return False
        frame_indices = [new_global_idx] * len(neighbor_global_indices)
        return bool(
            self.graph.add_factors(
                neighbor_global_indices, frame_indices, min_match_frac
            )
        )

    def add_inter_agent_factor(self, src_global_idx: int, candidate_global_idx: int, min_match_frac: float) -> bool:
        """Edge between keyframes from different agents (cross-agent loop closure)."""
        ii = [min(src_global_idx, candidate_global_idx)]
        jj = [max(src_global_idx, candidate_global_idx)]
        before = self.graph.ii.numel()
        success = bool(
            self.graph.add_factors(
                ii,
                jj,
                min_match_frac,
                log_prefix="[coordinator] inter-agent",
                # Inter-agent edges must clear the match_frac bar; the
                # "consecutive global index" exemption only makes sense for
                # temporally consecutive intra-agent KFs.
                allow_consecutive_exemption=False,
            )
        )
        # Detect if any new edge actually got accepted; record it.
        if success and self.graph.ii.numel() > before:
            self.inter_agent_edges.append((ii[0], jj[0]))
        return success

    def solve_global(self, use_calib: bool, max_iters: int = None) -> None:
        """Level-2 Gauss-Newton over all agents' Sim(3) poses jointly.

        Inter-agent edges look identical to loop-closure edges from the
        solver's perspective; gauge is fixed by pinning the first KF
        (which is whichever KF holds global index 0).

        `max_iters` (if provided) overrides the FactorGraph's local_opt
        max_iters for the global solve. The global problem is bigger and
        usually wants more iterations than the per-agent backend.
        """
        if use_calib:
            self.graph.solve_GN_calib(override_max_iters=max_iters)
        else:
            self.graph.solve_GN_rays(override_max_iters=max_iters)

    def num_inter_agent_edges(self) -> int:
        return len(self.inter_agent_edges)
