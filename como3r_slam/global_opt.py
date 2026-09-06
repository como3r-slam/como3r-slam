import lietorch
import torch
from como3r_slam.config import config
from como3r_slam.depth_refinement import build_segments
from como3r_slam.frame import SharedKeyframes
from como3r_slam.geometry import (
    constrain_points_to_ray,
)
from como3r_slam.mast3r_utils import mast3r_match_symmetric
import como3r_slam_backends


class FactorGraph:
    # Per-edge match data is (E, P) with P = h*w pixels, i.e. ~5 MB per edge at
    # 512x384. Edges are only ever appended, so growing with `torch.cat` on
    # every keyframe would reallocate and copy the entire block each time --
    # O(E^2) copying, and a brand-new block size per add, which fragments the
    # CUDA caching allocator until a long sequence OOMs. Instead we hold
    # capacity buffers that grow in fixed chunks and expose the live prefix as
    # views (contiguous, so the CUDA backends take them unchanged).
    EDGE_CHUNK = 16
    _EDGE_DTYPES = {
        "ii": torch.long,
        "jj": torch.long,
        "idx_ii2jj": torch.long,
        "idx_jj2ii": torch.long,
        "valid_match_j": torch.bool,
        "valid_match_i": torch.bool,
        "Q_ii2jj": torch.float32,
        "Q_jj2ii": torch.float32,
    }

    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self._n_edges = 0
        self._edge_capacity = 0
        self._edge_bufs = {}
        self._empty_edges = {
            name: torch.as_tensor([], dtype=dtype, device=self.device)
            for name, dtype in self._EDGE_DTYPES.items()
        }
        self.window_size = self.cfg["window_size"]

        self.K = K

    def _edge_view(self, name):
        buf = self._edge_bufs.get(name)
        if buf is None:
            return self._empty_edges[name]
        return buf[: self._n_edges]

    # Read-only views over the live prefix of each edge buffer.
    ii = property(lambda self: self._edge_view("ii"))
    jj = property(lambda self: self._edge_view("jj"))
    idx_ii2jj = property(lambda self: self._edge_view("idx_ii2jj"))
    idx_jj2ii = property(lambda self: self._edge_view("idx_jj2ii"))
    valid_match_j = property(lambda self: self._edge_view("valid_match_j"))
    valid_match_i = property(lambda self: self._edge_view("valid_match_i"))
    Q_ii2jj = property(lambda self: self._edge_view("Q_ii2jj"))
    Q_jj2ii = property(lambda self: self._edge_view("Q_jj2ii"))

    def _append_edges(self, new_edges: dict):
        """Append a batch of edges, growing the capacity buffers in chunks."""
        n_new = new_edges["ii"].shape[0]
        if n_new == 0:
            return
        need = self._n_edges + n_new
        if need > self._edge_capacity:
            new_capacity = max(need, self._edge_capacity + self.EDGE_CHUNK)
            for name, value in new_edges.items():
                buf = torch.empty(
                    (new_capacity, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                old = self._edge_bufs.get(name)
                if old is not None:
                    buf[: self._n_edges] = old[: self._n_edges]
                self._edge_bufs[name] = buf
            self._edge_capacity = new_capacity
        for name, value in new_edges.items():
            self._edge_bufs[name][self._n_edges : need] = value
        self._n_edges = need

    def edge_memory_bytes(self):
        return sum(b.element_size() * b.numel() for b in self._edge_bufs.values())

    def add_factors(
        self,
        ii,
        jj,
        min_match_frac,
        is_reloc=False,
        log_prefix=None,
        allow_consecutive_exemption=True,
    ):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        if allow_consecutive_exemption:
            # Force-accept temporally consecutive intra-agent edges even when
            # MASt3R matching is weak (e.g. fast motion / motion blur).
            # Disabled for inter-agent edges because consecutive global
            # indices there just mean two agents interleaved their appends,
            # not that the views overlap.
            consecutive_edges = ii_tensor == (jj_tensor - 1)
            invalid_edges = (~consecutive_edges) & invalid_edges

        if log_prefix is not None:
            mfi = match_frac_i.detach().cpu().tolist()
            mfj = match_frac_j.detach().cpu().tolist()
            for k in range(ii_tensor.shape[0]):
                accepted = not bool(invalid_edges[k].item())
                print(
                    f"{log_prefix} edge {int(ii_tensor[k].item())}<->"
                    f"{int(jj_tensor[k].item())} "
                    f"match_frac_i={mfi[k]:.4f} match_frac_j={mfj[k]:.4f} "
                    f"min(min_frac)={min(mfi[k], mfj[k]):.4f} thr={min_match_frac:.4f} "
                    f"-> {'ACCEPT' if accepted else 'REJECT'}"
                )

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self._append_edges(
            {
                "ii": ii_tensor,
                "jj": jj_tensor,
                "idx_ii2jj": idx_i2j,
                "idx_jj2ii": idx_j2i,
                "valid_match_j": valid_match_j,
                "valid_match_i": valid_match_i,
                "Q_ii2jj": Qj,
                "Q_jj2ii": Qi,
            }
        )

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def refine_depths(self, num_iters: int = None):
        """Super-primitive depth refinement along frozen MASt3R rays.

        Called once at the end of the run (final global cleanup), after the
        pose solve has converged. Skipped entirely during streaming.

        The keyframe pointmap is reparameterized as

            d_p = d_init_p * exp(s_{seg(p)})

        where each oversegment owns ONE log-scale ``s`` and ``d_init`` is
        the pristine per-KF MASt3R snapshot. The Hessian shrinks from N*P
        (per pixel) to M (per segment, typically a few hundred per KF), so
        the system is small, well-conditioned, and segment-coherent by
        construction. The per-pixel ray basis stays frozen at the call's
        starting X (K-consistent in calib mode).

        Smoothness across segment boundaries is added as a 1D residual per
        unique adjacent-segment pair (representative ``d_a, d_b`` from the
        boundary pixels). This is the only term that couples segments to
        each other; without it, neighboring segments could drift to
        independent scales and produce visible steps at boundaries.

        Pinned keyframes are the gauge anchor for Sim(3); their segments
        must NOT be updated. The CUDA solver skips them (see
        ``seg_update_kernel`` gate ``seg_kf[m] < num_fix``); we additionally
        slice ``[pin:]`` before writing X back.
        """
        dr_cfg = config.get("depth_refinement", {})
        if not dr_cfg.get("enabled", False):
            return
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique = unique_kf_idx.numel()
        if n_unique <= pin:
            return

        # Gather per-keyframe tensors (T_WCs and Cs include pinned KFs
        # because residuals across pinned<->free edges still inform free
        # segments -- only the WRITE-BACK is restricted to [pin:]).
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data[0] for kf in kfs]))
        Xs = torch.stack([kf.X_canon for kf in kfs])           # (N, P, 3)
        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        # In calib mode rays are defined by the intrinsics K, not the raw
        # MASt3R prediction. Project onto the K-ray basis first so the
        # (ray, depth) decomposition is K-consistent across consumers.
        if self.K is not None:
            img_size = self.frames[0].img.shape[-2:]
            Xs = constrain_points_to_ray(img_size, Xs, self.K)

        eps = 1e-8
        depths = torch.linalg.norm(Xs, dim=-1)                  # (N, P)
        rays = Xs / depths.clamp(min=eps).unsqueeze(-1)         # (N, P, 3)

        # depth_init = pristine MASt3R snapshot per KF (frozen anchor for
        # the segment template d_p = d_init_p * exp(s)). Fall back to
        # current depth for any KF without a snapshot.
        init_list = []
        for k, kf in enumerate(kfs):
            if kf.depth_init is not None:
                init_list.append(kf.depth_init)
            else:
                init_list.append(depths[k].clone())
        depth_init = torch.stack(init_list)

        # Remap ii, jj into local [0, n_unique).
        pos_of = torch.full(
            (int(unique_kf_idx.max().item()) + 1,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        pos_of[unique_kf_idx] = torch.arange(n_unique, device=self.device)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()
        ii_local = pos_of[ii]
        jj_local = pos_of[jj]
        assert bool((ii_local >= 0).all().item()) and bool((jj_local >= 0).all().item()), \
            "Edge endpoint not found in unique_kf_idx -- factor graph state inconsistency"

        if num_iters is None:
            num_iters = int(dr_cfg.get("depth_iters", 3))

        H_img, W_img = self.frames[0].img.shape[-2:]

        # Per-KF oversegmentation: SLIC on (normal_xyz, log-depth) features
        # derived directly from the canonical pointmap. SAM-free; low-conf
        # pixels are masked out (label = -1) so they are ignored by every
        # downstream kernel.
        pack = build_segments(Xs, Cs, depth_init, int(H_img), int(W_img), dr_cfg)

        if pack.M == 0:
            # All pixels masked out (e.g. confidence below threshold
            # everywhere) -- nothing to refine.
            return

        pose_data = T_WCs.data.contiguous()
        rays = rays.contiguous()
        Cs = Cs.contiguous()
        depth_init = depth_init.contiguous()
        ii_local = ii_local.contiguous()
        jj_local = jj_local.contiguous()
        idx_ii2jj = idx_ii2jj.contiguous()
        valid_match = valid_match.contiguous()
        Q_ii2jj = Q_ii2jj.contiguous()

        depths_out, _seg_scale = como3r_slam_backends.gauss_newton_seg_depths(
            pose_data,
            depth_init,
            rays,
            Cs,
            pack.labels_flat,
            pack.seg_offsets,
            pack.seg_kf,
            ii_local,
            jj_local,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            pack.bnd_seg_a,
            pack.bnd_seg_b,
            pack.bnd_log_d_diff,
            pin,                                                          # num_fix
            float(dr_cfg.get("sigma_point", 0.05)),
            float(dr_cfg.get("lambda_depth_prior", 1.0)),
            float(dr_cfg.get("depth_min", 0.01)),
            float(self.cfg["C_conf"]),
            float(self.cfg["Q_conf"]),
            1.345,                                                        # huber radius
            float(dr_cfg.get("lambda_smooth", 1.0)),
            float(dr_cfg.get("lambda_lm", 0.05)),
            float(dr_cfg.get("s_step_clip", 0.5)),
            num_iters,
        )

        # Write back X = depth * ray for non-pinned KFs only (preserves
        # the Sim(3) gauge anchored at pinned KFs).
        new_X = depths_out.unsqueeze(-1) * rays                  # (N, P, 3)
        self.frames.update_X_from_depth(new_X[pin:], unique_kf_idx[pin:])

    def _pose_GN_step_rays(self, max_iter):
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return False

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        como3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
        return True

    def _pose_GN_step_calib(self, max_iter):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return False

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        height, width = img_size

        como3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
        return True

    def solve_GN_rays(self, override_max_iters=None):
        max_iter = override_max_iters if override_max_iters is not None else int(self.cfg["max_iters"])
        self._pose_GN_step_rays(max_iter)

    def solve_GN_calib(self, override_max_iters=None):
        max_iter = override_max_iters if override_max_iters is not None else int(self.cfg["max_iters"])
        self._pose_GN_step_calib(max_iter)
