"""Super-primitive depth refinement (inspired by https://arxiv.org/pdf/2312.05889).

Replaces the older per-pixel depth refinement. Two pieces live here:

  1. ``build_segments``: oversegments each keyframe from MASt3R's canonical
     pointmap (no SAM). Per-pixel surface normals + log-depth are fed to SLIC,
     low-confidence pixels are tagged invalid (label = -1).

  2. ``prepare_seg_tensors``: packs the per-KF label maps + the per-KF
     ``depth_init`` into the flat layout expected by the CUDA solver -- a
     CSR-style ``seg_offsets`` plus dedup'd "adjacent segment pair"
     descriptors with representative boundary depths.

The actual Gauss-Newton iterations live in CUDA -- see
``como3r_slam_backends.gauss_newton_seg_depths``. Each segment owns ONE
parameter (log-scale ``s``) and per-pixel depth is ``d_init * exp(s)``;
the per-segment Hessian is dense within a segment but the system is sparse
across segments (only adjacent-pair coupling from the smoothness term).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic


# ---------------------------------------------------------------------------
# Per-pixel normals from the pointmap.
# ---------------------------------------------------------------------------

def _pixel_normals(X_img: torch.Tensor) -> torch.Tensor:
    """Estimate per-pixel surface normals from a canonical pointmap.

    Args:
        X_img: (H, W, 3) tensor of 3D points in camera frame.

    Returns:
        (H, W, 3) unit-norm normals. Borders are filled with a fallback
        normal so the result is well-defined at every pixel.
    """
    # Central differences along u (W) and v (H), padded with edge values
    # so the gradients at the border are zero rather than wrap-around.
    Xp = F.pad(X_img.permute(2, 0, 1).unsqueeze(0),
               (1, 1, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    du = Xp[1:-1, 2:] - Xp[1:-1, :-2]
    dv = Xp[2:, 1:-1] - Xp[:-2, 1:-1]
    n = torch.cross(du, dv, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return n


# ---------------------------------------------------------------------------
# Segmentation entry point.
# ---------------------------------------------------------------------------

@dataclass
class SegResult:
    """Per-keyframe oversegmentation result.

    ``labels`` is row-major flattened (P,) int32 with values in [0, n_seg)
    for valid pixels and -1 for masked/invalid pixels. ``n_seg`` is the
    number of distinct valid labels actually present in ``labels``.
    """
    labels: torch.Tensor      # (P,) int32 on device
    n_seg: int


def oversegment_keyframe(
    X_canon: torch.Tensor,
    C: torch.Tensor,
    H: int,
    W: int,
    n_segments: int = 200,
    compactness: float = 8.0,
    c_thresh: float = 1.5,
    sigma: float = 1.0,
) -> SegResult:
    """Oversegment a single keyframe using SLIC over geometric features.

    Features: per-pixel surface normal (3 channels) + log-depth (1 channel),
    derived directly from MASt3R's canonical pointmap.

    Args:
        X_canon: (P, 3) canonical pointmap.
        C: (P, 1) or (P,) average MASt3R confidence.
        H, W: image dims; P must equal H * W.
        n_segments: target SLIC seed count.
        compactness: SLIC spatial-vs-feature weighting.
        c_thresh: pixels with C < c_thresh are marked invalid (label = -1).
        sigma: SLIC pre-blur sigma.

    Returns:
        ``SegResult`` with a dense, contiguous label map for valid pixels.
    """
    device = X_canon.device
    assert X_canon.shape == (H * W, 3), f"expected ({H*W}, 3), got {X_canon.shape}"

    X_img = X_canon.reshape(H, W, 3)
    C_img = C.reshape(H, W) if C.dim() == 2 else C.reshape(H, W, -1)[..., 0]

    normals = _pixel_normals(X_img)                                     # (H, W, 3)
    depth = X_img.norm(dim=-1).clamp(min=1e-3)
    logd = torch.log(depth)

    # Normalize log-depth to [0, 1] so it's commensurate with normal
    # components (which already live in [-1, 1]). SLIC's compactness param
    # trades spatial vs feature distance and assumes features are roughly
    # bounded; without normalization, log-depth dominates everywhere.
    logd_min = logd.min()
    logd_max = logd.max()
    logd_norm = (logd - logd_min) / (logd_max - logd_min).clamp(min=1e-6)

    # SLIC runs on CPU numpy. Stack to (H, W, 4) channel-last.
    feat = torch.stack(
        (normals[..., 0], normals[..., 1], normals[..., 2], logd_norm),
        dim=-1,
    ).contiguous().cpu().numpy().astype(np.float32)

    # SLIC ignores the C-mask -- segments cross low-confidence regions
    # freely. We mask AFTER SLIC by stamping invalid pixels with -1.
    raw = slic(
        feat,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        channel_axis=-1,
        start_label=0,
        enforce_connectivity=True,
        convert2lab=False,
    ).astype(np.int32)

    invalid = (C_img < c_thresh).cpu().numpy()
    raw[invalid] = -1

    # Tiny segments are pointless and inflate M. Drop any segment with
    # fewer than min_seg_pixels valid pixels (mark as -1). Then relabel
    # so values are dense in [0, n_seg). A single-element segment would
    # also make the per-segment Newton step extremely noisy.
    min_seg_pixels = 16
    flat = raw.reshape(-1)
    valid_mask = flat >= 0
    if valid_mask.any():
        # bincount on valid labels only
        counts = np.bincount(flat[valid_mask])
        small = np.where(counts < min_seg_pixels)[0]
        if small.size > 0:
            small_set = np.zeros(counts.size, dtype=bool)
            small_set[small] = True
            flat[valid_mask & small_set[np.clip(flat, 0, None)]] = -1
        # Dense relabel
        valid_mask = flat >= 0
        if valid_mask.any():
            uniq, inv = np.unique(flat[valid_mask], return_inverse=True)
            new_flat = np.full_like(flat, -1)
            new_flat[valid_mask] = inv.astype(np.int32)
            flat = new_flat

    n_seg = int(flat.max()) + 1 if (flat >= 0).any() else 0
    labels = torch.from_numpy(flat).to(device=device, dtype=torch.int32)
    return SegResult(labels=labels, n_seg=n_seg)


# ---------------------------------------------------------------------------
# Boundary pair extraction (one residual per unique adjacent segment pair).
# ---------------------------------------------------------------------------

def _kf_boundary_pairs(
    labels_2d: torch.Tensor,
    depth_init_2d: torch.Tensor,
    seg_base: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find unique 4-adjacent (segA, segB) pairs for one keyframe.

    For each unique pair we precompute ``log_d_diff = log d_init_a - log
    d_init_b``, averaged across boundary pixel pairs. The CUDA smoothness
    residual is then ``(s_a - s_b) + log_d_diff``, pushing log d_a - log
    d_b toward the MASt3R-suggested ratio. Log-space keeps the residual
    linear in s and immune to exp() overflow.

    Args:
        labels_2d: (H, W) int32; -1 marks invalid.
        depth_init_2d: (H, W) float; pristine MASt3R depth.
        seg_base: KF's segment-offset, added so the returned IDs are
            global (indexing seg_scale flat).

    Returns:
        (seg_a, seg_b, log_d_diff) tensors of length B (B may be 0).
    """
    H, W = labels_2d.shape

    # Horizontal pairs (u, u+1) and vertical pairs (v, v+1). Keep all
    # 4-adjacent boundary pixel pairs, dedupe by (min(a,b), max(a,b)) and
    # average log d_init across the boundary -- one residual per unique
    # segment pair, with longer shared boundaries getting a tighter
    # estimate of the MASt3R-suggested log-ratio.
    a_h = labels_2d[:, :-1]
    b_h = labels_2d[:, 1:]
    da_h = depth_init_2d[:, :-1]
    db_h = depth_init_2d[:, 1:]
    mh = (a_h != b_h) & (a_h >= 0) & (b_h >= 0)

    a_v = labels_2d[:-1, :]
    b_v = labels_2d[1:, :]
    da_v = depth_init_2d[:-1, :]
    db_v = depth_init_2d[1:, :]
    mv = (a_v != b_v) & (a_v >= 0) & (b_v >= 0)

    seg_a = torch.cat([a_h[mh], a_v[mv]]).to(torch.int64)
    seg_b = torch.cat([b_h[mh], b_v[mv]]).to(torch.int64)
    da = torch.cat([da_h[mh], da_v[mv]])
    db = torch.cat([db_h[mh], db_v[mv]])

    if seg_a.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.int32, device=labels_2d.device)
        empty_f = torch.empty(0, dtype=depth_init_2d.dtype, device=labels_2d.device)
        return empty_long, empty_long, empty_f

    # Canonicalize (a, b) so a < b, then dedupe.
    swap = seg_a > seg_b
    a_s = torch.where(swap, seg_b, seg_a)
    b_s = torch.where(swap, seg_a, seg_b)
    da_s = torch.where(swap, db, da)
    db_s = torch.where(swap, da, db)

    # Pack (a, b) into a single int64 key for unique. Max segment count
    # per KF is small (a few hundred), so 32-bit halves comfortably fit.
    n_local = int(max(a_s.max().item(), b_s.max().item())) + 1
    key = a_s * n_local + b_s
    uniq_key, inv = torch.unique(key, return_inverse=True)

    # Aggregate per-pair log_d_init difference (mean across boundary).
    Bp = uniq_key.numel()
    counts = torch.zeros(Bp, dtype=torch.float32, device=labels_2d.device)
    sum_diff = torch.zeros(Bp, dtype=depth_init_2d.dtype, device=labels_2d.device)
    log_da = torch.log(da_s.clamp(min=1e-3))
    log_db = torch.log(db_s.clamp(min=1e-3))
    diff = log_da - log_db
    counts.scatter_add_(0, inv, torch.ones_like(diff, dtype=torch.float32))
    sum_diff.scatter_add_(0, inv, diff)
    counts = counts.clamp(min=1.0)
    log_d_diff_out = sum_diff / counts

    a_out = (uniq_key // n_local).to(torch.int32) + seg_base
    b_out = (uniq_key % n_local).to(torch.int32) + seg_base
    return a_out, b_out, log_d_diff_out


# ---------------------------------------------------------------------------
# Pack per-KF segmentations into flat solver inputs.
# ---------------------------------------------------------------------------

@dataclass
class SegPack:
    labels_flat: torch.Tensor       # (N, P) int32 -- local IDs per KF, -1 = invalid
    seg_offsets: torch.Tensor       # (N+1,) int32, M = seg_offsets[-1]
    seg_kf: torch.Tensor            # (M,) int32 -- KF index for each segment
    bnd_seg_a: torch.Tensor         # (B,) int32 -- global segment IDs
    bnd_seg_b: torch.Tensor
    # log d_init_a - log d_init_b averaged across the shared boundary;
    # the CUDA smoothness residual is (s_a - s_b) + log_d_diff.
    bnd_log_d_diff: torch.Tensor    # (B,) float

    @property
    def M(self) -> int:
        return int(self.seg_offsets[-1].item())

    @property
    def B(self) -> int:
        return int(self.bnd_seg_a.numel())


def pack_segments(
    seg_results: List[SegResult],
    depth_inits: torch.Tensor,
    H: int,
    W: int,
) -> SegPack:
    """Pack per-KF segmentations into the flat layout the CUDA solver expects.

    Args:
        seg_results: length-N list of per-KF oversegmentations.
        depth_inits: (N, P) pristine MASt3R depths (used for boundary residual).
        H, W: image dims; P = H * W.

    Returns:
        A ``SegPack`` ready to be passed straight through to
        ``como3r_slam_backends.gauss_newton_seg_depths``.
    """
    device = depth_inits.device
    N = len(seg_results)
    P = H * W
    assert depth_inits.shape == (N, P), (depth_inits.shape, N, P)

    labels_flat = torch.stack([s.labels for s in seg_results], dim=0)   # (N, P)
    n_per_kf = torch.tensor([s.n_seg for s in seg_results],
                            dtype=torch.int64, device=device)
    seg_offsets = torch.zeros(N + 1, dtype=torch.int32, device=device)
    seg_offsets[1:] = torch.cumsum(n_per_kf, dim=0).to(torch.int32)
    M = int(seg_offsets[-1].item())

    # seg_kf: KF index for each of the M segments (used for the pin check
    # in the update kernel -- pinned KFs' segments are not updated).
    seg_kf = torch.empty(M, dtype=torch.int32, device=device)
    for k in range(N):
        s, e = int(seg_offsets[k].item()), int(seg_offsets[k + 1].item())
        if e > s:
            seg_kf[s:e] = k

    # Boundary pair lists -- concatenate across KFs (boundaries are
    # within-KF only; cross-KF coupling already comes from the data term).
    a_list, b_list, diff_list = [], [], []
    for k, s in enumerate(seg_results):
        if s.n_seg < 2:
            continue
        labels_2d = s.labels.reshape(H, W)
        depth_2d = depth_inits[k].reshape(H, W)
        a, b, diff = _kf_boundary_pairs(labels_2d, depth_2d,
                                        int(seg_offsets[k].item()))
        if a.numel() > 0:
            a_list.append(a); b_list.append(b); diff_list.append(diff)

    if a_list:
        bnd_seg_a = torch.cat(a_list).contiguous()
        bnd_seg_b = torch.cat(b_list).contiguous()
        bnd_log_d_diff = torch.cat(diff_list).contiguous()
    else:
        bnd_seg_a = torch.empty(0, dtype=torch.int32, device=device)
        bnd_seg_b = torch.empty(0, dtype=torch.int32, device=device)
        bnd_log_d_diff = torch.empty(0, dtype=depth_inits.dtype, device=device)

    return SegPack(
        labels_flat=labels_flat.contiguous(),
        seg_offsets=seg_offsets.contiguous(),
        seg_kf=seg_kf.contiguous(),
        bnd_seg_a=bnd_seg_a,
        bnd_seg_b=bnd_seg_b,
        bnd_log_d_diff=bnd_log_d_diff,
    )


def build_segments(
    X_canons: torch.Tensor,
    Cs: torch.Tensor,
    depth_inits: torch.Tensor,
    H: int,
    W: int,
    cfg: Optional[dict] = None,
) -> SegPack:
    """Full pipeline: per-KF oversegmentation + pack into solver inputs.

    Args:
        X_canons: (N, P, 3) canonical pointmaps (already K-ray-projected
            in calib mode).
        Cs: (N, P, 1) average MASt3R confidences.
        depth_inits: (N, P) pristine MASt3R depths.
        H, W: image dims.
        cfg: ``depth_refinement`` config dict. Keys honored:
            ``seg_n_segments`` (default 200), ``seg_compactness`` (8.0),
            ``seg_c_thresh`` (1.5), ``seg_sigma`` (1.0).

    Returns:
        ``SegPack`` ready to feed the CUDA solver.
    """
    cfg = cfg or {}
    n_segments = int(cfg.get("seg_n_segments", 200))
    compactness = float(cfg.get("seg_compactness", 8.0))
    c_thresh = float(cfg.get("seg_c_thresh", 1.5))
    sigma = float(cfg.get("seg_sigma", 1.0))

    N = X_canons.shape[0]
    results = []
    for k in range(N):
        results.append(oversegment_keyframe(
            X_canons[k], Cs[k], H, W,
            n_segments=n_segments,
            compactness=compactness,
            c_thresh=c_thresh,
            sigma=sigma,
        ))
    return pack_segments(results, depth_inits, H, W)
