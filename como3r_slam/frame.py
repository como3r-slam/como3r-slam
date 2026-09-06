import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from como3r_slam.geometry import constrain_points_to_ray
from como3r_slam.mast3r_utils import resize_img
from como3r_slam.config import config


class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    # Pristine per-pixel MASt3R depth snapshot, frozen reference used by
    # refine_depths' quadratic prior. Anchored once when the keyframe is first
    # inserted (before any GN runs), so multi-call refinement does not drift
    # away from MASt3R's smooth prediction. Shape: (P,).
    depth_init: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None

    def get_score(self, C):
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        filtering_mode = config["tracking"]["filtering_mode"]

        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":

            def cartesian_to_spherical(P):
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None


def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    img = resize_img(img, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
    return frame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.INIT)
        self.reloc_sem = manager.Value("i", 0)
        self.global_optimizer_tasks = manager.list()
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos

    def get_frame(self):
        with self.lock:
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1


class SharedKeyframes:
    # Every per-slot tensor, i.e. everything that has to be reallocated and
    # copied when the store grows. `K` is intentionally absent: it is a single
    # 3x3 intrinsics matrix, not indexed by keyframe.
    _SLOT_TENSORS = (
        "dataset_idx", "img", "uimg", "img_shape", "img_true_shape", "T_WC",
        "X", "depth_init", "depth_init_set", "C", "N", "N_updates", "feat",
        "pos", "is_dirty", "agent_id",
    )

    def __init__(
        self,
        manager,
        h,
        w,
        buffer=512,
        dtype=torch.float32,
        device="cuda",
        can_grow=False,
        name="keyframes",
        buffer_hint="",
    ):
        """Fixed-capacity keyframe store backed by shared-memory tensors.

        `can_grow` may only be set for stores that live entirely inside ONE
        process (the per-agent local store). Growing reallocates every slot
        tensor, and other processes keep pointing at the original shared
        memory, so a cross-process store (the global keyframe store)
        must be sized correctly up front instead.

        `buffer_hint` is the knob to mention when a non-growable store fills up.
        """
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device
        self.can_grow = can_grow
        self.name = name
        self.buffer_hint = buffer_hint

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        self._allocate(buffer)
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()

    def _allocate(self, buffer):
        h, w, device, dtype = self.h, self.w, self.device, self.dtype
        self.buffer = buffer
        # fmt:off
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        # Pristine MASt3R depth snapshot, parallel to self.X. Frozen anchor
        # for refine_depths' quadratic prior so per-call refinement does not
        # drift the depth field away from MASt3R's smooth prediction.
        self.depth_init = torch.zeros(buffer, h * w, device=device, dtype=dtype).share_memory_()
        self.depth_init_set = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        # Per-keyframe agent id (-1 = unassigned, as in an agent local store).
        self.agent_id = torch.full((buffer,), -1, device=device, dtype=torch.int).share_memory_()
        # fmt: on

    def slot_bytes(self):
        """Approximate GPU bytes consumed by one keyframe slot."""
        total = 0
        for name in self._SLOT_TENSORS:
            t = getattr(self, name)
            if t.device.type == "cpu":
                continue
            total += t.element_size() * t[0].numel()
        return total

    def _ensure_capacity(self, idx):
        """Make room for slot `idx`, growing if this store is allowed to."""
        if idx < self.buffer:
            return
        mb = self.slot_bytes() / (1 << 20)
        if not self.can_grow:
            hint = (
                f" Raise {self.buffer_hint}." if self.buffer_hint else ""
            )
            raise RuntimeError(
                f"keyframe store '{self.name}' is full: {self.buffer} slots, "
                f"tried to write slot {idx}.{hint} Each slot costs ~{mb:.1f} MB "
                f"of GPU memory ({self.buffer} slots = "
                f"~{self.buffer * mb / 1024:.1f} GB)."
            )
        # Grow by at most 64 slots at a time: the copy holds the old and the
        # new tensors alive simultaneously, so doubling a large store would
        # spike GPU memory by 2x its current size (~0.6 GiB per step at
        # 512x384 instead of several GiB).
        new_buffer = max(idx + 1, self.buffer + min(self.buffer, 64))
        old = {name: getattr(self, name) for name in self._SLOT_TENSORS}
        n = self.n_size.value
        self._allocate(new_buffer)
        for name in self._SLOT_TENSORS:
            getattr(self, name)[:n] = old[name][:n]
        del old
        print(
            f"[keyframes] grew '{self.name}' store to {new_buffer} slots "
            f"(~{new_buffer * mb / 1024:.1f} GB GPU)"
        )

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # put all of the data into a frame
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.uimg[idx],
                lietorch.Sim3(self.T_WC[idx]),
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if bool(self.depth_init_set[idx]):
                kf.depth_init = self.depth_init[idx]
            if config["use_calib"]:
                kf.K = self.K
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            self._ensure_capacity(idx)
            self.n_size.value = max(idx + 1, self.n_size.value)

            # set the attributes
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True

            # depth_init snapshot rules:
            #  - If caller supplies value.depth_init (e.g. propagated from a
            #    local store that already holds the pristine snapshot), trust
            #    and store it.
            #  - Else, on first write to this slot, snapshot |X_canon|. Refine
            #    writes go through update_X_from_depth, NOT setitem, so this
            #    initial snapshot remains frozen across refine iterations.
            #  - In calib mode the ray basis is K-defined; constrain X to the
            #    K-rays before taking the norm so depth_init is consistent
            #    with refine_depths' decomposition.
            if value.depth_init is not None:
                self.depth_init[idx] = value.depth_init
                self.depth_init_set[idx] = True
            elif not bool(self.depth_init_set[idx]):
                X = value.X_canon
                if config["use_calib"]:
                    img_size = self.img.shape[-2:]
                    X = constrain_points_to_ray(img_size, X[None], self.K)[0]
                self.depth_init[idx] = torch.linalg.norm(X, dim=-1)
                self.depth_init_set[idx] = True
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame, agent_id: int = -1):
        with self.lock:
            idx = self.n_size.value
            self[idx] = value
            self.agent_id[idx] = agent_id
            return idx

    def get_agent_id(self, idx):
        with self.lock:
            return int(self.agent_id[idx])

    def get_agent_keyframe_indices(self, agent_id):
        with self.lock:
            n = self.n_size.value
            return [i for i in range(n) if int(self.agent_id[i]) == agent_id]

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def update_X_from_depth(
        self,
        new_X: torch.Tensor,
        idx_tensor: torch.Tensor,
    ) -> None:
        """Write refined per-keyframe pointmaps back to the shared buffer.

        ``idx_tensor`` and ``new_X`` are aligned: ``new_X[k]`` corresponds to
        global keyframe index ``idx_tensor[k]``. Caller is responsible for
        having already excluded any pinned (gauge-fixed) keyframes.
        """
        with self.lock:
            idx_list = idx_tensor.tolist()
            for local_k, kf_k in enumerate(idx_list):
                self.X[kf_k] = new_X[local_k]
                self.is_dirty[kf_k] = True

    def set_intrinsics(self, K):
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
