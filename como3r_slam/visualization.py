"""Multi-agent visualization for CoMo3R-SLAM.

Renders all agents' keyframes and current camera frustums in one shared 3D
view. Each agent gets its own color so its trajectory is visually distinct.

Reads:
- list[SharedStates], one per agent  -> current frame / mode / edges
- MultiAgentKeyframes                 -> all agents' keyframes (tagged with
                                         agent_id) and the global edge snapshot
                                         the coordinator publishes after each
                                         Level-2 solve.
"""
import dataclasses
import pathlib
import weakref
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from como3r_slam.config import config, set_global_config
from como3r_slam.frame import Mode
from como3r_slam.geometry import get_pixel_coords
from como3r_slam.lietorch_utils import as_SE3
from como3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)


@dataclasses.dataclass
class WindowMsg:
    """Viewer -> main-process control message."""

    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


# Distinct colors per agent (RGBA). Picked from matplotlib's tab10 palette so
# the same agent gets a stable color across runs.
_AGENT_PALETTE_RGB = [
    (0.894, 0.102, 0.110),  # red
    (0.216, 0.494, 0.722),  # blue
    (0.302, 0.686, 0.290),  # green
    (1.000, 0.498, 0.000),  # orange
    (0.596, 0.306, 0.639),  # purple
    (0.651, 0.337, 0.157),  # brown
    (0.969, 0.506, 0.749),  # pink
    (0.498, 0.498, 0.498),  # grey
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.812),  # cyan
]


def agent_color(agent_id: int, alpha: float = 1.0):
    r, g, b = _AGENT_PALETTE_RGB[agent_id % len(_AGENT_PALETTE_RGB)]
    return [r, g, b, alpha]


def _lighten(rgb_color, factor: float = 0.5):
    """Blend toward white by `factor` to produce a 'current cam' variant."""
    r, g, b, a = rgb_color
    return [r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor, a]


class MultiAgentWindow(WindowEvents):
    title = "CoMo3R-SLAM (multi-agent)"
    window_size = (1960, 1080)

    def __init__(
        self,
        agent_states_list,
        global_keyframes,
        main2viz,
        viz2main,
        per_frame_npz_paths=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#FFFFFF", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()

        self.agent_states_list = agent_states_list
        self.global_keyframes = global_keyframes
        self.num_agents = len(agent_states_list)

        # Which agent the camera should follow when follow_cam is on.
        self.follow_agent_id = 0

        self.show_all = True
        self.show_keyframe_edges = True
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True
        self.show_curr_cams = True

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_imgs = [Image() for _ in range(self.num_agents)]
        self.curr_img_nps = [None for _ in range(self.num_agents)]
        # Cache the latest current-frame pointmap textures per agent so
        # _draw_current_pointmaps can re-render them without paying the GPU
        # texture upload on every frame.
        self._curr_pointmap_tex = [None for _ in range(self.num_agents)]
        # Per-agent (last_frame_id, last_K_set) used to skip redundant uploads
        # and to lazily set the camera intrinsics on each agent's current frame.
        self._curr_pointmap_last_fid = [-1] * self.num_agents
        # Calibration ray-direction cache (per agent) -- ``_frame_X`` builds a
        # (H, W, 3) per-pixel ray table from K; recompute only when K changes.
        self._dP_dz_per_agent = [None] * self.num_agents

        self.main2viz = main2viz
        self.viz2main = viz2main

        # Full-trajectory overlay (every frame, not just KFs). Each agent
        # writes a per_frame_agent_<id>.npz when it finishes processing, plus
        # a per_frame_agent_<id>.ready sentinel touched AFTER np.savez closes
        # so we never read a half-written npz. We poll every second; once all
        # sentinels are present, load the records and enable the UI toggle.
        self.per_frame_npz_paths = (
            {int(k): pathlib.Path(v) for k, v in per_frame_npz_paths.items()}
            if per_frame_npz_paths is not None
            else None
        )
        self.full_traj_ready = False
        self.full_traj_data = {}          # agent_id -> dict of np arrays
        self.show_full_traj = False
        # When the full-traj overlay is on, optionally also draw a small
        # frustum at every non-KF camera pose (KFs already get their own
        # frustums via `show_keyframe`). Scaled smaller than KF frustums and
        # dimmer so the dense per-frame cluster doesn't swamp the KFs.
        self.show_full_traj_frustums = False
        self.full_traj_frustum_scale_factor = 0.5
        self._last_full_traj_poll_t = -1e9
        self.full_traj_poll_interval = 1.0  # seconds

        # Pose-change diagnostic: prints whenever the viz observes a keyframe
        # translation moving between renders. If a global GN solve runs but no
        # changes show up here, the viz is not actually reading shared memory.
        self._prev_translations = {}      # kf_idx -> (x,y,z)
        self._last_pose_log_t = 0.0
        self.pose_log_interval = 1.0      # seconds
        self.pose_change_thresh = 0.02    # meters; below this we don't print

    # ---------- 3D rendering helpers ----------------------------------------

    def _draw_current_cameras(self):
        """One frustum per agent's most recent tracked frame."""
        for agent_id, states in enumerate(self.agent_states_list):
            if states.get_mode() == Mode.INIT:
                # No useful pose yet.
                continue
            try:
                curr_frame = states.get_frame()
            except Exception:
                continue
            try:
                self.curr_img_nps[agent_id] = curr_frame.uimg.cpu().numpy()
                self.curr_imgs[agent_id].write(self.curr_img_nps[agent_id])
            except Exception:
                pass

            h, w = curr_frame.img_shape.flatten()
            self.frustums.make_frustum(h, w)

            cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
            if self.follow_cam and agent_id == self.follow_agent_id:
                T_WC = cam_T_WC.matrix().numpy().astype(
                    dtype=np.float32
                ) @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))
                self.camera.follow_cam(np.linalg.inv(T_WC))

            if self.show_curr_cams:
                color = _lighten(agent_color(agent_id, 1.0), 0.4)
                self.frustums.add(
                    cam_T_WC,
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )
        if not self.follow_cam:
            self.camera.unfollow_cam()

    def _refresh_dirty_keyframe_textures(self):
        kfs = self.global_keyframes.keyframes
        with kfs.lock:
            dirty_idx = kfs.get_dirty_idx()
        for kf_idx in dirty_idx:
            keyframe = kfs[int(kf_idx)]
            h, w = keyframe.img_shape.flatten()
            X = self._frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)
            tex_key = (int(kf_idx), int(keyframe.frame_id))
            if tex_key not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[tex_key] = ptex, ctex, itex
                ptex, ctex, itex = self.textures[tex_key]
                itex.write(keyframe.uimg.cpu().numpy().astype(np.float32).tobytes())
            ptex, ctex, itex = self.textures[tex_key]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())

    def _draw_keyframes_and_pointmaps(self):
        kfs = self.global_keyframes.keyframes
        with kfs.lock:
            N = len(kfs)
            agent_ids = kfs.agent_id[:N].cpu().tolist()

        for kf_idx in range(N):
            keyframe = kfs[kf_idx]
            h, w = keyframe.img_shape.flatten()
            agent_id = int(agent_ids[kf_idx])
            color = agent_color(agent_id, 1.0) if agent_id >= 0 else [1, 1, 1, 1]

            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )

            tex_key = (kf_idx, int(keyframe.frame_id))
            if tex_key not in self.textures:
                # Texture lazily created via the dirty path; skip until ready.
                continue
            ptex, ctex, itex = self.textures[tex_key]
            if self.show_all:
                # Keyframes always show captured image colors. Per-agent
                # differentiation lives in the frustum color and (for the
                # live cloud) in ``_draw_current_pointmaps``.
                self._render_pointmap(
                    keyframe.T_WC.cpu(), w, h, ptex, ctex, itex,
                    use_img=True,
                )

    def _draw_current_pointmaps(self):
        """Live pointmap for each agent's currently-tracked frame.

        Each agent gets its own (ptex, ctex, itex) trio keyed by its slot in
        ``self.textures`` so keyframe texture eviction doesn't clobber the
        live cloud.
        """
        if not self.show_curr_pointmap:
            return
        kfs = self.global_keyframes.keyframes
        for agent_id, states in enumerate(self.agent_states_list):
            if states.get_mode() == Mode.INIT:
                continue
            try:
                curr_frame = states.get_frame()
            except Exception:
                continue
            if getattr(curr_frame, "X_canon", None) is None:
                continue

            # ``_frame_X`` rebuilds X from depth using the camera intrinsics
            # in calib mode -- ensure K is set on the current frame, taking
            # it from the shared keyframe store on first access.
            if config["use_calib"] and getattr(curr_frame, "K", None) is None:
                curr_frame.K = kfs.get_intrinsics()

            h, w = (int(v) for v in curr_frame.img_shape.flatten())
            # An agent whose mode has already left INIT but whose first real
            # ``frame`` event hasn't been applied yet still carries the
            # all-zero placeholder frame (img_shape == [0, 0], X == zeros).
            # Rendering it would size the GL texture to 0x0 (moderngl clamps
            # that to 1x1) and then mismatch on the full-size ``write``.
            if h <= 0 or w <= 0:
                continue
            X = self._frame_X(curr_frame, agent_id=agent_id)
            C = curr_frame.C.cpu().numpy().astype(np.float32)
            uimg = curr_frame.uimg.cpu().numpy().astype(np.float32)

            tex_key = ("curr", agent_id)
            tex = self.textures.get(tex_key)
            # Defensive: drop a cached trio whose size no longer matches the
            # current frame so a stale-sized texture can never be written to.
            if tex is not None and tex[0].size != (w, h):
                for t in tex:
                    t.release()
                tex = None
                self.textures.pop(tex_key, None)
            if tex is None:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                tex = (ptex, ctex, itex)
                self.textures[tex_key] = tex
            ptex, ctex, itex = tex
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(uimg.tobytes())

            # A single-stream viewer would draw the live cloud with captured
            # image colors (use_img=True). Here we want each robot's cloud
            # visually separable -- so we flat-shade with the agent's
            # palette color. Keyframes stay image-textured (mono-style).
            color = agent_color(agent_id, 1.0)
            self._render_pointmap(
                curr_frame.T_WC.cpu(), w, h, ptex, ctex, itex,
                use_img=False,
                base_color=color[:3],
                depth_bias=self.depth_bias,
            )

    def _maybe_load_full_traj(self, t: float):
        """Poll for per-agent .ready sentinels and load the npz files once
        every agent has produced one. Files only appear AFTER agents finish
        their entire stream, so flipping ``full_traj_ready`` to True doubles
        as a "global processing complete" signal in the UI.
        """
        if self.full_traj_ready or self.per_frame_npz_paths is None:
            return
        if t - self._last_full_traj_poll_t < self.full_traj_poll_interval:
            return
        self._last_full_traj_poll_t = t
        for aid, p in self.per_frame_npz_paths.items():
            ready = p.parent / f"{p.stem}.ready"
            if not ready.exists():
                return  # not all agents done yet
        # All sentinels present -- safe to load.
        for aid, p in self.per_frame_npz_paths.items():
            try:
                d = np.load(p)
                self.full_traj_data[aid] = {
                    "frame_ids": np.asarray(d["frame_ids"]),
                    "anchor_global_idx": np.asarray(d["anchor_global_idx"]),
                    "T_rel": np.asarray(d["T_rel"]),
                }
            except Exception as e:
                print(f"[viz] failed to load {p}: {e}")
                return
        self.full_traj_ready = True
        n = sum(d["frame_ids"].shape[0] for d in self.full_traj_data.values())
        print(
            f"[viz] full-trajectory records loaded ({n} non-KF entries across "
            f"{len(self.full_traj_data)} agents) -- toggle 'show full traj' "
            f"in the GUI to display them"
        )

    def _draw_full_trajectories(self):
        """Compose each non-KF frame's pose from current global anchor + T_rel
        and draw a per-agent polyline through every frame (KFs included).

        Recomputed every render so further global solves (after agents
        finished) keep the line consistent with the live KF poses.
        """
        if not self.show_full_traj or not self.full_traj_ready:
            return
        kfs = self.global_keyframes.keyframes
        with kfs.lock:
            N = len(kfs)
            if N == 0:
                return
            ag_ids = kfs.agent_id[:N].cpu().tolist()
            dataset_idx = kfs.dataset_idx[:N].cpu().numpy()
            T_WC_cpu = kfs.T_WC[:N].detach().cpu().clone()  # (N, 1, 8)

        # Batched non-KF composition per agent.
        frustum_scale = self.frustum_scale * self.full_traj_frustum_scale_factor
        frustum_thick = max(self.line_thickness * self.scale * 0.5, 1.0)
        for aid in range(self.num_agents):
            positions = {}  # frame_id -> (3,) world position
            for gidx in range(N):
                if int(ag_ids[gidx]) != aid:
                    continue
                fid = int(dataset_idx[gidx])
                positions[fid] = T_WC_cpu[gidx, 0, :3].numpy()
            data = self.full_traj_data.get(aid)
            non_kf_se3_data = None  # (M', 7) for frustum drawing below
            if data is not None and data["frame_ids"].shape[0] > 0:
                aidx = data["anchor_global_idx"]
                mask = (aidx >= 0) & (aidx < N)
                if mask.any():
                    fids = data["frame_ids"][mask]
                    aidx = aidx[mask]
                    T_rel = data["T_rel"][mask]  # (M, 8)
                    T_anchor = lietorch.Sim3(T_WC_cpu[aidx])  # (M, 1, 8)
                    T_rel_sim3 = lietorch.Sim3(
                        torch.from_numpy(T_rel).reshape(-1, 1, 8).to(T_WC_cpu.dtype)
                    )
                    T_global = T_anchor * T_rel_sim3
                    pos_arr = T_global.data[:, 0, :3].numpy()
                    keep = []
                    for k in range(fids.shape[0]):
                        fid = int(fids[k])
                        if fid in positions:
                            continue  # KF wins
                        positions[fid] = pos_arr[k]
                        keep.append(k)
                    if self.show_full_traj_frustums and keep:
                        keep_idx = torch.tensor(keep, dtype=torch.long)
                        # Drop the Sim(3) scale dimension to get SE(3) data
                        # (t, q); matches the `as_SE3` path used by KF
                        # frustum drawing.
                        non_kf_se3_data = T_global.data[keep_idx, :, :7].detach().cpu()

            if len(positions) >= 2:
                sorted_fids = sorted(positions)
                pts = np.stack(
                    [positions[f] for f in sorted_fids], axis=0
                ).astype(np.float32)
                starts = pts[:-1]
                ends = pts[1:]
                self.lines.add(
                    starts,
                    ends,
                    thickness=self.line_thickness * self.scale,
                    color=agent_color(aid, alpha=1.0),
                )

            if self.show_full_traj_frustums and non_kf_se3_data is not None:
                color = agent_color(aid, alpha=0.6)
                for i in range(non_kf_se3_data.shape[0]):
                    T_se3 = lietorch.SE3(non_kf_se3_data[i])  # (1, 7)
                    self.frustums.add(
                        T_se3,
                        scale=frustum_scale,
                        color=color,
                        thickness=frustum_thick,
                    )

    def _draw_global_edges(self):
        if not self.show_keyframe_edges:
            return
        kfs = self.global_keyframes.keyframes
        with self.global_keyframes.lock:
            ii = list(self.global_keyframes.global_edges_ii)
            jj = list(self.global_keyframes.global_edges_jj)
        if not ii or not jj:
            return
        with kfs.lock:
            N = len(kfs)
            full_agent_ids = kfs.agent_id[:N].cpu().tolist()
        # Be defensive: filter out edges referencing KFs not yet appended.
        pairs = [(i, j) for i, j in zip(ii, jj) if 0 <= i < N and 0 <= j < N]
        if not pairs:
            return
        ii_t = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=kfs.T_WC.device)
        jj_t = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=kfs.T_WC.device)
        with kfs.lock:
            T_WCi = lietorch.Sim3(kfs.T_WC[ii_t, 0])
            T_WCj = lietorch.Sim3(kfs.T_WC[jj_t, 0])
        t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
        t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()

        # Color edges by their type: intra-agent gets that agent's color
        # (dimmed); inter-agent gets cyan so cross-agent loop closures pop
        # visually on the white background.
        for (i_idx, j_idx), s, e in zip(pairs, t_WCi, t_WCj):
            ai = int(full_agent_ids[i_idx])
            aj = int(full_agent_ids[j_idx])
            if ai == aj and ai >= 0:
                c = agent_color(ai, alpha=0.6)
            else:
                c = [0.302, 0.686, 0.290, 1.0]
            self.lines.add(
                s.reshape(1, 3),
                e.reshape(1, 3),
                thickness=self.line_thickness * self.scale,
                color=c,
            )

    # ---------- main render --------------------------------------------------

    def render(self, t: float, frametime: float):
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        # Ensure frustum geometry is initialized every frame -- the underlying
        # Frustums helper has a constructor bug where self.frustum ends up None,
        # so we must call make_frustum() at least once before any .add().
        kfs = self.global_keyframes.keyframes
        self.frustums.make_frustum(kfs.h, kfs.w)

        self._maybe_load_full_traj(t)
        self._draw_current_cameras()
        self._refresh_dirty_keyframe_textures()
        self._draw_keyframes_and_pointmaps()
        self._draw_current_pointmaps()
        self._draw_global_edges()
        self._draw_full_trajectories()
        self._diagnostic_pose_log(t)

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self._render_ui()

    def _diagnostic_pose_log(self, t: float):
        """Print KF translations and report ones that moved between renders.

        This is the definitive check that pose updates from the coordinator
        (via global Sim(3) GN) are visible to the visualizer. If a KF's
        translation changes here, the viz IS reading the updated shared
        memory. If a KF never moves even though coordinator logs say global
        solves ran, that points to a shared-memory read bug instead.
        """
        if t - self._last_pose_log_t < self.pose_log_interval:
            return
        self._last_pose_log_t = t
        kfs = self.global_keyframes.keyframes
        with kfs.lock:
            N = len(kfs)
            if N == 0:
                return
            # T_WC layout is (tx, ty, tz, qx, qy, qz, qw, s); read translation.
            translations = kfs.T_WC[:N, 0, :3].detach().cpu().numpy()
            ag_ids = kfs.agent_id[:N].cpu().tolist()
        moved = []
        for i in range(N):
            new_t = tuple(float(x) for x in translations[i])
            old_t = self._prev_translations.get(i)
            self._prev_translations[i] = new_t
            if old_t is None:
                continue
            d = (
                (new_t[0] - old_t[0]) ** 2
                + (new_t[1] - old_t[1]) ** 2
                + (new_t[2] - old_t[2]) ** 2
            ) ** 0.5
            if d >= self.pose_change_thresh:
                moved.append((i, int(ag_ids[i]), old_t, new_t, d))
        if moved:
            print(
                f"[viz] === observed {len(moved)} KF translation change(s) "
                f"in shared memory (>{self.pose_change_thresh:.2f}m) ==="
            )
            for i, ag, old_t, new_t, d in moved:
                print(
                    f"  KF{i:2d} agent={ag}  "
                    f"({old_t[0]:+.2f},{old_t[1]:+.2f},{old_t[2]:+.2f}) -> "
                    f"({new_t[0]:+.2f},{new_t[1]:+.2f},{new_t[2]:+.2f})  "
                    f"delta={d:.2f}m"
                )

    def _render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("Multi-Agent GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)
        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )
        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        # Per-agent agent_id selector for camera follow.
        if self.follow_cam:
            for aid in range(self.num_agents):
                r, g, b, _ = agent_color(aid)
                imgui.push_style_color(imgui.COLOR_TEXT, r, g, b)
                if imgui.radio_button(f"follow agent {aid}", self.follow_agent_id == aid):
                    self.follow_agent_id = aid
                imgui.pop_style_color()
                if aid < self.num_agents - 1:
                    imgui.same_line()

        imgui.spacing()
        shader_options = ["surfelmap.glsl", "trianglemap.glsl"]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )
        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i
        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()
        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_cams = imgui.checkbox("show_curr_cams", self.show_curr_cams)

        # Full-trajectory overlay. The checkbox is disabled until the agents
        # finish processing -- which is when each per_frame_agent_<id>.ready
        # sentinel appears -- so users can't toggle it while data is missing.
        if not self.full_traj_ready:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            imgui.checkbox("show_full_traj (waiting...)", False)
            imgui.checkbox("show_full_traj_frustums", False)
            imgui.pop_style_var()
        else:
            _, self.show_full_traj = imgui.checkbox(
                "show_full_traj", self.show_full_traj
            )
            # Per-frame camera-frame boxes for the full trajectory. Only
            # meaningful while ``show_full_traj`` is on -- grey it out
            # otherwise so the dependency is obvious.
            if not self.show_full_traj:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                imgui.checkbox(
                    "show_full_traj_frustums (enable show_full_traj)",
                    self.show_full_traj_frustums,
                )
                imgui.pop_style_var()
            else:
                _, self.show_full_traj_frustums = imgui.checkbox(
                    "show_full_traj_frustums", self.show_full_traj_frustums
                )
                if self.show_full_traj_frustums:
                    _, self.full_traj_frustum_scale_factor = imgui.drag_float(
                        "full_traj_frustum_scale_factor",
                        self.full_traj_frustum_scale_factor,
                        0.05,
                        min_value=0.05,
                        max_value=2.0,
                    )
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )
        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        # Agent statistics
        imgui.spacing()
        imgui.separator()
        imgui.text("Agents:")
        kfs = self.global_keyframes.keyframes
        with kfs.lock:
            N = len(kfs)
            ag_ids = kfs.agent_id[:N].cpu().tolist() if N > 0 else []
        for aid in range(self.num_agents):
            r, g, b, _ = agent_color(aid)
            cnt = sum(1 for x in ag_ids if int(x) == aid)
            mode = self.agent_states_list[aid].get_mode()
            mode_name = Mode(int(mode)).name if not isinstance(mode, Mode) else mode.name
            imgui.push_style_color(imgui.COLOR_TEXT, r, g, b)
            imgui.text(f"  agent {aid}: KFs={cnt}  mode={mode_name}")
            imgui.pop_style_color()

        # Per-agent current image thumbnails.
        imgui.spacing()
        gui_size = imgui.get_content_region_available()
        for aid in range(self.num_agents):
            tex = self.curr_imgs[aid].texture
            if tex.size[0] == 0:
                continue
            scale = gui_size[0] / tex.size[0]
            scale = min(self.scale, scale)
            size = (tex.size[0] * scale, tex.size[1] * scale)
            r, g, b, _ = agent_color(aid)
            imgui.push_style_color(imgui.COLOR_TEXT, r, g, b)
            image_with_text(self.curr_imgs[aid], size, f"agent {aid}", same_line=False)
            imgui.pop_style_color()

        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.viz2main.put(self.state)

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    # ---------- pointmap helpers --------------------------------------------

    def _render_pointmap(
        self, T_WC, w, h, ptex, ctex, itex,
        use_img=True, depth_bias=0, base_color=None,
    ):
        """Render a single pointmap with optional per-agent tinting.

        ``base_color`` is forwarded to the fragment shader's ``base_color``
        uniform and is only consulted when ``use_img`` is False (matching the
        shader's branch at ``surfelmap.glsl`` / ``trianglemap.glsl``). Pass
        the agent's RGB to give each robot its own visually distinct cloud.
        """
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = T_WC.matrix().numpy().astype(np.float32).T

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        if base_color is not None and "base_color" in vao.program:
            r, g, b = float(base_color[0]), float(base_color[1]), float(base_color[2])
            vao.program["base_color"].value = (r, g, b)
        vao.render(mode=moderngl.POINTS, vertices=w * h)
        vao.release()

    def _frame_X(self, frame, agent_id: int = -1):
        """Reconstruct world-frame (or rather camera-frame) X for one frame.

        In calib mode we cache a per-pixel ray-direction table keyed by
        ``agent_id`` so different agents with different intrinsics don't
        clobber each other's cache; -1 falls back to a shared "global" slot
        (used for keyframes, which all share the same K via ``kfs.K``).
        """
        if not config["use_calib"]:
            return frame.X_canon.cpu().numpy().astype(np.float32)

        Xs = frame.X_canon[None]
        # Per-agent cache; agent_id < 0 means "shared slot" (keyframes case).
        if agent_id >= 0:
            cache = self._dP_dz_per_agent[agent_id]
        else:
            cache = self.dP_dz

        if cache is None:
            device = Xs.device
            dtype = Xs.dtype
            img_size = frame.img_shape.flatten()[:2]
            K = frame.K
            p = get_pixel_coords(
                Xs.shape[0], img_size, device=device, dtype=dtype
            ).view(*Xs.shape[:-1], 2)
            tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
            tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
            dPdz = torch.empty(
                p.shape[:-1] + (3, 1), device=device, dtype=dtype
            )
            dPdz[..., 0, 0] = tmp1
            dPdz[..., 1, 0] = tmp2
            dPdz[..., 2, 0] = 1.0
            cache = dPdz[..., 0].cpu().numpy().astype(np.float32)
            if agent_id >= 0:
                self._dP_dz_per_agent[agent_id] = cache
            else:
                self.dP_dz = cache

        return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * cache)[0]


def run_multi_agent_visualization(
    cfg,
    agent_states_list,
    global_keyframes,
    main2viz,
    viz2main,
    per_frame_npz_paths=None,
):
    set_global_config(cfg)

    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=MultiAgentWindow.title,
        size=MultiAgentWindow.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = MultiAgentWindow(
        agent_states_list=agent_states_list,
        global_keyframes=global_keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        per_frame_npz_paths=per_frame_npz_paths,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )
    window._config = weakref.ref(window_config)
    window.swap_buffers()
    window.set_default_viewport()
    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()
        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)
        window.use()
        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
