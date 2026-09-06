"""CoMo3R-SLAM entry point: collaborative monocular dense SLAM.

Usage:
    python multi_agent_main.py \
        --agents data/Barn/agent1 data/Barn/agent2 \
        --config config/multi_agent.yaml

Each --agents path is a folder of RGB images, one folder per agent. The
system spawns one process per agent (tracking + local Sim(3) Gauss-Newton)
and one coordinator process that retrieves cross-agent keyframes, verifies
them by dense pointmap matching, synchronizes the agents' Sim(3) gauges and
refines every keyframe in a single multi-agent Sim(3) graph.

The default config runs uncalibrated. Pass --calib <intrinsics.yaml> together
with a config that sets use_calib: True to feed known pinhole intrinsics in.
"""
import argparse
import contextlib
import datetime
import os
import pathlib
import re
import subprocess

# `expandable_segments:True` cuts allocator fragmentation in the worker
# processes, which churn large short-lived match tensors around long-lived
# growing buffers -- but a tensor allocated inside an expandable segment can
# only be handed to another process via pidfd_getfd, which many container
# runtimes block ("RuntimeError: pidfd_getfd: Operation not permitted" while a
# child rebuilds a shared CUDA tensor). THIS process owns every cross-process
# buffer (global keyframe store, SharedStates, MASt3R weights), so it must
# always allocate with the default allocator. Workers only consume those
# handles and never export their own, so they can opt in via
# --expandable-segments.
_EXPANDABLE = "expandable_segments:True"
if _EXPANDABLE.split(":")[0] in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    print(
        "[main] WARNING: PYTORCH_CUDA_ALLOC_CONF requests expandable_segments; "
        "clearing it for the main process (it breaks CUDA IPC sharing of the "
        "keyframe store). Use --expandable-segments to enable it for the "
        "worker processes instead."
    )
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]


@contextlib.contextmanager
def _worker_alloc_env(enabled):
    """Apply the worker allocator config to processes started inside the block.

    Spawned children inherit os.environ as it is at ``start()`` time, so this
    is how a worker gets a different allocator config from its parent.
    """
    if not enabled:
        yield
        return
    prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _EXPANDABLE
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = prev


import lietorch
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from como3r_slam.agent import run_agent
from como3r_slam.config import config, load_config
from como3r_slam.coordinator import run_coordinator
from como3r_slam.dataloader import Intrinsics, load_dataset
from como3r_slam.frame import SharedStates
from como3r_slam.lietorch_utils import as_SE3
from como3r_slam.mast3r_utils import load_mast3r
from como3r_slam.multi_agent import MultiAgentKeyframes
from como3r_slam.visualization import run_multi_agent_visualization
from como3r_slam.multiprocess_utils import new_queue
from como3r_slam.ply import save_ply

def _build_frame_maps(agent_paths, subsample):
    """For each agent, map its (subsampled) dataset index -> TUM timestamp.

    The timestamp is derived from the image filename so that an agent's
    estimate associates row-for-row with the dataset's gt_traj.txt:

    * Waymo: the filename *is* a float timestamp (``1550002043.729173.jpg``)
      and gt_traj.txt uses the same float timestamps -> parse the whole stem.
    * TNT (``000196.jpg`` -> 196) / Replica (``frame000123.jpg`` -> 123):
      the trailing integer of the filename is the frame number, which is
      what gt_traj.txt is keyed on.
    """
    maps = []
    for ds_path in agent_paths:
        ds = load_dataset(ds_path)
        ds.subsample(subsample)
        idx2frame = []
        for k, f in enumerate(ds.rgb_files):
            stem = pathlib.Path(f).stem
            if re.fullmatch(r"\d+\.\d+", stem):
                # Waymo: filename is a float timestamp; keep it intact.
                idx2frame.append(float(stem))
            else:
                # TNT / Replica: trailing integer is the frame number.
                digits = re.findall(r"\d+", stem)
                idx2frame.append(int(digits[-1]) if digits else k * subsample)
        maps.append(idx2frame)
    return maps


def _save_agent_trajectories(global_keyframes, num_agents, save_dir, frame_maps):
    """Save one TUM-style keyframe-only trajectory file per agent, in the global frame."""
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for agent_id in range(num_agents):
        indices = global_keyframes.get_local_to_global_map(agent_id)
        out = save_dir / f"agent_{agent_id}.txt"
        with open(out, "w") as f:
            for global_idx in indices:
                kf = global_keyframes.keyframes[global_idx]
                frame_id = int(global_keyframes.keyframes.dataset_idx[global_idx])
                t = frame_maps[agent_id][frame_id]
                T_WC = as_SE3(kf.T_WC)
                x, y, z, qx, qy, qz, qw = T_WC.data.cpu().numpy().reshape(-1)
                f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
        print(f"[main] wrote {out}")
        paths.append(out)
    return paths


def _save_agent_full_trajectories(
    global_keyframes, num_agents, save_dir, frame_maps, per_frame_npz_paths
):
    """Save one TUM-style full (every-frame) trajectory file per agent.

    KFs are written using their final global pose. Non-KFs are reconstructed
    via T_global = T_anchor_global @ T_rel, where T_rel was recorded at
    tracking time (frame-invariant) and the anchor's pose is read AFTER all
    Level-2 solves have converged.
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for agent_id in range(num_agents):
        # Collect KF entries.
        entries = {}  # frame_id -> SE3 data tuple (x, y, z, qx, qy, qz, qw)
        kf_indices = global_keyframes.get_local_to_global_map(agent_id)
        for global_idx in kf_indices:
            kf = global_keyframes.keyframes[global_idx]
            frame_id = int(global_keyframes.keyframes.dataset_idx[global_idx])
            T_WC_se3 = as_SE3(kf.T_WC)
            entries[frame_id] = T_WC_se3.data.cpu().numpy().reshape(-1)

        # Collect non-KF entries from this agent's per-frame npz.
        npz_path = per_frame_npz_paths.get(agent_id)
        if npz_path is not None and pathlib.Path(npz_path).exists():
            data = np.load(npz_path)
            frame_ids = data["frame_ids"]
            anchor_global_idx = data["anchor_global_idx"]
            T_rel = data["T_rel"]  # (M, 8) Sim3 data
            for k in range(frame_ids.shape[0]):
                fid = int(frame_ids[k])
                if fid in entries:
                    # KF entry already present (shouldn't happen for non-KFs).
                    continue
                aidx = int(anchor_global_idx[k])
                T_anchor = lietorch.Sim3(
                    global_keyframes.keyframes.T_WC[aidx].clone().cpu()
                )
                T_rel_sim3 = lietorch.Sim3(
                    torch.from_numpy(T_rel[k]).reshape(1, 8).to(torch.float32)
                )
                T_global = T_anchor * T_rel_sim3
                T_global_se3 = as_SE3(T_global)
                entries[fid] = T_global_se3.data.cpu().numpy().reshape(-1)
        else:
            print(f"[main] WARNING: no per-frame records for agent {agent_id} "
                  f"(expected at {npz_path})")

        out = save_dir / f"agent_{agent_id}_full.txt"
        with open(out, "w") as f:
            for fid in sorted(entries):
                x, y, z, qx, qy, qz, qw = entries[fid]
                t = frame_maps[agent_id][fid]
                f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
        print(f"[main] wrote {out} ({len(entries)} frames)")
        paths.append(out)
    return paths


def _run_evo_ape_sim3(gt_path, est_path):
    """Invoke evo_ape with Sim(3) alignment (`-as`) and parse RMSE out of stdout.

    Returns (rmse, raw_stdout) or (None, raw_stderr) on failure.
    """
    cmd = ["evo_ape", "tum", str(gt_path), str(est_path), "-as"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return None, "evo_ape not found in PATH"
    if res.returncode != 0:
        return None, res.stderr + res.stdout
    # evo_ape prints a block like:
    #     APE w.r.t. translation part (m)
    #     (with Sim(3) Umeyama alignment)
    #
    #     max         ...
    #     mean        ...
    #     median      ...
    #     min         ...
    #     rmse        0.123456
    #     sse         ...
    #     std         ...
    m = re.search(r"^\s*rmse\s+([0-9.eE+\-]+)\s*$", res.stdout, re.MULTILINE)
    if not m:
        return None, res.stdout
    return float(m.group(1)), res.stdout


def _save_joint_reconstruction(global_keyframes, save_path, c_conf_threshold=1.0):
    """Save a single PLY combining point clouds from all agents in the global frame."""
    from como3r_slam.geometry import constrain_points_to_ray

    pointclouds = []
    colors = []
    n = len(global_keyframes)
    for i in range(n):
        kf = global_keyframes.keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                kf.img_shape.flatten()[:2], kf.X_canon[None], kf.K
            )
            kf.X_canon = X_canon.squeeze(0)
        pW = kf.T_WC.act(kf.X_canon).cpu().numpy().reshape(-1, 3)
        color = (kf.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            kf.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    if pointclouds:
        pts = np.concatenate(pointclouds, axis=0)
        cols = np.concatenate(colors, axis=0)
        save_ply(save_path, pts, cols)
        print(f"[main] wrote {save_path} ({pts.shape[0]} points)")


def main():
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="Paths to per-agent image folders (RGB sequences).",
    )
    parser.add_argument("--config", default="config/multi_agent.yaml")
    parser.add_argument(
        "--calib",
        default=None,
        help="Pinhole intrinsics YAML ([fx, fy, cx, cy], optionally with "
        "distortion coefficients). Only needed when the config sets "
        "use_calib: True -- the default config runs uncalibrated.",
    )
    parser.add_argument("--save-as", default="multi_agent")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Cap each agent's stream to this many frames (after subsample). -1 = no cap.",
    )
    parser.add_argument(
        "--expandable-segments",
        action="store_true",
        help="Run the agent/coordinator/viz processes with "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Reduces allocator "
        "fragmentation on long sequences, but fails on hosts whose container "
        "runtime blocks pidfd_getfd (children die with 'pidfd_getfd: Operation "
        "not permitted' while rebuilding shared CUDA tensors).",
    )
    parser.add_argument(
        "--global-kf-buffer",
        type=int,
        default=None,
        help="Slots in the global (cross-agent) keyframe store. Must cover the "
        "TOTAL keyframe count of all agents; it is shared across processes and "
        "cannot grow at runtime. Overrides multi_agent.global_kf_buffer.",
    )
    args = parser.parse_args()

    load_config(args.config)
    cfg = dict(config)
    num_agents = len(args.agents)
    print(f"[main] starting multi-agent SLAM: {num_agents} agents")
    print(f"[main] config: {args.config}")
    print(f"[main] calib:  {args.calib if cfg['use_calib'] else 'none (uncalibrated)'}")

    intrinsics_yaml = None
    if cfg["use_calib"]:
        if args.calib is None:
            raise SystemExit(
                "this config sets use_calib: True, so --calib <intrinsics.yaml> "
                "is required"
            )
        with open(args.calib, "r") as f:
            intrinsics_yaml = yaml.safe_load(f)

    # Build K_frame to seed the global keyframe store. We use the dataset's
    # actual raw image dimensions for the (W, H) that feed Intrinsics.from_calib
    # -- the width/height fields in the calib YAML are hints that often mismatch
    # the real resolution, and K_frame has to match the input pixels.
    dataset_probe = load_dataset(args.agents[0])
    dataset_probe.use_calibration = cfg["use_calib"]
    raw_h, raw_w = dataset_probe.get_img_shape()[1]
    if cfg["use_calib"]:
        yaml_w = int(intrinsics_yaml.get("width", raw_w))
        yaml_h = int(intrinsics_yaml.get("height", raw_h))
        if (yaml_w, yaml_h) != (raw_w, raw_h):
            print(
                f"[main] WARNING: calib YAML width/height = ({yaml_w}, {yaml_h}) "
                f"but dataset images are ({raw_w}, {raw_h}); using the dataset's "
                f"actual shape so K_frame is consistent with the input pixels"
            )
        dataset_probe.camera_intrinsics = Intrinsics.from_calib(
            dataset_probe.img_size,
            raw_w,
            raw_h,
            intrinsics_yaml["calibration"],
        )
    h, w = dataset_probe.get_img_shape()[0]
    print(
        f"[main] dataset raw image shape: {raw_h}x{raw_w}, "
        f"resized keyframe shape: {h}x{w}"
    )

    K = None
    if cfg["use_calib"]:
        K = torch.from_numpy(dataset_probe.camera_intrinsics.K_frame).to(
            "cuda:0", dtype=torch.float32
        )

    manager = mp.Manager()
    global_kf_buffer = args.global_kf_buffer or int(
        cfg.get("multi_agent", {}).get("global_kf_buffer", 512)
    )
    global_keyframes = MultiAgentKeyframes(
        manager, h, w, num_agents, buffer=global_kf_buffer
    )
    slot_mb = global_keyframes.keyframes.slot_bytes() / (1 << 20)
    print(
        f"[main] global keyframe store: {global_kf_buffer} slots "
        f"(~{slot_mb:.1f} MB/slot, ~{global_kf_buffer * slot_mb / 1024:.1f} GB GPU); "
        f"per-agent local stores start at "
        f"{int(cfg.get('multi_agent', {}).get('local_kf_buffer', 256))} slots and grow on demand"
    )
    if K is not None:
        global_keyframes.keyframes.set_intrinsics(K)

    # Per-agent SharedStates allocated in main so the visualizer can read them.
    agent_states_list = [
        SharedStates(manager, h, w, device="cuda:0") for _ in range(num_agents)
    ]

    agent_to_coord_qs = [mp.Queue() for _ in range(num_agents)]
    coord_to_agent_qs = [mp.Queue() for _ in range(num_agents)]

    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    model = load_mast3r(device="cuda:0")
    model.share_memory()

    # Gauge anchor: ensure global KF index 0 is always agent 0's first keyframe.
    # Agent 0 sets agent0_inited after appending its first KF; all other agents
    # wait on it before starting their own INIT.
    agent0_inited = manager.Event()

    coord_proc = mp.Process(
        target=run_coordinator,
        args=(
            cfg,
            model,
            global_keyframes,
            agent_to_coord_qs,
            coord_to_agent_qs,
            K,
            num_agents,
        ),
    )
    with _worker_alloc_env(args.expandable_segments):
        coord_proc.start()

    # Pre-create save_dir so we can hand each agent a per-frame npz path; the
    # agent writes its own per-frame records (frame_id, anchor_global_idx,
    # T_rel) to that file at shutdown so main can compose the full trajectory.
    save_dir = pathlib.Path("logs") / args.save_as
    save_dir.mkdir(parents=True, exist_ok=True)
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")
    print(f"[main] saving results to {save_dir} (run {datetime_now})")
    per_frame_npz_paths = {
        agent_id: save_dir / f"per_frame_agent_{agent_id}.npz"
        for agent_id in range(num_agents)
    }
    # Delete stale .ready sentinels from any prior run; the viz keys the
    # full-traj toggle off these sentinels so a leftover would make it
    # display the previous run's trajectory composed with the current run's
    # (uninitialized) global poses.
    for p in per_frame_npz_paths.values():
        ready = p.parent / f"{p.stem}.ready"
        if ready.exists():
            ready.unlink()

    viz_proc = None
    if not args.no_viz:
        viz_proc = mp.Process(
            target=run_multi_agent_visualization,
            args=(
                cfg,
                agent_states_list,
                global_keyframes,
                main2viz,
                viz2main,
                per_frame_npz_paths,
            ),
        )
        with _worker_alloc_env(args.expandable_segments):
            viz_proc.start()

    agent_procs = []
    for agent_id, ds_path in enumerate(args.agents):
        p = mp.Process(
            target=run_agent,
            args=(
                agent_id,
                cfg,
                model,
                ds_path,
                intrinsics_yaml if cfg["use_calib"] else None,
                global_keyframes,
                agent_states_list[agent_id],
                agent_to_coord_qs[agent_id],
                coord_to_agent_qs[agent_id],
                agent0_inited,
                "cuda:0",
                args.max_frames,
                str(per_frame_npz_paths[agent_id]),
            ),
        )
        with _worker_alloc_env(args.expandable_segments):
            p.start()
        agent_procs.append(p)

    for p in agent_procs:
        p.join()
    coord_proc.join()

    # A child that died mid-stream (OOM, buffer overflow, ...) leaves a
    # truncated map behind, and every number computed below -- trajectories,
    # ATE, reconstruction -- silently describes that truncated run. Collect the
    # failures now and shout about them at the end rather than let the ATE
    # table imply a clean run.
    crashed = [
        f"agent {i}" for i, p in enumerate(agent_procs) if p.exitcode not in (0, None)
    ]
    if coord_proc.exitcode not in (0, None):
        crashed.append("coordinator")
    if crashed:
        print(
            f"\n[main] *** RUN INCOMPLETE: {', '.join(crashed)} exited with an "
            f"error (see the traceback above). Results below are computed from "
            f"a TRUNCATED map and are not valid. ***\n"
        )
    if viz_proc is not None:
        # All processing is done; the viz process stays alive until the user
        # closes the window. Join here so save-out happens after inspection.
        print("[main] processing complete -- close the viz window to save results")
        viz_proc.join()

    subsample = int(cfg["dataset"]["subsample"])
    frame_maps = _build_frame_maps(args.agents, subsample)
    kf_paths = _save_agent_trajectories(
        global_keyframes, num_agents, save_dir, frame_maps
    )
    full_paths = _save_agent_full_trajectories(
        global_keyframes, num_agents, save_dir, frame_maps, per_frame_npz_paths
    )
    _save_joint_reconstruction(
        global_keyframes,
        save_dir / "joint_reconstruction.ply",
        c_conf_threshold=1.0,
    )

    # ATE against a TUM-format gt_traj.txt at the scene root (timestamps =
    # image frame numbers); evo associates each agent's estimate with the
    # matching subset of rows. Skipped when the dataset ships no ground truth.
    print("\n[main] === ATE evaluation (Sim(3) Umeyama alignment) ===")
    for agent_id, ds_path in enumerate(args.agents):
        ds_path_p = pathlib.Path(ds_path)
        # `--agents` may point at the per-agent folder OR its `color/` subdir,
        # so the scene root is either that folder or its parent.
        gt_path = next(
            (c for c in (ds_path_p / "gt_traj.txt",
                         ds_path_p.parent / "gt_traj.txt") if c.exists()),
            None,
        )
        if gt_path is None:
            print(
                f"[main] agent {agent_id}: no gt_traj.txt found under "
                f"{ds_path} -- skipping ATE"
            )
            continue

        kf_rmse, kf_log = _run_evo_ape_sim3(gt_path, kf_paths[agent_id])
        full_rmse, full_log = _run_evo_ape_sim3(gt_path, full_paths[agent_id])
        print(f"[main] agent {agent_id}:")
        if kf_rmse is not None:
            print(f"        ATE (keyframes only) RMSE = {kf_rmse:.6f} m")
        else:
            print(f"        ATE (keyframes only) FAILED:\n{kf_log}")
        if full_rmse is not None:
            print(f"        ATE (full trajectory) RMSE = {full_rmse:.6f} m")
        else:
            print(f"        ATE (full trajectory) FAILED:\n{full_log}")

    if crashed:
        print(
            f"[main] *** RUN INCOMPLETE ({', '.join(crashed)} crashed) -- the "
            f"ATE numbers above are from a truncated map ***"
        )
    print("[main] done")


if __name__ == "__main__":
    main()
