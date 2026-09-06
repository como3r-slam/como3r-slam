"""Split one image sequence into K agent streams with a controlled overlap.

Each agent sees a contiguous window of the sequence, and adjacent windows
share a fraction of the frames -- the only place the agents' maps can be
linked. The result mirrors what a real team produces: independent streams,
independent reference frames, limited co-visibility.

Layout produced for a scene directory holding `image/`:

    <scene>/agent1/*.jpg    symlinks into ../image/
    <scene>/agent2/*.jpg
    ...

For N frames, K agents and an adjacent-pair overlap fraction `o` (in units of
N), the window length is L = round(N * (1 + (K-1) * o) / K) and agent k starts
at round(k * (N - L) / (K - 1)), so consecutive agents share about o * N frames.
"""

import argparse
import os
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def compute_windows(n: int, k: int, overlap: float):
    if k < 1:
        raise ValueError("num_agents must be >= 1")
    if k == 1:
        return [(0, n)]
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    L = int(round(n * (1.0 + (k - 1) * overlap) / k))
    L = max(1, min(n, L))
    step = (n - L) / (k - 1)
    windows = []
    for i in range(k):
        start = int(round(i * step))
        end = start + L
        if i == k - 1:
            end = n
            start = max(0, end - L)
        windows.append((start, end))
    return windows


def split_scene(scene_dir: Path, num_agents: int, overlap: float, force: bool) -> None:
    src = scene_dir / "image"
    if not src.is_dir():
        raise SystemExit(f"{scene_dir}: no image/ folder (run scripts/download_barn.sh first)")

    frames = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    n = len(frames)
    if n == 0:
        raise SystemExit(f"{src} is empty")

    windows = compute_windows(n, num_agents, overlap)
    for aid, (start, end) in enumerate(windows, start=1):
        dst = scene_dir / f"agent{aid}"
        if dst.exists():
            if not force:
                print(f"[skip] {dst} already exists (use --force to recreate)")
                continue
            for p in dst.iterdir():
                p.unlink()
            dst.rmdir()
        dst.mkdir()
        for f in frames[start:end]:
            (dst / f.name).symlink_to(os.path.relpath(f, dst))

    summary = "  ".join(f"agent{i+1}=[{s},{e})" for i, (s, e) in enumerate(windows))
    overlaps = [max(0, windows[i][1] - windows[i + 1][0]) for i in range(len(windows) - 1)]
    ov_str = ", ".join(f"{ov} ({ov / n:.1%})" for ov in overlaps) or "n/a"
    print(f"[done] {scene_dir}: N={n}  K={num_agents}  {summary}  overlap={ov_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", type=Path, default=Path("data/Barn"),
                    help="Scene directory holding an image/ folder")
    ap.add_argument("--num-agents", "-k", type=int, default=2, help="Number of agents (>=1)")
    ap.add_argument("--overlap", "-o", type=float, default=0.05,
                    help="Adjacent-pair overlap as a fraction of the sequence length")
    ap.add_argument("--force", action="store_true", help="Overwrite existing agentN/ folders")
    args = ap.parse_args()
    split_scene(args.scene, args.num_agents, args.overlap, args.force)


if __name__ == "__main__":
    main()
