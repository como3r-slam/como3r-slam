#!/usr/bin/env bash
# Download the Tanks and Temples "Barn" sequence (410 RGB frames, 1920x1080)
# and lay it out as data/Barn/ for the two-agent example.
#
#     data/Barn/image/000001.jpg ... 000410.jpg
#     data/Barn/gt_traj.txt        (reference trajectory, shipped with this repo)
#
# The archive is ~527 MB. Set BARN_URL to fetch it from somewhere else, e.g. a
# copy you downloaded by hand from https://www.tanksandtemples.org/download/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$ROOT/data/Barn"
ZIP="$ROOT/data/Barn.zip"
URL="${BARN_URL:-https://huggingface.co/datasets/hongliu6/tanks_and_temples/resolve/main/Barn.zip}"

mkdir -p "$ROOT/data"

if [ -d "$SCENE/image" ] && [ "$(ls "$SCENE/image" | wc -l)" -eq 410 ]; then
    echo "[barn] $SCENE/image already holds 410 frames, skipping download"
else
    if [ ! -f "$ZIP" ]; then
        echo "[barn] downloading Barn.zip (~527 MB) from $URL"
        curl -L --fail --retry 3 -C - -o "$ZIP" "$URL"
    else
        echo "[barn] reusing $ZIP"
    fi

    echo "[barn] extracting"
    rm -rf "$SCENE/image"
    mkdir -p "$SCENE"
    unzip -q -o "$ZIP" -d "$SCENE/.unpack"
    # The archive holds a single Barn/ folder of jpgs.
    mv "$SCENE/.unpack/Barn" "$SCENE/image"
    rm -rf "$SCENE/.unpack"
fi

cp "$ROOT/examples/barn/gt_traj.txt" "$SCENE/gt_traj.txt"

N=$(ls "$SCENE/image" | wc -l)
echo "[barn] $SCENE/image: $N frames"
if [ "$N" -ne 410 ]; then
    echo "[barn] WARNING: expected 410 frames, got $N"
fi
echo "[barn] done -- next: python scripts/split_agents.py --scene data/Barn --num-agents 2"
