#!/bin/bash
# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
# =============================================================================
# Run a fixed demo scenario (prompt + trajectory preset) end to end.
#
# Usage:
#   SCENARIO=demo_scenarios/scenario.json bash run_demo.sh
#
# Each scenario JSON sets the prompt, trajectory, start frame and seed; the
# camera poses are taken as-is from the data. Defaults (override via env):
#   CKPT=checkpoints/model.pt   DATA_DIR=data/swm_sample   GPU=0
#
# Output: outputs/<scenario>/output.mp4  +  visualization.mp4
# =============================================================================
set -e
cd "$(dirname "$0")"          # repo root
REPO="$(pwd)"

SCENARIO=${SCENARIO:?set SCENARIO to a demo_scenarios/*.json file}
[ -f "$SCENARIO" ] || { echo "Scenario not found: $SCENARIO"; exit 1; }

export CKPT=${CKPT:-"$REPO/checkpoints/model.pt"}
export DATA_DIR=${DATA_DIR:-"$REPO/data/swm_sample"}
export GPU=${GPU:-0}

# Pull the preset fields from the scenario JSON into generate.sh's env vars.
eval "$(.venv/bin/python - "$SCENARIO" <<'PY'
import json, sys, shlex
s = json.load(open(sys.argv[1]))
for env_key, json_key in [("TRAJECTORY", "trajectory"), ("START_FRAME", "start_frame"),
                          ("PROMPT", "prompt"), ("SPEED", "speed"), ("SEED", "seed")]:
    if s.get(json_key) is not None:
        print(f"export {env_key}={shlex.quote(str(s[json_key]))}")
PY
)"

name="$(basename "$SCENARIO" .json)"
export OUT=${OUT:-"$REPO/outputs/$name"}

echo "== demo scenario: $name  (trajectory=$TRAJECTORY  start=$START_FRAME)"
exec bash sh/inference/generate.sh
