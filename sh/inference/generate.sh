#!/bin/bash
# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
# =============================================================================
# Generate a driving video along a chosen camera trajectory.
#
# Give a folder of posed reference frames (image + depth + intrinsics + extrinsics),
# pick a start frame, a trajectory and a prompt — the model retrieves the nearest
# references along the trajectory and forward-warps them to condition generation.
#
# Usage:
#   DATA_DIR=/path/to/frames bash sh/inference/generate.sh
#   DATA_DIR=... TRAJECTORY=smooth_right START_FRAME=4 bash sh/inference/generate.sh
#   DATA_DIR=... ALL=1 bash sh/inference/generate.sh      # generate every trajectory
#   TEASER=1 DATA_DIR=data/swm_sample bash sh/inference/generate.sh   # regenerate the teaser video
#
# Inputs (env vars):
#   DATA_DIR     folder with images/ depth/ intrinsics/ extrinsics/   (required)
#   TRAJECTORY   straight_left90 | straight_right90 | left90_right90 |
#                right90_left90 | smooth_left | smooth_right           (default: smooth_left)
#   START_FRAME  index of the forward-facing frame to start from      (default: 0)
#   PROMPT       literal text prompt (wins if set)
#   PROMPT_KEY   pick a named prompt from PROMPT_FILE (a {key: prompt} JSON)
#   PROMPT_FILE  prompt library path (default sh/inference/teaser_prompts.json)
#   SPEED        forward step per frame, in meters                    (default: 0.3)
#   TURN_DEG     magnitude of each turn, in degrees                   (default: 90)
#   TURN_RATE    turn speed for the *90 turns, in deg per frame       (default: 4)
#   GROUP_TOL    retrieval radius in meters (same-spot tolerance)     (default: 2.0)
#   OUT          exact output directory (videos written straight here)
#   OUT_ROOT     output root; used as OUT_ROOT/<trajectory>_start<START_FRAME> if OUT unset
#   CKPT / GPU / SEED / NUM_STEPS / GUIDANCE
#   TEASER       set to 1 to regenerate the teaser from the sample folder (DATA_DIR = the
#                sample, with teaser.json); TRAJECTORY / START_FRAME / SPEED / TURN_* /
#                GROUP_TOL / PROMPT_KEY are not used; output: OUT or OUT_ROOT/teaser
#
# Output (directly in the chosen dir): output.mp4 (the generated video) and
# visualization.mp4 ([condition | generated] side by side, for sanity-checking).
# Run logs go to a hidden .logs/ subdir.
# =============================================================================
set -e
cd "$(dirname "$0")/../.."   # repo root

: "${DATA_DIR:?set DATA_DIR to a folder with images/ depth/ intrinsics/ extrinsics/}"
GPU=${GPU:-0}
CKPT=${CKPT:?set CKPT to the model checkpoint directory}
START_FRAME=${START_FRAME:-0}
SPEED=${SPEED:-0.3}
TURN_DEG=${TURN_DEG:-90}
TURN_RATE=${TURN_RATE:-4}
GUIDANCE=${GUIDANCE:-7.0}; NUM_STEPS=${NUM_STEPS:-35}; SEED=${SEED:-42}
OUT_ROOT=${OUT_ROOT:-"outputs/generate"}

launch () {  # launch <experiment> <output dir>
  # Default to offline (reproducible), but honor an external override so the base
  # net / T5 / VAE / Cosmos-Guardrail1 models can be fetched on first run: HF_HUB_OFFLINE=0 bash generate.sh
  export CUDA_VISIBLE_DEVICES="$GPU"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  .venv/bin/torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-$((13300 + RANDOM % 200))}" -m scripts.inference_ar \
      --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
      --guidance="$GUIDANCE" --num-steps="$NUM_STEPS" --n-samples=1 --num-chunks=2 \
      --fps=10 --seed="$SEED" --output-dir="$2" --la-mode=normal \
      -- experiment="$1" \
      checkpoint.load_path="$CKPT" checkpoint.strict_resume=False job.wandb_mode=disabled
  echo ">> $2/output.mp4"
}

run_one () {
  local traj="$1"
  local out="${OUT:-$OUT_ROOT/${traj}_start${START_FRAME}}"
  echo "== generate  traj=$traj  start=$START_FRAME  speed=${SPEED}m  -> $out"
  export DATA_DIR TRAJECTORY="$traj" START_FRAME SPEED TURN_DEG TURN_RATE
  [ -n "$GROUP_TOL" ] && export GROUP_TOL
  [ -n "$PROMPT" ] && export PROMPT
  [ -n "$PROMPT_KEY" ] && export PROMPT_KEY
  [ -n "$PROMPT_FILE" ] && export PROMPT_FILE
  launch predict2_generate "$out"
}

if [ "${TEASER:-0}" = 1 ]; then
  [ -f "$DATA_DIR/teaser.json" ] || { echo "TEASER=1: $DATA_DIR/teaser.json not found (DATA_DIR must be the sample folder)"; exit 1; }
  out="${OUT:-$OUT_ROOT/teaser}"
  echo "== teaser  data=$DATA_DIR  -> $out"
  export DATA_DIR
  [ -n "$PROMPT" ] && export PROMPT
  launch predict2_teaser "$out"
elif [ -n "$ALL" ]; then
  for t in straight_left90 straight_right90 left90_right90 right90_left90 smooth_left smooth_right; do run_one "$t"; done
else
  run_one "${TRAJECTORY:-smooth_left}"
fi
