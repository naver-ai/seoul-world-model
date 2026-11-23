# Inference

Generate a driving video along a camera trajectory you choose. You give the model
a folder of **posed reference frames** (RGB + depth + intrinsics + extrinsics), a
start frame, a trajectory, and a text prompt. The model lays down a metric camera
path; for every pose it retrieves the nearest reference, forward-warps it on the
GPU, and generates the video conditioned on those warps. No external services or
pre-computed warps are required.

## Quick start

```bash
DATA_DIR=/path/to/frames CKPT=/path/to/model.pt \
    bash sh/inference/generate.sh
```

That writes `outputs/generate/smooth_left_start0/output.mp4`.

Pick a different trajectory / start frame / prompt:

```bash
DATA_DIR=... CKPT=... TRAJECTORY=smooth_right START_FRAME=4 \
    PROMPT="A car driving down a rainy street at night" \
    bash sh/inference/generate.sh

# generate every trajectory in one go
DATA_DIR=... CKPT=... ALL=1 bash sh/inference/generate.sh
```

## Input data layout

`DATA_DIR` is a folder of posed frames. Every frame contributes one image and its
geometry across four sibling folders that share the same basename — **any naming
scheme works** and the **image extension is free** (`.jpg`, `.jpeg`, `.png`,
`.bmp`, `.webp`, `.tif`/`.tiff`); only the `image ↔ depth/intrinsics/extrinsics`
pairing matters, and the geometry files are always `.npy`:

```
DATA_DIR/
├── images/<name>.{jpg,png,…}  RGB image
├── depth/<name>.npy           depth map            (H, W)        float
├── intrinsics/<name>.npy      camera intrinsics K  (3, 3)
└── extrinsics/<name>.npy      world-to-camera pose (3, 4)
```

Frames are ordered by basename (so `START_FRAME` as an integer indexes this sorted
order — zero-pad names like `frame_007` if you rely on numeric ordering).

Every frame in the folder is treated as a candidate reference. See
[DATA_PREPARATION.md](DATA_PREPARATION.md) for how to produce this layout from a
set of images.

## Trajectories

`TRAJECTORY` selects the camera path, built in metric space from the start frame's
pose (forward = camera +Z, turns rotate about world-up). The set mirrors the OOD
reference trajectories (minus plain straight):

| Trajectory         | Motion |
|--------------------|--------|
| `straight`         | drive straight ahead, no turn |
| `straight_left90`  | drive straight, then turn **left** by `TURN_DEG`, then straight |
| `straight_right90` | drive straight, then turn **right** by `TURN_DEG`, then straight |
| `left90_right90`   | turn **left** by `TURN_DEG`, then an opposite **right** turn (S-curve, net ≈ 0) |
| `right90_left90`   | turn **right** by `TURN_DEG`, then an opposite **left** turn (S-curve, net ≈ 0) |
| `smooth_left`      | drive forward while turning **left**  by `TURN_DEG` continuously over the clip |
| `smooth_right`     | drive forward while turning **right** by `TURN_DEG` continuously over the clip |

`SPEED` is the forward step per frame in meters. `TURN_DEG` is the magnitude of each
turn (default `90`). `TURN_RATE` is how fast the discrete `*90` turns ramp, in degrees
per frame (default `4`); `smooth_*` ignore it and spread the turn over the whole clip.
The discrete turns are timed to land where the OOD reference set places them at the
default 2-chunk / 149-frame config (single turn ≈ frame 35; S-curve at ≈ frames 20 & 87).

## Autoregressive generation

Each run produces **2 AR chunks** chained into one continuous video:

- Chunk 0: 1 conditional frame (the start image) + 76 generated
- Chunk 1: 5 conditional frames (last 5 of chunk 0) + 72 generated

The 5-pixel (≈2 latent) overlap gives temporal continuity. Per pose the dataset
retrieves the closest reference by **position, then most similar viewing
direction**: references within `GROUP_TOL` metres of the nearest candidate are
treated as the same spot, and among them the one whose viewing direction best
matches the target pose wins. The look-ahead frame is the closest reference
**facing forward** from the chunk's last pose.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `DATA_DIR` | (required) | Folder with `images/ depth/ intrinsics/ extrinsics/` |
| `CKPT` | (required) | Path to a consolidated checkpoint file (`model.pt`) |
| `TRAJECTORY` | `smooth_left` | `straight` \| `straight_left90` \| `straight_right90` \| `left90_right90` \| `right90_left90` \| `smooth_left` \| `smooth_right` |
| `START_FRAME` | `0` | Start frame — an image name, or an index into the sorted frames |
| `PROMPT` | (Seoul driving prompt) | Text prompt |
| `SPEED` | `0.3` | Forward step per frame, in meters |
| `TURN_DEG` | `90` | Magnitude of each turn, in degrees |
| `TURN_RATE` | `4` | Turn speed for the `*90` turns, in degrees per frame |
| `GROUP_TOL` | `2.0` | Retrieval radius in meters — refs within this distance of the nearest count as one spot, then the closest viewing direction wins |
| `ALL` | (off) | Set non-empty to generate every trajectory sequentially |
| `GPU` | `0` | GPU index |
| `OUT` | (unset) | Exact output directory (videos written straight here) |
| `OUT_ROOT` | `outputs/generate` | Output root; used as `OUT_ROOT/<traj>_start<START_FRAME>` when `OUT` is unset |
| `GUIDANCE` | `7.0` | CFG guidance scale |
| `NUM_STEPS` | `35` | Sampling steps per chunk |
| `SEED` | `42` | RNG seed |
| `TEASER` | (unset) | Set to `1` to regenerate the teaser from the sample folder (`DATA_DIR=data/swm_sample`, reads `teaser.json`; the trajectory variables are not used; output `OUT` or `OUT_ROOT/teaser`) |

## Output layout

Videos are written straight into the output directory (`OUT`, or
`OUT_ROOT/<trajectory>_start<START_FRAME>` if `OUT` is unset):

```
<output dir>/
├── output.mp4            # the full trajectory video
├── visualization.mp4     # [forward-warp condition | generated] + ref / look-ahead
└── .logs/                # run logs (console.log) — hidden
```

`visualization.mp4` shows, side by side, the warped frames fed to the model and the
frames it generated — handy for sanity-checking the conditioning.

## Guardrails

Every generation runs [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1), the
guardrails Cosmos-Predict2.5 runs in its own inference. They cannot be turned off.

- **Prompt.** Each prompt is checked with a keyword blocklist and
  [Qwen3Guard](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) before it conditions the model.
  An unsafe prompt stops the run before anything is generated.
- **Generated video.** The finished video is checked by the video content safety filter (a
  [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)-based classifier); a video
  whose frames are more than 10% unsafe is not saved. Faces in the saved frames are then
  pixelated with RetinaFace.

The video checks run once the whole video has been generated, so they do not change what the
model generates or how chunks are conditioned on each other. The warp and reference panels in
`visualization.mp4` show the input data as-is.

Cosmos-Guardrail1 is gated: accept its license on the model page and run
`huggingface-cli login` once. The guardrail models (Cosmos-Guardrail1, about 7 GB, plus
Qwen3Guard-Gen-0.6B and SigLIP) are downloaded on the first run with `HF_HUB_OFFLINE=0`.

## Underneath

`generate.sh` runs:

```
torchrun --nproc_per_node=1 -m scripts.inference_ar \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --guidance=$GUIDANCE --num-steps=$NUM_STEPS --n-samples=1 --num-chunks=2 \
    --fps=10 --seed=$SEED --output-dir="$OUT" --la-mode=normal \
    -- experiment=predict2_generate \
    checkpoint.load_path="$CKPT" checkpoint.strict_resume=False job.wandb_mode=disabled
```

The experiment `predict2_generate`
(`cosmos_predict2/experiments/base/generate.py`) wires up:

- Dataset: `TrajectoryDataset` (retrieval + GPU forward-warp; reads `DATA_DIR`)
- Model: `fsdp_rectified_flow_warp_ref_LA`
- Net: `cosmos_v1_2B_warp_ref_LA_cam`

`CKPT` is a consolidated `model.pt` for this model + net. Training writes sharded
DCP checkpoints (`iter_XXXXXXXXX/` directories); convert one to a single `.pt` first:

```bash
python scripts/convert_distcp_to_pt.py /path/to/iter_000008000/model /path/to/iter_000008000
# -> /path/to/iter_000008000/model.pt
```
