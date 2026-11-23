# Data Preparation

## Sample data

Download the sample data from Hugging Face:

```bash
huggingface-cli download naver-ai/seoul-world-model --include "swm_sample/*" --local-dir data
# -> data/swm_sample/
```

`data/swm_sample/` is a `DATA_DIR` (see below) of 52 posed street-view frames, plus the images
the teaser video ([`assets/sample-teaser.mp4`](https://huggingface.co/naver-ai/seoul-world-model/resolve/main/assets/sample-teaser.mp4)) was conditioned on (`tokens/`) and `teaser.json`.
Run the scenario on it (see [README](README.md)), or regenerate the teaser:

```bash
SCENARIO=demo_scenarios/scenario.json bash run_demo.sh
TEASER=1 DATA_DIR=data/swm_sample CKPT=checkpoints/model.pt bash sh/inference/generate.sh
```

## Your own data

`DATA_DIR` is a folder of posed frames with four sibling folders — `images/`, `depth/`,
`intrinsics/`, `extrinsics/` — one file per frame sharing the same basename (`.npy` for the
arrays). Produce them from raw images with
[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (joint
depth / intrinsics / extrinsics; its `DA3NESTED-GIANT-LARGE-1.1` weights are licensed for
non-commercial use under CC BY-NC 4.0), then point a scenario at it:

```bash
DATA_DIR=/path/to/your_frames SCENARIO=demo_scenarios/scenario.json bash run_demo.sh
```
