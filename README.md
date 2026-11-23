<div align="center">

  <h1>
    Seoul World Model
  </h1>
    Grounding World Simulation Models in a Real-World Metropolis

  <h3>
    <a href="https://arxiv.org/abs/2603.15583">Paper</a> &nbsp;|&nbsp;
    <a href="https://seoul-world-model.github.io/">Project Page</a>
  </h3>

</div>

Official code for **Seoul World Model (SWM)**, a city-scale autoregressive world model grounded in the real city of Seoul.

<div align="center">
  <a href="https://huggingface.co/naver-ai/seoul-world-model/resolve/main/assets/sample-teaser.mp4"><img src="https://huggingface.co/naver-ai/seoul-world-model/resolve/main/assets/sample-teaser.webp" alt="SWM teaser" width="100%"></a>
</div>

What if a world model could render not an imagined place, but the real city outside your window? SWM fine-tunes a pretrained video world model on 440k Seoul street-view images, real-world driving videos, and synthetic urban data. At generation time, given coordinates, camera actions, and a text prompt, it retrieves nearby street-view images from a reference database and conditions on their geometric and appearance cues—a retrieval-augmented (RAG) recipe that renders faithful videos spanning kilometers of real cityscape.

## Getting started

### Step 1. Installation

```bash
uv python install
uv sync --extra=cu128        # CUDA 12.8
source .venv/bin/activate
```

For environment details and troubleshooting, please refer to the [cosmos repository setup guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md).

All features in this repository have been tested on NVIDIA H100 GPUs.

### Step 2. Download checkpoint & sample data

> The checkpoint and the sample data are provided for **non-commercial use**
> under CC BY-NC 4.0. The checkpoint is a derivative of NVIDIA Cosmos-Predict2.5 and is
> additionally subject to the NVIDIA Open Model License. See [License](#license) below.

| | Hugging Face |
|---|---|
| SWM checkpoint (`model.pt`, bf16, ~4.1 GB) | [naver-ai/seoul-world-model](https://huggingface.co/naver-ai/seoul-world-model) |
| Sample data (`swm_sample/`, ~280 MB) | [naver-ai/seoul-world-model](https://huggingface.co/naver-ai/seoul-world-model/tree/main/swm_sample) |

```bash
# Checkpoint  -> checkpoints/model.pt
huggingface-cli download naver-ai/seoul-world-model model.pt --local-dir checkpoints

# Sample data  -> data/swm_sample/
huggingface-cli download naver-ai/seoul-world-model --include "swm_sample/*" --local-dir data
```

The VAE ([`Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)) and text
encoder ([`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B)) are
public (no gated access or login). Every generation runs NVIDIA
[`Cosmos-Guardrail1`](https://huggingface.co/nvidia/Cosmos-Guardrail1), as Cosmos-Predict2.5
does: the prompt is checked before generation, and the generated video is checked for unsafe
content and has faces pixelated before it is saved (see [INFERENCE.md](INFERENCE.md#guardrails)).
Cosmos-Guardrail1 is gated: accept its license and run `huggingface-cli login` once.
`generate.sh` runs Hugging Face offline by default, so prefix your **first** run with
`HF_HUB_OFFLINE=0` to download them.

### Step 3. Run a demo

We provide a preset scenario that runs on the sample data (`data/swm_sample`). To use your own poses or write new scenarios, see [INFERENCE.md](INFERENCE.md).

```bash
SCENARIO=demo_scenarios/scenario.json bash run_demo.sh
```

Output goes to `outputs/scenario/`:

- `output.mp4` — the generated trajectory video
- `visualization.mp4` — `[forward-warp condition | generated]` side by side

The scenario fixes a prompt and a trajectory:

| Scenario | Trajectory | Prompt |
|---|---|---|
| `scenario.json` | straight → left turn | vehicles ahead on fire |

<a href="https://huggingface.co/naver-ai/seoul-world-model/resolve/main/assets/demo-scenario.mp4"><img src="https://huggingface.co/naver-ai/seoul-world-model/resolve/main/assets/demo-scenario.webp" alt="scenario output" width="100%"></a>

The sample data also holds everything the teaser video at the top of this page
([`assets/sample-teaser.mp4`](https://huggingface.co/naver-ai/seoul-world-model/resolve/main/assets/sample-teaser.mp4)) was generated from. To regenerate it:

```bash
TEASER=1 DATA_DIR=data/swm_sample CKPT=checkpoints/model.pt bash sh/inference/generate.sh
# -> outputs/generate/teaser/{output.mp4, visualization.mp4}
```

The camera path, reference images and prompt come from `data/swm_sample/teaser.json`. GPU
non-determinism makes the result close to, not identical with, the released video.

### Step 4. Run on your own data

Take landscape RGB photos (16:9 or narrower) every few meters along a street. The camera path
is a preset (`TRAJECTORY`) that starts from the first photo by name, so that photo should face
the driving direction, and the photos must cover the path (about 45 m with the defaults).

Get metric depth and camera poses with
[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3). Install it in a separate
environment (it pins `numpy<2`, SWM needs `numpy>=2`), then run once per route. The
`DA3NESTED-GIANT-LARGE-1.1` weights below are licensed for non-commercial use (CC BY-NC 4.0):

```python
import glob, os, numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3

SRC, OUT = "/path/to/photos", "/path/to/my_route"   # your photos -> SWM input folder
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
paths = sorted(p for p in glob.glob(f"{SRC}/*") if p.lower().endswith((".jpg", ".jpeg", ".png")))
pred = model.inference(paths, process_res=1274)   # one call per route: shared world frame and scale
for sub in ("images", "depth", "intrinsics", "extrinsics"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
for i, p in enumerate(paths):
    n = os.path.splitext(os.path.basename(p))[0]
    Image.fromarray(pred.processed_images[i]).save(f"{OUT}/images/{n}.png")
    np.save(f"{OUT}/depth/{n}.npy", pred.depth[i])            # meters
    np.save(f"{OUT}/intrinsics/{n}.npy", pred.intrinsics[i])  # K (3, 3)
    np.save(f"{OUT}/extrinsics/{n}.npy", pred.extrinsics[i])  # world-to-camera (3, 4)
```

Then generate in the SWM environment:

```bash
DATA_DIR=/path/to/my_route CKPT=checkpoints/model.pt TRAJECTORY=straight \
    PROMPT="A car driving down a rainy street at night" bash sh/inference/generate.sh
# -> outputs/generate/straight_start0/{output.mp4, visualization.mp4}
```

See [INFERENCE.md](INFERENCE.md) for all trajectories and options.

## Results

Ann-Arbor-City-Bench (30 sequences built from the public [MARS](https://ai4ce.github.io/MARS/) dataset).
mPSNR and mLPIPS are computed on static regions only (pedestrians and vehicles are masked out with SAM3).

| | FID ↓ | FVD ↓ | Img.Q. ↑ | RotErr ↓ | TransErr ↓ | mPSNR ↑ | mLPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper (Table 1) | 56.61 | 640.17 | 0.66 | 0.055 | **0.154** | 15.18 | 0.481 |
| This checkpoint | **41.58** | **579.46** | **0.69** | **0.055** | 0.160 | **15.82** | **0.442** |

## CARLA Synthetic Dataset

We release the **CARLA synthetic dataset**, one of the datasets used to train our model. Built on the UE4-based CARLA simulator, it provides paired **reference street-view ↔ target video sequence** data. See [SYNTHETIC_DATASET.md](SYNTHETIC_DATASET.md) for details, download instructions, and the data format.

## Citation

```bibtex
@inproceedings{seo2026swm,
  title={{Seoul World Model: Grounding World Simulation Models in a Real-World Metropolis}},
  author={Seo, Junyoung and Choi, Hyunwook and Kwon, Minkyung and Choi, Jinhyeok and Jin, Siyoon and Lee, Gayoung and Kim, Junho and Lee, JoungBin and Gu, Geonmo and Han, Dongyoon and Yun, Sangdoo and Kim, Seungryong and Kim, Jin-Hwa},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

**Code.** This repository is licensed under the [Apache License 2.0](LICENSE).
Original work by NAVER Corp. and the Cosmos-Predict2.5 / GEN3C-derived
components are all Apache-2.0; NVIDIA files retain their original copyright
notices. See [NOTICE](NOTICE) and [source provenance](PROVENANCE.md).

**Model weights.** The SWM checkpoint is released for **non-commercial use** under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(copy in [LICENSES/CC-BY-NC-4.0.txt](LICENSES/CC-BY-NC-4.0.txt)). It is also a
Derivative Model of
[NVIDIA Cosmos-Predict2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B), so its
use must additionally comply with the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
(copy in [LICENSES/NVIDIA-Open-Model-License.txt](LICENSES/NVIDIA-Open-Model-License.txt)).
The code in this repository stays Apache-2.0 regardless of which weights you load.
The text encoder ([Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B))
is under the NVIDIA Open Model License; the VAE
([Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)) is Apache-2.0.
The depth and camera poses of the training data were estimated with
[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3), whose
`DA3NESTED-GIANT-LARGE` weights are licensed under CC BY-NC 4.0; the checkpoint keeps
those non-commercial terms.

**Data.** The sample data and any other street-view data
released for this project contain NAVER proprietary imagery and are provided for
**non-commercial use** under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Depth maps and camera poses released with this street-view data were estimated with
[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(`DA3NESTED-GIANT-LARGE-1.1`, CC BY-NC 4.0) and are likewise for non-commercial use only.
The CARLA synthetic dataset is described, with its own terms, in
[SYNTHETIC_DATASET.md](SYNTHETIC_DATASET.md).

## Acknowledgements

**Built on NVIDIA Cosmos.** This project is built on
[NVIDIA Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5); the
reference-conditioned forward-warp approach is inspired by
[GEN3C](https://github.com/nv-tlabs/GEN3C); the depth and camera poses of the training
data, and the depth and camera poses released with the street-view data, were estimated with
[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance Seed
(model weights under CC BY-NC 4.0).

We sincerely thank everyone at [NAVER](https://www.navercorp.com/en) and
[NAVER Cloud](https://www.navercloudcorp.com/en) who contributed their time,
expertise, and feedback throughout this work. We are grateful to Jinbae Im,
Moonbin Yim, and Bado Lee for their help with data preprocessing, and to
Jongchae Na, Hyunjoon Cho, Hochul Hwang, Myoung-Suk Chae, and Kwangkean Kim for
their support with data processing and their guidance on the use of map data and
metadata. We also thank Jonghak Kim, Jieun Shin, and Hyeeun Shin for valuable
discussions and thoughtful feedback. Their support and collaboration are greatly
appreciated.

The Seoul street-view imagery used to train the model and to build the reference
database comes from the street-view service of [NAVER Map](https://map.naver.com/).
It was collected with
[P1](https://www.navercorp.com/media/pressReleasesDetail?seq=32200), the
vehicle-based panoramic mapping system developed by
[NAVER LABS](https://www.naverlabs.com/en), and is a
proprietary asset of NAVER.

The video interpolation model used in our research, which we do not plan to release, was
fine-tuned from [NVIDIA Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) on the
following driving datasets:

- 도로주행영상 (Road Driving Video), [AI-Hub dataset #180](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=180)
- 교통사고영상 (Traffic Accident Video), [AI-Hub dataset #597](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=597)
- [comma2k19](https://github.com/commaai/comma2k19) by comma.ai ([arXiv:1812.05752](https://arxiv.org/abs/1812.05752))

This research used datasets from The Open AI Dataset Project (AI-Hub, S. Korea).
All data information can be accessed through [AI-Hub](https://www.aihub.or.kr).
이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된
"도로주행영상" 및 "교통사고영상"을 활용하여 수행된 연구입니다. 본 연구에 활용된 데이터는
AI 허브(aihub.or.kr)에서 다운로드할 수 있습니다. No AI-Hub data is redistributed with this
repository.
