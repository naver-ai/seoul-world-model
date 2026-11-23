# Seoul World Model — CARLA Driving Dataset

Training data for the **Seoul World Model**: a large-scale synthetic driving dataset captured from the
[CARLA simulator](https://carla.org/), providing **RGB + depth + camera parameters** along
vehicle- and pedestrian-following trajectories. The model learns to generate an urban driving
video — the **target** — conditioned on a set of **reference** frames. The data therefore comes in two
roles:

- 🎬 **Target** — the urban video sequences the model predicts: RGB + depth + camera, captured as
  93- or 200-frame clips along each trajectory. These make up the four **capture datasets** below (and
  the `target_seqs/` in the pairs repo).
- 🖼️ **Reference** — the frames each target is conditioned on. All reference data lives in a **single
  file**, `references.tar`, inside the [**Reference ↔ Target Pairs**](#-reference--target-pairs) repository.

**How they link:** every target frame's `camera.json` carries a `matched_references` list — the IDs of the
reference frames relevant to that frame, accumulated along the trajectory. Those IDs index into
`references.tar`, so a `(reference, target)` training pair can be reconstructed for any frame.

All repositories are stored in [WebDataset](https://github.com/webdataset/webdataset) (`.tar`) format for
efficient streaming.

| Role | Where it lives | Covered below |
|------|----------------|---------------|
| 🎬 Target | `carla-dataset`, `carla-dataset-ped`, `carla-dataset-ped2`, `carla-dataset-ped3` + `target_seqs/` | [Target Capture Datasets](#-target-capture-datasets) |
| 🖼️ Reference | `references.tar` (single file, in the pairs repo) | [Reference ↔ Target Pairs](#-reference--target-pairs) |

---

## 📦 Target Capture Datasets

The four capture datasets share the same per-frame schema (RGB / depth / camera / metadata) and the
same `Town{01-06}/{actor_type}/carla-stage2-XXXXXX.tar` layout. They differ mainly in **actor focus**,
**scene length**, and **scale**.

| # | Dataset | Focus | Scenes | Images | Frames / scene |
|---|---------|-------|-------:|-------:|---------------:|
| 1 | [`mkxdxd/carla-dataset`](https://huggingface.co/datasets/mkxdxd/carla-dataset) | Vehicle **and** pedestrian | 6,049 | 562,557 | 93 |
| 2 | [`mkxdxd/carla-dataset-ped`](https://huggingface.co/datasets/mkxdxd/carla-dataset-ped) | Pedestrian only | 2,908 | 581,600 | 200 |
| 3 | [`mkxdxd/carla-dataset-ped2`](https://huggingface.co/datasets/mkxdxd/carla-dataset-ped2) | Pedestrian only (in Town 05 only) | 2,720 | 544,000 | 200 |
| 4 | [`kaistcvlab/carla-dataset-ped3`](https://huggingface.co/datasets/kaistcvlab/carla-dataset-ped3) | Pedestrian only (in Town 05 only) | 1,598 | 319,600 | 200 |

**Combined:** 13,275 scenes and **2.01M image frames** across Town01–Town06.

### Totals by actor type

| Actor | Scenes | Images |
|-------|-------:|-------:|
| Pedestrian | 10,255 | 1,726,897 |
| Vehicle | 3,020 | 280,860 |
| **Total** | **13,275** | **2,007,757** |

> Pedestrian = `carla-dataset` (ped split) + `carla-dataset-ped` + `carla-dataset-ped2` + `carla-dataset-ped3`;
> vehicle = `carla-dataset` only.

---

## 🎯 Reference ↔ Target Pairs

[`enrue1893/Synthetic-Ref-to-Urban-Video-Pairs-Town1-6`](https://huggingface.co/datasets/enrue1893/Synthetic-Ref-to-Urban-Video-Pairs-Town1-6)
holds the **reference data** for the Seoul World Model. The 🖼️ **reference pool is the single
`references.tar`** — this is the *only* reference data anywhere; the `camera.matched_references` IDs in the
[target capture datasets](#-target-capture-datasets) all index into it. The repo also bundles 🎬 **target**
sequences under `target_seqs/` for convenience (same role as the capture datasets), so a complete
`(reference, target)` pair can be assembled from one repo. WebDataset format, **2.79 TB** total
(1.68M rows), covering **Town01–Town06**, both pedestrian and vehicle actors.

```
Synthetic-Ref-to-Urban-Video-Pairs-Town1-6/
├── references.tar                                  # 151.7 GB — shared reference-frame pool
│                                                   #   (e.g. references/Town02/subset_1/reference/depth/...)
└── target_seqs/
    ├── target_seqs_Town01_pedestrian.tar           # 218 GB
    ├── target_seqs_Town01_vehicle.tar              # 219 GB
    ├── ...                                          # one tar per Town × actor
    └── target_seqs_Town06_vehicle.tar              # 219 GB   (12 tars, 2.64 TB total)
```

| Component | Files | Size |
|-----------|-------|------|
| `references.tar` | 1 shared reference pool | 151.7 GB |
| `target_seqs/` | 12 tars — Town01–06 × {pedestrian, vehicle} | 2.64 TB |
| **Total** | | **2.79 TB** |

---

## 🧩 Composition

### Which dataset to use?

- **`carla-dataset`** — the core / "target" set. Both **vehicle-following** and **pedestrian-following**
  trajectories, 6 towns, short **93-frame** clips.
- **`carla-dataset-ped`** — pedestrian-centric, with longer **200-frame** sequences for richer
  temporal coverage of pedestrian motion.
- **`carla-dataset-ped2`** — a large **Town05 pedestrian** extension (2,720 clips × 200 frames ≈ 544,000 images,
  2.53 TB across `Town05/pedestrian` + `pedestrian2/3/4`). Useful when you need much more pedestrian data
  in a single map. *(Note: the HF dataset-viewer preview may error on this repo due to a Parquet schema
  field — streaming via `webdataset` is unaffected.)*
- **`carla-dataset-ped3`** — a further **Town05 pedestrian** extension (1,598 clips × 200 frames = 319,600 images,
  1.49 TB), organized under `Town05/pedestrian5` and `Town05/pedestrian6`. Same viewer caveat as ped2.

### Per-frame contents (all datasets)

Each frame in a shard is a WebDataset sample keyed `{scene_id}_{frame_idx:03d}`:

```
{scene_id}_{frame_idx:03d}.rgb.png        ← RGB image       (1280 × 704)
{scene_id}_{frame_idx:03d}.depth.npy      ← Depth map       (numpy, 704 × 1280)
{scene_id}_{frame_idx:03d}.camera.json    ← Camera params + reference frames
{scene_id}_{frame_idx:03d}.metadata.json  ← Scene metadata
```

| Field | Type | Description |
|-------|------|-------------|
| `rgb` | `PIL.Image` `(1280×704)` | RGB image |
| `depth` | `np.ndarray` `(704, 1280)` | Per-pixel depth map |
| `camera.intrinsic` | matrix | Camera intrinsic matrix |
| `camera.extrinsic` | matrix | Camera extrinsic matrix |
| `camera.carla_transform` | dict | Raw CARLA transform |
| `camera.matched_references` | list | Reference frame IDs accumulated along the trajectory |
| `metadata` | dict | `scene_id`, `frame_id`, `town`, `actor_type` |

### Repository layout

```
<repo>/
├── Town01/
│   ├── vehicle/                       # (carla-dataset only)
│   │   ├── carla-stage2-000000.tar
│   │   └── ...
│   └── pedestrian/
│       ├── carla-stage2-000000.tar
│       └── ...
├── Town02/ ...
└── Town06/ ...
```

- `carla-dataset`: 3 scenes per `.tar` (93 frames each → 279 frames/shard).
- `carla-dataset-ped`: 2 scenes per `.tar` (200 frames each → 400 frames/shard).

---

## 🚀 Quickstart

```bash
pip install webdataset datasets torch pillow numpy huggingface_hub
```

### Stream directly from Hugging Face

```python
import io, json
import numpy as np
import webdataset as wds

# Pick a repo + town + actor type, and a shard range:
url = (
    "https://huggingface.co/datasets/mkxdxd/carla-dataset/resolve/main/"
    "Town01/pedestrian/{carla-stage2-000000..carla-stage2-000010}.tar"
)

dataset = wds.WebDataset(url).decode("pil")

for sample in dataset:
    rgb      = sample["rgb.png"]                              # PIL.Image (1280×704)
    depth    = np.load(io.BytesIO(sample["depth.npy"]))      # np.ndarray (704×1280)
    camera   = json.loads(sample["camera.json"])             # intrinsic / extrinsic / refs
    metadata = json.loads(sample["metadata.json"])           # scene_id, frame_id, town, ...
    print(sample["__key__"], rgb.size, depth.shape, metadata["town"])
    break
```

### With a PyTorch DataLoader

```python
import io, json
import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader

def decode(sample):
    return {
        "rgb":      sample["rgb.png"],
        "depth":    np.load(io.BytesIO(sample["depth.npy"])),
        "camera":   json.loads(sample["camera.json"]),
        "metadata": json.loads(sample["metadata.json"]),
    }

url = ("https://huggingface.co/datasets/mkxdxd/carla-dataset/resolve/main/"
       "Town01/pedestrian/{carla-stage2-000000..carla-stage2-000010}.tar")

ds = wds.WebDataset(url).decode("pil").map(decode)
loader = DataLoader(ds, batch_size=8, num_workers=4)
```

All datasets are **public** — no authentication or token is required to stream them.

---

## 🛠️ How it was generated

Captured from the **CARLA simulator** (Town01–Town06). A camera + depth sensor rig follows either a
**vehicle** or a **pedestrian** actor along its trajectory, recording synchronized RGB and depth at
each frame together with full intrinsic/extrinsic camera parameters. Reference frames are matched and
accumulated along each trajectory to support reference-conditioned view synthesis.

## 📄 License

All datasets listed in this document are released under **CC-BY-4.0**. They were
rendered with the [CARLA simulator](https://carla.org/): CARLA code is MIT-licensed, and
CARLA-specific assets are © CARLA Team under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## 🔗 Links

- Vehicle + pedestrian: <https://huggingface.co/datasets/mkxdxd/carla-dataset>
- Pedestrian (200-frame): <https://huggingface.co/datasets/mkxdxd/carla-dataset-ped>
- Town05 pedestrian (extension): <https://huggingface.co/datasets/mkxdxd/carla-dataset-ped2>
- Town05 pedestrian (extension 2): <https://huggingface.co/datasets/kaistcvlab/carla-dataset-ped3>
- Reference ↔ target pairs (Town1–6): <https://huggingface.co/datasets/enrue1893/Synthetic-Ref-to-Urban-Video-Pairs-Town1-6>
