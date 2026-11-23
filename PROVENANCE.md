# Source provenance

This file records the ownership and license rules applied to the release tree.

| Category | Files | Copyright and license |
| --- | --- | --- |
| NAVER original work | Release documentation and demos; callbacks; local trajectory/dataloader utilities and `datasets/local_datasets/warp/transforms.py` | Copyright (c) 2026-present NAVER Corp.; Apache-2.0 |
| NVIDIA Cosmos-Predict2.5 | Files present in upstream commit `fd18be86376a395fbdc7bf78a8c237937d6d95ff`, plus local model, conditioner, DiT, and inference derivatives | Original NVIDIA copyright; Apache-2.0; locally changed files carry a modification notice |
| NVIDIA GEN3C | `models/gen3c_warping/cache_3d.py`, `camera_utils.py`, and `forward_warp_utils_pytorch.py`; `datasets/local_datasets/warp/projection.py` and `splatting.py` | Original NVIDIA copyright; Apache-2.0; source URL and modification notice retained |
| Data/generated metadata | JSON scenarios, prompt libraries, lockfiles, and generated attribution data | Covered by the applicable project or upstream component notice; no source-code header is inserted into formats that do not support comments |

## Source file headers

Python source files use one of the following headers, chosen by how much of the
file is NAVER work:

- **NAVER original work**: `Seoul World Model`, `Copyright (c) 2026-present NAVER Corp.`,
  and `Apache-2.0`.
- **Substantially extended upstream files** (the warp/reference models and
  conditioners, the `*_warp_ref*` DiT variants, the video2world network config,
  the validation sample callback, `scripts/inference_ar.py`, and
  `scripts/convert_distcp_to_pt.py`): the original NVIDIA copyright, followed by
  `Copyright (c) 2026-present NAVER Corp.`, the Apache-2.0 SPDX identifier, and
  `Modified by NAVER Corp. in 2026.`
- **Lightly modified upstream files** (local fixes and integration changes to
  the trainer, checkpointers, configs and experiment registry, model loader,
  base rectified-flow models, attention, base DiT, and GEN3C warping code): the original NVIDIA copyright, the Apache-2.0 SPDX identifier, and
  `Modified by NAVER Corp. in 2026.` GEN3C files also keep their source URL.
- **Unmodified upstream files**: the original NVIDIA header only.

Both modified-file forms state that the file was changed, as Apache-2.0
section 4(b) requires. The NAVER copyright line is added only where the NAVER
contribution is a substantial part of the file.

The upstream Cosmos baseline was identified by comparing the initial repository
snapshot against the public v1.4.1-era source. The only initial differences were
local configuration/documentation files and lockfile changes.

No restricted-license third-party camera source code is included in the release
tree. Camera conditioning (`networks/camera_net.py`) is an independent NAVER
implementation motivated by the camera-conditioning design of Hunyuan-GameCraft;
it is written against the existing Seoul World Model interface and checkpoint
schema and contains no Hunyuan-GameCraft source code.

Model weights released with this project (the Seoul World Model checkpoint)
are licensed by NAVER Corp. under CC BY-NC 4.0 (non-commercial); see
`LICENSES/CC-BY-NC-4.0.txt`. They are also Derivative Models of NVIDIA
Cosmos-Predict2.5 and remain subject to the NVIDIA Open Model License; see
NOTICE and `LICENSES/NVIDIA-Open-Model-License.txt`. The source code is
Apache-2.0 and is not affected by the checkpoint license.

The video interpolation model used in our research (fine-tuned from NVIDIA
Cosmos3-Nano) is not released, and no Cosmos3-Nano model materials are part of
the release tree.

The depth and camera poses of the training data, and the depth and camera poses
released with the street-view data, were estimated with Depth-Anything-3 (ByteDance
Seed), whose DA3NESTED-GIANT-LARGE / DA3NESTED-GIANT-LARGE-1.1 weights are
licensed under CC BY-NC 4.0. The released depth and camera poses are provided
under the same non-commercial terms; no Depth-Anything-3 code or weights are part of the
release tree. See NOTICE.
