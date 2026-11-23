# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026-present NAVER Corp.
# SPDX-License-Identifier: Apache-2.0
# Modified by NAVER Corp. in 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a sharded DCP training checkpoint to a single consolidated PyTorch file.

Training writes checkpoints in distributed (DCP) format — an ``iter_XXXXXXXXX/``
directory whose ``model/`` subfolder holds ``__N_0.distcp`` shards. Inference
(`sh/inference/generate.sh`) expects a single consolidated ``model.pt`` instead.

Usage:
    python scripts/convert_distcp_to_pt.py <iter_dir>/model <iter_dir>

This writes, into the output directory:
    model.pt            full state dict (regular + EMA weights)
    model_ema_fp32.pt   EMA weights only, fp32      (with --ema, default)
    model_ema_bf16.pt   EMA weights only, bf16
"""

import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Path to the DCP shard directory (the 'model/' folder).")
    parser.add_argument("output_dir", type=Path, help="Directory to write the consolidated checkpoint(s) into.")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="Skip exporting EMA-only weights.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = args.output_dir / "model.pt"

    # Consolidate the sharded DCP checkpoint into a single state dict.
    dcp_to_torch_save(str(args.input_dir), str(pt_path))
    print(f"Converted '{args.input_dir}' -> '{pt_path}'")

    if not args.ema:
        return

    # Export EMA weights on their own (net_ema.* renamed to net.* so they load as the
    # regular network). model_ema_bf16.pt is the smallest and is recommended for inference.
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    ema_fp32 = {k.replace("net_ema.", "net."): v for k, v in state_dict.items() if k.startswith("net_ema.")}
    if not ema_fp32:
        raise ValueError("Checkpoint contains no EMA weights (no 'net_ema.*' keys); rerun with --no-ema.")

    ema_fp32_path = args.output_dir / "model_ema_fp32.pt"
    torch.save(ema_fp32, ema_fp32_path)
    print(f"Saved EMA fp32 weights -> '{ema_fp32_path}'")

    ema_bf16 = {
        k: (v.bfloat16() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v)
        for k, v in ema_fp32.items()
    }
    ema_bf16_path = args.output_dir / "model_ema_bf16.pt"
    torch.save(ema_bf16, ema_bf16_path)
    print(f"Saved EMA bf16 weights -> '{ema_bf16_path}'")


if __name__ == "__main__":
    main()
