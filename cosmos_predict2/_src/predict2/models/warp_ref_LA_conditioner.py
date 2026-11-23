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

"""Lookahead (LA) frame conditioning for video generation.

This module provides:
1. Video2WorldConditionWarpRefLA: Extended condition dataclass with LA latents
2. encode_la_frame: Utility function for encoding the LA frame via 3D VAE
"""

from dataclasses import dataclass
from typing import Optional

import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.models.warp_ref_conditioner import Video2WorldConditionWarpRef


@dataclass(frozen=True)
class Video2WorldConditionWarpRefLA(Video2WorldConditionWarpRef):
    """Extended Video2WorldConditionWarpRef with Lookahead (LA) frame latents.

    Additional fields:
        la_latents: [B, 1, C, 1, H_latent, W_latent] encoded LA frame latents
        la_time_indices: [B, 1] latent time indices (20-24 range) for RoPE positioning
    """
    la_latents: Optional[torch.Tensor] = None  # [B, 1, C, 1, H, W]
    la_time_indices: Optional[torch.Tensor] = None  # [B, 1]
    la_extrinsics: Optional[torch.Tensor] = None  # [B, 1, 3, 4] w2c
    la_intrinsics: Optional[torch.Tensor] = None  # [B, 1, 3, 3]
    la_mask: Optional[torch.Tensor] = None  # [B, 1] binary mask (1=valid, 0=dropped)


def encode_la_frame(
    tokenizer,
    la_frame: torch.Tensor,
) -> torch.Tensor:
    """Encode a single LA frame via 3D VAE.

    Args:
        tokenizer: Video tokenizer (e.g., Wan2pt1VAEInterface)
        la_frame: Raw LA frame [B, 1, C, H, W] in range [0, 255] uint8

    Returns:
        LA latents tensor [B, 1, latent_ch, 1, H_latent, W_latent]
    """
    B, N, C, H, W = la_frame.shape
    assert N == 1, f"Expected exactly 1 LA frame, got {N}"
    device = la_frame.device

    # Get latent dimensions from tokenizer
    latent_ch = tokenizer.latent_ch  # 16
    H_latent = H // tokenizer.spatial_compression_factor  # H // 8
    W_latent = W // tokenizer.spatial_compression_factor  # W // 8

    # Reshape: [B, 1, C, H, W] -> [B, C, 1, H, W] (add temporal dim)
    la_reshaped = la_frame.squeeze(1).unsqueeze(2)  # [B, C, 1, H, W]

    # Normalize to [-1, 1]
    if la_reshaped.dtype == torch.uint8:
        la_normalized = la_reshaped.float() / 127.5 - 1.0
    else:
        la_normalized = la_reshaped

    # Encode via 3D VAE
    with torch.no_grad():
        la_latents = tokenizer.encode(la_normalized)
        # Shape: [B, latent_ch, 1, H_latent, W_latent]

    # Reshape to [B, 1, latent_ch, 1, H_latent, W_latent]
    la_latents = la_latents.unsqueeze(1)

    log.debug(
        f"Encoded LA frame -> latent shape {la_latents.shape}"
    )

    return la_latents
