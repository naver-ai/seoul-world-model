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

"""Reference image conditioning for video generation.

This module provides:
1. Video2WorldConditionWarpRef: Extended condition dataclass with reference latents
2. Utility functions for encoding reference frames
"""

from dataclasses import dataclass
from typing import Optional

import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition


@dataclass(frozen=True)
class Video2WorldConditionWarpRef(Video2WorldCondition):
    """Extended Video2WorldCondition with reference image latents.

    This condition class extends Video2WorldCondition to support reference images
    that are encoded via 3D VAE and concatenated as tokens.

    Additional fields:
        ref_latents: [B, max_ref, C, 1, H_latent, W_latent] encoded reference frame latents
        ref_mask: [B, max_ref] binary mask indicating valid reference frames
        ref_time_indices: [B, max_ref] latent time indices (0-23) for RoPE positioning
    """
    ref_latents: Optional[torch.Tensor] = None  # [B, max_ref, C, 1, H, W]
    ref_mask: Optional[torch.Tensor] = None  # [B, max_ref] binary mask
    ref_time_indices: Optional[torch.Tensor] = None  # [B, max_ref] time indices for RoPE
    warp_mask: Optional[torch.Tensor] = None  # [B, 1, T_latent, H_latent, W_latent] warped pixel validity
    # Camera parameters for camera embedding (CameraNet)
    main_extrinsics: Optional[torch.Tensor] = None  # [B, T, 3, 4] w2c
    main_intrinsics: Optional[torch.Tensor] = None  # [B, T, 3, 3]
    ref_extrinsics: Optional[torch.Tensor] = None   # [B, N, 3, 4] w2c
    ref_intrinsics: Optional[torch.Tensor] = None    # [B, N, 3, 3]
    cam_mask: Optional[torch.Tensor] = None  # [B] binary mask (1=keep, 0=drop camera)


def encode_reference_frames(
    tokenizer,
    ref_frames: torch.Tensor,
    ref_mask: torch.Tensor,
) -> torch.Tensor:
    """Encode reference frames via 3D VAE.

    Each reference frame is encoded as a single-frame video latent.
    Invalid reference frames (ref_mask=0) produce zero latents.

    Args:
        tokenizer: Video tokenizer (e.g., Wan2pt1VAEInterface)
        ref_frames: Raw reference frames [B, max_ref, C, H, W] in range [0, 255] uint8
        ref_mask: Binary mask [B, max_ref] indicating valid reference frames

    Returns:
        Reference latents tensor [B, max_ref, latent_ch, 1, H_latent, W_latent]
    """
    B, max_ref, C, H, W = ref_frames.shape
    device = ref_frames.device

    # Get latent dimensions from tokenizer
    latent_ch = tokenizer.latent_ch  # Should be 16
    H_latent = H // tokenizer.spatial_compression_factor  # H // 8
    W_latent = W // tokenizer.spatial_compression_factor  # W // 8

    # Initialize output tensor
    ref_latents = torch.zeros(
        B, max_ref, latent_ch, 1, H_latent, W_latent,
        device=device,
        dtype=torch.float32,
    )

    # Reshape for batch encoding: [B * max_ref, C, 1, H, W]
    ref_frames_reshaped = ref_frames.reshape(B * max_ref, C, H, W)
    ref_frames_reshaped = ref_frames_reshaped.unsqueeze(2)  # Add temporal dim

    # Normalize to [-1, 1]
    if ref_frames_reshaped.dtype == torch.uint8:
        ref_frames_normalized = ref_frames_reshaped.float() / 127.5 - 1.0
    else:
        ref_frames_normalized = ref_frames_reshaped

    # Encode all frames at once
    with torch.no_grad():
        try:
            latents_flat = tokenizer.encode(ref_frames_normalized)
            # Shape: [B * max_ref, latent_ch, 1, H_latent, W_latent]

            # Reshape back: [B, max_ref, latent_ch, 1, H_latent, W_latent]
            ref_latents = latents_flat.reshape(B, max_ref, latent_ch, 1, H_latent, W_latent)

            # Zero out invalid reference frames based on mask
            # ref_mask: [B, max_ref] -> [B, max_ref, 1, 1, 1, 1]
            mask_expanded = ref_mask.view(B, max_ref, 1, 1, 1, 1).type_as(ref_latents)
            ref_latents = ref_latents * mask_expanded

        except Exception as e:
            log.error(f"Failed to encode reference frames: {e}")
            # Return zeros on failure
            pass

    log.debug(
        f"Encoded {max_ref} reference frames -> latent shape {ref_latents.shape}"
    )

    return ref_latents
