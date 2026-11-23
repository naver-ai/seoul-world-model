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

"""Modified MinimalV1LVGDiT for warp + reference image conditioning.

This network extends the warp conditioning to also support reference image token concatenation:

1. Warped video: Channel concat (same as MinimalV1LVGDiTWarp)
   - Uses warp-expanded x_embedder (34 channels)
   - Pretrained from unified model (18ch=72dims) → expanded to (34ch=136dims)

2. Reference images: Token concat
   - Uses separate ref_embedder (17 channels = 68 dims)
   - Pretrained from unified model's x_embedder (18ch=72dims) → extracted to (17ch=68dims)
   - condition_mask channel is discarded (ref images don't need frame conditioning)
   - Offset RoPE time embedding to avoid collision with main video
   - Reference tokens are concatenated with main tokens in self-attention
"""

from typing import List, Optional, Tuple

import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit_warp_ref import MiniTrainDITWarpRef


class MinimalV1LVGDiTWarpRef(MiniTrainDITWarpRef):
    """Extended MinimalV1LVGDiT for warp + reference image conditioning.

    This class extends MiniTrainDITWarpRef to add channel concatenation for warped video.

    Pretrained model (Cosmos-Predict2.5-2B unified):
    - 18 channels: 16 latent + 1 condition_mask + 1 padding_mask = 72 dims

    Input channels for main video (x_embedder, channel concat):
    - 16 channels: video latents (noisy/target)
    - 1 channel: condition_video_input_mask (marks conditional frames for frame_replace)
    - 16 channels: warped video latents (dense, from depth-based forward warping)
    - 1 channel: padding_mask (added by parent)
    Total: 34 channels = 136 dims

    Reference images (ref_embedder, token concat):
    - 16 channels: reference frame latents (encoded via 3D VAE)
    - 1 channel: padding_mask
    Total: 17 channels = 68 dims

    Weight initialization:
    - x_embedder: 72 dims → 136 dims (latent + condition_mask preserved, warped zero-init)
    - ref_embedder: 72 dims → 68 dims (latent + padding_mask, condition_mask discarded)
    """

    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        max_ref_frames: int = 5,
        ref_time_offset: int = 50,
        ref_time_stride: int = 5,
        **kwargs
    ):
        """Initialize the network.

        Args:
            timestep_scale: Scale factor for timesteps
            max_ref_frames: Maximum number of reference frames (default: 5)
            ref_time_offset: Time offset for first reference frame RoPE (default: 50)
            ref_time_stride: Time stride between reference frames (default: 5)
                e.g., with offset=50, stride=5: [50, 55, 60, 65, 70]
            **kwargs: Arguments for parent MiniTrainDITWarpRef
        """
        assert "in_channels" in kwargs, "in_channels must be provided"

        # Store original in_channels (16) for reference embedder
        self.original_in_channels = kwargs["in_channels"]  # 16

        # For main video (warp): add 1 for condition_mask + 16 for warped latents
        # padding_mask (1) is added by parent
        # Total: 16 + 1 + 16 = 33 (before padding_mask) -> 34 (after parent adds padding_mask)
        self.num_warped_channels = 16
        kwargs["in_channels"] += (1 + self.num_warped_channels)  # 16 + 17 = 33

        # Set ref_in_channels for parent (original latent channels)
        kwargs["ref_in_channels"] = self.original_in_channels  # 16
        kwargs["max_ref_frames"] = max_ref_frames
        kwargs["ref_time_offset"] = ref_time_offset
        kwargs["ref_time_stride"] = ref_time_stride

        self.timestep_scale = timestep_scale

        super().__init__(*args, **kwargs)

        log.info(
            f"MinimalV1LVGDiTWarpRef initialized: "
            f"x_embedder={33 + 1}ch (warp), ref_embedder={self.original_in_channels + 1}ch, "
            f"max_ref_frames={max_ref_frames}, RoPE: TIME_OFFSET(offset={ref_time_offset}, stride={ref_time_stride})"
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        conditional_latents_B_C_T_H_W: Optional[torch.Tensor] = None,  # Warped video latents
        ref_latents_B_N_C_T_H_W: Optional[torch.Tensor] = None,  # Reference image latents
        ref_mask_B_N: Optional[torch.Tensor] = None,
        ref_time_indices_B_N: Optional[torch.Tensor] = None,  # Reference time indices for RoPE
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        # Camera parameters (passed through to parent)
        main_extrinsics_B_T_3_4: Optional[torch.Tensor] = None,
        main_intrinsics_B_T_3_3: Optional[torch.Tensor] = None,
        ref_extrinsics_B_N_3_4: Optional[torch.Tensor] = None,
        ref_intrinsics_B_N_3_3: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with warped video channel concat + reference image token concat.

        Args:
            x_B_C_T_H_W: Noisy video latents [B, 16, T, H, W]
            timesteps_B_T: Diffusion timesteps [B, T]
            crossattn_emb: Text embeddings
            condition_video_input_mask_B_C_T_H_W: Frame replace mask [B, 1, T, H, W]
            conditional_latents_B_C_T_H_W: Warped video latents [B, 16, T, H, W] (dense, all frames)
            ref_latents_B_N_C_T_H_W: Reference latents [B, max_ref, 16, 1, H, W]
            ref_mask_B_N: Reference mask [B, max_ref]
            ref_time_indices_B_N: Reference time indices [B, max_ref] for RoPE (0-23 range)
            fps: Frame rate
            padding_mask: Padding mask
            data_type: Data type (VIDEO or IMAGE)
            intermediate_feature_ids: IDs for intermediate features
            img_context_emb: Image context embeddings
            **kwargs: Additional arguments (ignored)

        Returns:
            Model output (velocity field prediction)
        """
        del kwargs

        B, C, T, H, W = x_B_C_T_H_W.shape

        # =========================================
        # 1. Channel concat for main video (WARP)
        # =========================================
        if data_type == DataType.VIDEO:
            # Concatenate: video latents + condition_mask + warped latents
            # x_B_C_T_H_W: [B, 16, T, H, W]
            # condition_video_input_mask_B_C_T_H_W: [B, 1, T, H, W]
            # conditional_latents_B_C_T_H_W: [B, 16, T, H, W] (warped)

            if conditional_latents_B_C_T_H_W is not None:
                x_concat = torch.cat([
                    x_B_C_T_H_W,
                    condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W),
                    conditional_latents_B_C_T_H_W.type_as(x_B_C_T_H_W),
                ], dim=1)  # [B, 33, T, H, W]
            else:
                # If warped latents not provided, use zeros
                zero_warped = torch.zeros(
                    B, self.num_warped_channels, T, H, W,
                    device=x_B_C_T_H_W.device,
                    dtype=x_B_C_T_H_W.dtype,
                )
                x_concat = torch.cat([
                    x_B_C_T_H_W,
                    condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W),
                    zero_warped,
                ], dim=1)  # [B, 33, T, H, W]
        else:
            # IMAGE mode: add zeros for mask and warped channels
            zero_mask = torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
            zero_warped = torch.zeros(
                (B, self.num_warped_channels, T, H, W),
                dtype=x_B_C_T_H_W.dtype,
                device=x_B_C_T_H_W.device,
            )
            x_concat = torch.cat([x_B_C_T_H_W, zero_mask, zero_warped], dim=1)

        # =========================================
        # 2. Call parent forward (handles reference token concat)
        # =========================================
        return super().forward(
            x_B_C_T_H_W=x_concat,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
            ref_latents_B_N_C_T_H_W=ref_latents_B_N_C_T_H_W,
            ref_mask_B_N=ref_mask_B_N,
            ref_time_indices_B_N=ref_time_indices_B_N,
            main_extrinsics_B_T_3_4=main_extrinsics_B_T_3_4,
            main_intrinsics_B_T_3_3=main_intrinsics_B_T_3_3,
            ref_extrinsics_B_N_3_4=ref_extrinsics_B_N_3_4,
            ref_intrinsics_B_N_3_3=ref_intrinsics_B_N_3_3,
        )

    # NOTE: ref_embedder initialization is handled by parent class's load_state_dict().
    # This ensures both main model and EMA model get proper ref_embedder weights
    # when loading from pretrained checkpoint (where ref_embedder is missing).
