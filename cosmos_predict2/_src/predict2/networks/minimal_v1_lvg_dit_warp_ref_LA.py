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

"""Modified MinimalV1LVGDiT for warp + reference + LA frame conditioning.

This network extends MinimalV1LVGDiTWarpRef to also support LA (Lookahead) frame
token concatenation:

1. Warped video: Channel concat (same as MinimalV1LVGDiTWarpRef)
2. Reference images: Token concat (same as MinimalV1LVGDiTWarpRef)
3. LA frame: Token concat with RoPE at actual temporal position (NEW)
4. Warp mask: Separate mask_embedder + element-wise addition (NEW)
"""

from typing import List, Optional, Tuple

import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import PatchEmbed
from cosmos_predict2._src.predict2.networks.minimal_v4_dit_warp_ref_LA import MiniTrainDITWarpRefLA


class MinimalV1LVGDiTWarpRefLA(MiniTrainDITWarpRefLA):
    """Extended MinimalV1LVGDiT for warp + reference + LA frame conditioning.

    This class extends MiniTrainDITWarpRefLA to add channel concatenation for warped video
    and a separate mask_embedder for warp validity mask.

    Input channels for main video (x_embedder, channel concat):
    - 16 channels: video latents (noisy/target)
    - 1 channel: condition_video_input_mask
    - 16 channels: warped video latents
    - 1 channel: padding_mask (added by parent)
    Total: 34 channels = 136 dims

    Warp mask (mask_embedder, element-wise addition):
    - 1 channel: warp_mask (warped pixel validity)
    Total: 1 channel = 4 dims → projected to hidden_dim, added to main tokens

    Reference images (ref_embedder, token concat):
    - 16 channels: reference frame latents
    - 1 channel: padding_mask
    Total: 17 channels = 68 dims

    LA frame (la_embedder, token concat):
    - 16 channels: LA frame latents
    - 1 channel: padding_mask
    Total: 17 channels = 68 dims
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
        assert "in_channels" in kwargs, "in_channels must be provided"

        # Store original in_channels (16) for reference/LA embedder
        self.original_in_channels = kwargs["in_channels"]  # 16

        # For main video (warp): add 1 for condition_mask + 16 for warped latents
        self.num_warped_channels = 16
        kwargs["in_channels"] += (1 + self.num_warped_channels)  # 16 + 17 = 33

        # Set ref_in_channels for parent (original latent channels)
        kwargs["ref_in_channels"] = self.original_in_channels  # 16
        kwargs["max_ref_frames"] = max_ref_frames
        kwargs["ref_time_offset"] = ref_time_offset
        kwargs["ref_time_stride"] = ref_time_stride

        self.timestep_scale = timestep_scale

        super().__init__(*args, **kwargs)

        # Build mask_embedder: separate PatchEmbed for warp_mask (1 channel)
        # Zero-initialized so it has no effect at the start of training.
        self.mask_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=1,
            out_channels=self.model_channels,
        )
        # Zero-init the linear projection so mask_embedder produces zeros initially
        self.mask_embedder.proj[1].weight.data.zero_()

        # Temporary storage for warp_mask during forward pass
        self._warp_mask = None
        self._mask_embedder_initialized = False

        log.info(
            f"MinimalV1LVGDiTWarpRefLA initialized: "
            f"x_embedder={33 + 1}ch (warp), mask_embedder=1ch (warp_mask, zero-init), "
            f"ref_embedder={self.original_in_channels + 1}ch, "
            f"la_embedder={self.original_in_channels + 1}ch, "
            f"max_ref_frames={max_ref_frames}, RoPE: TIME_OFFSET(offset={ref_time_offset}, stride={ref_time_stride})"
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to handle mask_embedder initialization logging.

        mask_embedder is intentionally zero-initialized (not copied from x_embedder).
        - If missing from checkpoint (fresh start): keep zero-init, log confirmation
        - If present in checkpoint (resume): load normally, log weight stats
        """
        # Find mask_embedder key in state_dict
        mask_key = None
        for key in state_dict.keys():
            if key.endswith("mask_embedder.proj.1.weight"):
                mask_key = key
                break

        # Detect if mask_embedder was loaded from checkpoint
        mask_needs_init = False
        if mask_key is not None:
            result = self._check_embedder_missing_in_checkpoint("mask_embedder")
            if result is not None:
                mask_needs_init = result
            else:
                log.info("[load_state_dict] mask_embedder in state_dict (non-DCP), assuming loaded from checkpoint")
        else:
            mask_needs_init = True
            log.info("[load_state_dict] mask_embedder key not in state_dict, needs init")

        if mask_needs_init:
            # mask_embedder is designed to be zero-initialized — no need to copy from x_embedder
            log.info("[load_state_dict] mask_embedder missing in checkpoint → using zero-init (by design)")
            weight_norm = self.mask_embedder.proj[1].weight.data.norm().item()
            log.info(f"[load_state_dict] mask_embedder weight norm: {weight_norm:.6f} (expected 0.0)")
            self._mask_embedder_initialized = True
        else:
            log.info("[load_state_dict] mask_embedder found in checkpoint, loading normally")
            self._mask_embedder_initialized = True

        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Log mask_embedder weight stats after loading
        try:
            w = self.mask_embedder.proj[1].weight
            if hasattr(w, 'to_local'):
                w = w.to_local()
            else:
                w = w.data
            log.info(
                f"[load_state_dict] mask_embedder weight after load: "
                f"shape={list(w.shape)}, norm={w.norm().item():.6f}, "
                f"min={w.min().item():.6f}, max={w.max().item():.6f}"
            )
        except Exception as e:
            log.warning(f"[load_state_dict] mask_embedder weight stats failed: {e}")

        return result

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Override to add mask_embedder output to main video tokens.

        After x_embedder produces video tokens, we process the warp_mask through
        mask_embedder and add the result element-wise. This preserves spatial
        correspondence between mask values and video tokens.
        """
        x_B_T_H_W_D, rope_emb, extra_pos_emb = super().prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask,
        )

        # Add mask embedding (element-wise addition, same spatial positions)
        if self._warp_mask is not None:
            mask_input = self._warp_mask.to(dtype=x_B_T_H_W_D.dtype)
            mask_tokens = self.mask_embedder(mask_input)  # [B, T, H, W, D]
            x_B_T_H_W_D = x_B_T_H_W_D + mask_tokens

        return x_B_T_H_W_D, rope_emb, extra_pos_emb

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        conditional_latents_B_C_T_H_W: Optional[torch.Tensor] = None,  # Warped video latents
        warp_mask_B_1_T_H_W: Optional[torch.Tensor] = None,  # Warp validity mask
        ref_latents_B_N_C_T_H_W: Optional[torch.Tensor] = None,  # Reference image latents
        ref_mask_B_N: Optional[torch.Tensor] = None,
        ref_time_indices_B_N: Optional[torch.Tensor] = None,  # Reference time indices for RoPE
        la_latents_B_N_C_T_H_W: Optional[torch.Tensor] = None,  # LA frame latents (NEW)
        la_time_indices_B_N: Optional[torch.Tensor] = None,  # LA time indices (NEW)
        # Camera parameters
        main_extrinsics_B_T_3_4: Optional[torch.Tensor] = None,
        main_intrinsics_B_T_3_3: Optional[torch.Tensor] = None,
        ref_extrinsics_B_N_3_4: Optional[torch.Tensor] = None,
        ref_intrinsics_B_N_3_3: Optional[torch.Tensor] = None,
        la_extrinsics_B_N_3_4: Optional[torch.Tensor] = None,
        la_intrinsics_B_N_3_3: Optional[torch.Tensor] = None,
        cam_mask_B: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with warped video channel concat + mask_embedder + LA + reference.

        Args:
            All args from MinimalV1LVGDiTWarpRef, plus:
            warp_mask_B_1_T_H_W: Warp validity mask [B, 1, T, H, W] (processed by mask_embedder)
            la_latents_B_N_C_T_H_W: LA latents [B, 1, 16, 1, H, W]
            la_time_indices_B_N: LA time indices [B, 1] (20~24)

        Returns:
            Model output (velocity field prediction)
        """
        del kwargs

        B, C, T, H, W = x_B_C_T_H_W.shape


        # =========================================
        # 0. Zero out warp conditioning at GT frame positions
        #    GT frames already provide ground truth, so warped video
        #    and warp_mask should have no influence there.
        # =========================================
        if data_type == DataType.VIDEO and condition_video_input_mask_B_C_T_H_W is not None:
            inv_cond_mask = 1.0 - condition_video_input_mask_B_C_T_H_W[:, :1].type_as(x_B_C_T_H_W)
            if conditional_latents_B_C_T_H_W is not None:
                conditional_latents_B_C_T_H_W = conditional_latents_B_C_T_H_W * inv_cond_mask
            if warp_mask_B_1_T_H_W is not None:
                warp_mask_B_1_T_H_W = warp_mask_B_1_T_H_W * inv_cond_mask

        # =========================================
        # 1. Channel concat for main video (WARP only, no warp_mask)
        # =========================================
        if data_type == DataType.VIDEO:
            if conditional_latents_B_C_T_H_W is not None:
                x_concat = torch.cat([
                    x_B_C_T_H_W,
                    condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W),
                    conditional_latents_B_C_T_H_W.type_as(x_B_C_T_H_W),
                ], dim=1)  # [B, 33, T, H, W]
            else:
                zero_warped = torch.zeros(
                    B, self.num_warped_channels, T, H, W,
                    device=x_B_C_T_H_W.device,
                    dtype=x_B_C_T_H_W.dtype,
                )
                x_concat = torch.cat([
                    x_B_C_T_H_W,
                    condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W),
                    zero_warped,
                ], dim=1)
        else:
            zero_mask = torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
            zero_warped = torch.zeros(
                (B, self.num_warped_channels, T, H, W),
                dtype=x_B_C_T_H_W.dtype,
                device=x_B_C_T_H_W.device,
            )
            x_concat = torch.cat([x_B_C_T_H_W, zero_mask, zero_warped], dim=1)

        # =========================================
        # 2. Store warp_mask for prepare_embedded_sequence
        # =========================================
        self._warp_mask = warp_mask_B_1_T_H_W

        # =========================================
        # 3. Call parent forward (handles LA + ref token concat)
        #    prepare_embedded_sequence override adds mask_embedder output
        # =========================================
        result = super().forward(
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
            la_latents_B_N_C_T_H_W=la_latents_B_N_C_T_H_W,
            la_time_indices_B_N=la_time_indices_B_N,
            main_extrinsics_B_T_3_4=main_extrinsics_B_T_3_4,
            main_intrinsics_B_T_3_3=main_intrinsics_B_T_3_3,
            ref_extrinsics_B_N_3_4=ref_extrinsics_B_N_3_4,
            ref_intrinsics_B_N_3_3=ref_intrinsics_B_N_3_3,
            la_extrinsics_B_N_3_4=la_extrinsics_B_N_3_4,
            la_intrinsics_B_N_3_3=la_intrinsics_B_N_3_3,
            cam_mask_B=cam_mask_B,
        )

        # Clear temporary state
        self._warp_mask = None

        return result
