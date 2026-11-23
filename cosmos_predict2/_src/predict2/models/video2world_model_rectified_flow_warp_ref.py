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

"""Extended Video2World model with warp + reference image conditioning.

This model supports:
1. Warped video conditioning (channel concat) - from warp model
2. Reference image conditioning (token concat) - new functionality

Reference images are:
- Encoded via 3D VAE
- Token-concatenated with offset RoPE embeddings
"""

from dataclasses import replace
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import Text2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import Video2WorldModelRectifiedFlow
from cosmos_predict2._src.predict2.models.warp_ref_conditioner import (
    Video2WorldConditionWarpRef,
    encode_reference_frames,
)


class Video2WorldModelRectifiedFlowWarpRef(Video2WorldModelRectifiedFlow):
    """Extended Video2World model with warp + reference image conditioning.

    This model supports:
    1. Warped video latents (channel concat) - dense conditioning from depth-based forward warping
    2. Reference images (token concat) - clean GT frames for appearance guidance

    Key features:
    - Warped video: Encoded via 3D VAE, channel-concatenated with noisy latents
    - Reference images: Encoded via 3D VAE, token-concatenated with offset RoPE
    """

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldConditionWarpRef]:
        """Get data and condition, encoding both warped video and reference frames.

        Args:
            data_batch: Batch of data containing:
                - video: [B, C, T, H, W] original video in uint8 [0, 255]
                - warped_video: [B, C, T, H, W] warped video in uint8 [0, 255]
                - ref_frames: [B, max_ref, C, H, W] reference frames in uint8 [0, 255]
                - ref_mask: [B, max_ref] binary mask for valid reference frames

        Returns:
            Tuple of (raw_state, latent_state, condition) where condition includes
            conditional_latents (warped) and ref_latents (reference)
        """
        # Get base data and condition from parent
        raw_state, latent_state, base_condition = super().get_data_and_condition(data_batch)

        # Convert base_condition to Video2WorldConditionWarpRef
        base_kwargs = base_condition.to_dict(skip_underscore=False)
        condition = Video2WorldConditionWarpRef(**base_kwargs)

        # =========================================
        # 1. Check disable_warp ablation flag
        # =========================================
        if self.config.disable_warp:
            log.info("disable_warp=True: using zero conditional_latents and warp_mask (no warp ablation)")
            B, C, T, H, W = latent_state.shape
            zero_latents = torch.zeros_like(latent_state)
            condition = replace(condition, conditional_latents=zero_latents)
        else:
            # =========================================
            # 1b. GPU warping if ingredients present (GEN3C GPU WARP)
            # =========================================
            if "warped_video" not in data_batch and "src_frames" in data_batch:
                warped_video = self._gpu_warp_from_ingredients(data_batch, latent_state.device)
                data_batch["warped_video"] = warped_video

            # =========================================
            # 2. Encode warped video (WARP SPECIFIC)
            # =========================================
            if "warped_video" in data_batch:
                warped_video = data_batch["warped_video"]  # [B, C, T, H, W] uint8 or float

                # Move to same device/dtype as latent_state
                warped_video = warped_video.to(device=latent_state.device)

                # Normalize warped video to [-1, 1]
                if warped_video.dtype == torch.uint8:
                    warped_video_normalized = warped_video.float() / 127.5 - 1.0
                else:
                    # GPU warp output is already float in [-1, 1] or [0, 1]
                    warped_video_normalized = warped_video.float()
                    # Clamp to [-1, 1] for safety (gen3c forward_warp outputs [-1, 1])
                    warped_video_normalized = warped_video_normalized.clamp(-1.0, 1.0)

                # Encode warped video using the same tokenizer
                with torch.no_grad():
                    warped_latents = self.tokenizer.encode(warped_video_normalized)

                log.debug(f"Encoded warped video: input={warped_video.shape}, latents={warped_latents.shape}")

                # Store warped latents as conditional_latents
                condition = replace(condition, conditional_latents=warped_latents)
            else:
                log.warning("No 'warped_video' found in data_batch, using zeros for conditional_latents")
                B, C, T, H, W = latent_state.shape
                zero_latents = torch.zeros_like(latent_state)
                condition = replace(condition, conditional_latents=zero_latents)

        # =========================================
        # 2. Encode reference frames (REF SPECIFIC)
        # =========================================
        if "ref_frames" in data_batch and "ref_mask" in data_batch:
            ref_frames = data_batch["ref_frames"]  # [B, max_ref, C, H, W] uint8
            ref_mask = data_batch["ref_mask"]  # [B, max_ref] float

            # Move to same device as latent_state
            ref_frames = ref_frames.to(device=latent_state.device)
            ref_mask = ref_mask.to(device=latent_state.device)

            # Encode reference frames via 3D VAE
            ref_latents = encode_reference_frames(
                self.tokenizer,
                ref_frames,
                ref_mask,
            )

            log.debug(f"Encoded reference frames: input={ref_frames.shape}, latents={ref_latents.shape}, mask={ref_mask.shape}")

            # Get ref_time_indices for dynamic RoPE positioning
            ref_time_indices = None
            if "ref_time_indices" in data_batch:
                ref_time_indices = data_batch["ref_time_indices"].to(device=latent_state.device)
                log.debug(f"Using ref_time_indices: {ref_time_indices.shape}, values: {ref_time_indices[0]}")

            # Update condition with reference latents and time indices
            condition = replace(
                condition,
                ref_latents=ref_latents,
                ref_mask=ref_mask,
                ref_time_indices=ref_time_indices,
            )
        else:
            log.debug("No reference frames in data_batch")

        # =========================================
        # 3. Extract camera parameters (CAMERA SPECIFIC)
        # =========================================
        if "main_extrinsics" in data_batch:
            condition = replace(
                condition,
                main_extrinsics=data_batch["main_extrinsics"].to(device=latent_state.device),
                main_intrinsics=data_batch["main_intrinsics"].to(device=latent_state.device),
            )
            log.debug(f"Extracted main camera params: extrinsics={data_batch['main_extrinsics'].shape}")

        if "ref_extrinsics" in data_batch:
            condition = replace(
                condition,
                ref_extrinsics=data_batch["ref_extrinsics"].to(device=latent_state.device),
                ref_intrinsics=data_batch["ref_intrinsics"].to(device=latent_state.device),
            )
            log.debug(f"Extracted ref camera params: extrinsics={data_batch['ref_extrinsics'].shape}")

        if "cam_mask" in data_batch:
            condition = replace(
                condition,
                cam_mask=data_batch["cam_mask"].to(device=latent_state.device),
            )

        # =========================================
        # 4. Transfer warp_mask to GPU (already at latent resolution from dataset)
        # =========================================
        if self.config.disable_warp:
            # Zero warp_mask: indicate no valid warp pixels
            if "warp_mask" in data_batch:
                condition = replace(condition, warp_mask=torch.zeros_like(data_batch["warp_mask"].to(device=latent_state.device)))
        elif "warp_mask" in data_batch:
            condition = replace(condition, warp_mask=data_batch["warp_mask"].to(device=latent_state.device))

        return raw_state, latent_state, condition

    @torch.no_grad()
    def _gpu_warp_from_ingredients(
        self,
        data_batch: dict[str, torch.Tensor],
        device: torch.device,
        chunk_size: int = 154,
    ) -> torch.Tensor:
        """Perform GPU-side forward warping via direct forward_warp calls.

        Bypasses Cache4D to avoid CPU↔GPU transfers. All computation stays on GPU.
        Processes in chunks to manage GPU memory.

        Args:
            data_batch: Batch containing:
                - src_frames:     [B, T, 3, H, W] float [0,1]
                - src_depths:     [B, T, 1, H, W] float
                - src_w2cs:       [B, T, 4, 4] float
                - src_intrinsics: [B, T, 3, 3] float
                - target_w2cs:    [B, T, 4, 4] float
                - mask_b1hw:      [B, 1, H, W] float
            device: Target device for computation
            chunk_size: Number of frames per GPU chunk (default: 16)

        Returns:
            warped_video: [B, C, T, H, W] float
        """
        from cosmos_predict2._src.predict2.models.gen3c_warping.forward_warp_utils_pytorch import forward_warp

        src_frames = data_batch["src_frames"].to(device).float()       # [B, T, 3, H, W]
        src_depths = data_batch["src_depths"].to(device).float()       # [B, T, 1, H, W]
        mask_b1hw = data_batch["mask_b1hw"].to(device).float()         # [B, 1, H, W]
        src_w2cs = data_batch["src_w2cs"].to(device).float()           # [B, T, 4, 4]
        src_intrinsics = data_batch["src_intrinsics"].to(device).float()  # [B, T, 3, 3]
        target_w2cs = data_batch["target_w2cs"].to(device).float()     # [B, T, 4, 4]

        B, T, C, H, W = src_frames.shape

        # Flatten B*T into batch dim for forward_warp
        frames_flat = src_frames.reshape(B * T, C, H, W)           # [B*T, 3, H, W]
        depths_flat = src_depths.reshape(B * T, 1, H, W)           # [B*T, 1, H, W]
        src_w2cs_flat = src_w2cs.reshape(B * T, 4, 4)              # [B*T, 4, 4]
        tgt_w2cs_flat = target_w2cs.reshape(B * T, 4, 4)           # [B*T, 4, 4]
        intrinsics_flat = src_intrinsics.reshape(B * T, 3, 3)      # [B*T, 3, 3]
        # Expand mask: [B, 1, H, W] -> [B*T, 1, H, W]
        mask_flat = mask_b1hw.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, 1, H, W)

        # Process in GPU-only chunks
        rendered_chunks = []
        mask_chunks = []
        # fp32 like training (warps were made in the CPU dataloader): inference runs under a bf16
        # autocast and with TF32 matmuls on, either of which would quantize the projected pixel
        # coordinates (bmm/matmul). Both are switched off for the warp only, then restored.
        _tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.autocast("cuda", enabled=False):
                for i in range(0, B * T, chunk_size):
                    j = min(i + chunk_size, B * T)
                    warped_chunk, mask_chunk, _, _ = forward_warp(
                        frame1=frames_flat[i:j],
                        mask1=mask_flat[i:j],
                        depth1=depths_flat[i:j],
                        transformation1=src_w2cs_flat[i:j],
                        transformation2=tgt_w2cs_flat[i:j],
                        intrinsic1=intrinsics_flat[i:j],
                        intrinsic2=intrinsics_flat[i:j],
                        is_image=True,
                        is_depth=True,
                    )
                    rendered_chunks.append(warped_chunk)
                    mask_chunks.append(mask_chunk)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = _tf32

        # Concat and reshape: [B*T, 3, H, W] -> [B, C, T, H, W]
        warped_flat = torch.cat(rendered_chunks, dim=0)             # [B*T, 3, H, W]
        warped_video = warped_flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        # src_frames must arrive in [-1, 1]; forward_warp(is_image=True) then outputs
        # content in [-1, 1] with holes filled at -1 (black) — already the convention the
        # tokenizer expects. (Feeding [0,1] would leave content in [0,1] with holes at -1,
        # which is wrong; the dataset emits [-1, 1].)

        # Build the warp validity mask at latent resolution if the dataset did not
        # provide one (forward-warp mode): 8x8 spatial mean-pool + temporal ::4.
        if "warp_mask" not in data_batch:
            mask_flat2 = torch.cat(mask_chunks, dim=0).float()          # [B*T, 1, H, W]
            spatial_ds = 8
            temporal_ds = 4
            mask_bt = mask_flat2.reshape(B, T, 1, H, W)
            mask_lat = mask_bt.reshape(
                B, T, 1, H // spatial_ds, spatial_ds, W // spatial_ds, spatial_ds
            ).mean(dim=(4, 6))                                          # [B, T, 1, H/8, W/8]
            mask_lat = mask_lat.permute(0, 2, 1, 3, 4)                  # [B, 1, T, H/8, W/8]
            mask_lat = mask_lat[:, :, ::temporal_ds, :, :].contiguous()  # [B, 1, T_lat, H/8, W/8]
            data_batch["warp_mask"] = mask_lat

        msg = (
            f"[FORWARD-WARP] GPU warping complete: {B} samples, {T} frames, "
            f"output shape={list(warped_video.shape)}, "
            f"range=[{warped_video.min().item():.3f}, {warped_video.max().item():.3f}]"
        )
        log.info(msg)
        print(msg, flush=True)  # always visible (log.info may be suppressed during generation)

        return warped_video

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        uncond_ref_dropout: bool = False,
        no_ref_images: bool = False,
        ref_guidance: float = 0.0,
    ) -> Callable:
        """Override to preserve both warped and reference latents in condition.

        Handles:
        - gt_frames via set_video_condition()
        - conditional_latents (warped video) via get_data_and_condition()
        - ref_latents (reference images) via get_data_and_condition()

        Args:
            uncond_ref_dropout: If True, use zero ref_latents and ref_mask for uncondition.
                               This enables CFG to treat reference images as conditional input.
                               Ignored when ref_guidance > 0 (decomposed mode takes over).
            no_ref_images: If True, use zero ref_latents for BOTH condition and uncondition.
                          This completely disables reference image conditioning.
            ref_guidance: If > 0, use decomposed CFG with separate ref guidance scale.
                         Formula: uncond_text + cfg_scale*(cond - uncond_text) + ref_scale*(cond - uncond_ref)
                         Where uncond_text = ref kept + text dropped, uncond_ref = ref dropped + text kept.
                         This requires 3 forward passes per step instead of 2.
        """
        from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import NUM_CONDITIONAL_FRAMES_KEY

        # Get num_conditional_frames
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        # Get condition/uncondition from conditioner
        if is_negative_prompt:
            base_condition, base_uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            base_condition, base_uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # Convert to Video2WorldConditionWarpRef
        condition = Video2WorldConditionWarpRef(**base_condition.to_dict(skip_underscore=False))
        uncondition = Video2WorldConditionWarpRef(**base_uncondition.to_dict(skip_underscore=False))

        # Update data type
        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        # Get x0 and full condition with warped + reference latents
        _, x0, condition_full = self.get_data_and_condition(data_batch)

        # Set video condition (gt_frames, mask)
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        # Add warped conditional_latents (WARP SPECIFIC)
        if hasattr(condition_full, 'conditional_latents') and condition_full.conditional_latents is not None:
            condition = replace(condition, conditional_latents=condition_full.conditional_latents)
            uncondition = replace(uncondition, conditional_latents=condition_full.conditional_latents)
            log.debug(f"Added warped conditional_latents: shape={condition_full.conditional_latents.shape}")
        else:
            raise ValueError('Warped conditional latents do not exist. Ensure warped_video is in data_batch.')

        # Add reference latents (REF SPECIFIC)
        if condition_full.ref_latents is not None and condition_full.ref_mask is not None:
            # Zero out reference latents and mask (for no_ref_images / uncond_ref_dropout modes)
            zero_ref_latents = torch.zeros_like(condition_full.ref_latents)
            zero_ref_mask = torch.zeros_like(condition_full.ref_mask)
            # Get ref_time_indices for RoPE positioning
            ref_time_indices = condition_full.ref_time_indices

            if no_ref_images:
                # Completely disable reference images: both cond and uncond use zero ref_latents
                condition = replace(
                    condition,
                    ref_latents=zero_ref_latents,
                    ref_mask=zero_ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                uncondition = replace(
                    uncondition,
                    ref_latents=zero_ref_latents,
                    ref_mask=zero_ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                log.info(f"No ref images mode: both condition and uncondition use zero ref_latents")
            elif ref_guidance > 0:
                # Decomposed CFG mode: separate text guidance and ref guidance
                # velocity = [ref,null] + cfg_scale*([null,text] - [null,null]) + ref_scale*([ref,text] - [null,text])

                # condition: [ref, text] - full conditioning
                condition = replace(
                    condition,
                    ref_latents=condition_full.ref_latents,
                    ref_mask=condition_full.ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                # uncondition: [ref, null] - text dropped, ref kept (baseline)
                uncondition = replace(
                    uncondition,
                    ref_latents=condition_full.ref_latents,
                    ref_mask=condition_full.ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                # uncond_ref: [null, text] - ref dropped, text kept
                uncond_ref = replace(
                    condition,  # start from condition (which has text kept)
                    ref_latents=zero_ref_latents,
                    ref_mask=zero_ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                # uncond_null: [null, null] - both dropped (fully unconditional)
                uncond_null = replace(
                    uncondition,  # start from uncondition (which has text dropped)
                    ref_latents=zero_ref_latents,
                    ref_mask=zero_ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                log.info(f"Decomposed CFG: cfg_scale={guidance}, ref_scale={ref_guidance}")
                log.info(f"  Formula: [ref,null] + cfg*([null,text]-[null,null]) + ref*([ref,text]-[null,text])")
                log.info(f"  condition: [ref, text], uncondition: [ref, null]")
                log.info(f"  uncond_ref: [null, text], uncond_null: [null, null]")
            else:
                condition = replace(
                    condition,
                    ref_latents=condition_full.ref_latents,
                    ref_mask=condition_full.ref_mask,
                    ref_time_indices=ref_time_indices,
                )

                # For uncondition: optionally use zero ref_latents (CFG on reference images)
                if uncond_ref_dropout:
                    uncondition = replace(
                        uncondition,
                        ref_latents=zero_ref_latents,
                        ref_mask=zero_ref_mask,
                        ref_time_indices=ref_time_indices,
                    )
                    log.info(f"CFG ref dropout enabled: uncondition uses zero ref_latents")
                else:
                    uncondition = replace(
                        uncondition,
                        ref_latents=condition_full.ref_latents,
                        ref_mask=condition_full.ref_mask,
                        ref_time_indices=ref_time_indices,
                    )
            log.debug(f"Added reference latents: shape={condition_full.ref_latents.shape}")
        else:
            raise ValueError('Reference latents do not exist. Ensure ref_frames and ref_mask are in data_batch.')

        # Add warp_mask to condition/uncondition (always same, no dropout)
        if condition_full.warp_mask is not None:
            condition = replace(condition, warp_mask=condition_full.warp_mask)
            uncondition = replace(uncondition, warp_mask=condition_full.warp_mask)
            if ref_guidance > 0 and not no_ref_images:
                uncond_ref = replace(uncond_ref, warp_mask=condition_full.warp_mask)
                uncond_null = replace(uncond_null, warp_mask=condition_full.warp_mask)

        # Add camera params (including cam_mask)
        cam_fields = {}
        for field in ['main_extrinsics', 'main_intrinsics', 'ref_extrinsics', 'ref_intrinsics', 'cam_mask']:
            val = getattr(condition_full, field, None)
            if val is not None:
                cam_fields[field] = val

        if cam_fields:
            condition = replace(condition, **cam_fields)
            uncondition = replace(uncondition, **cam_fields)
            if ref_guidance > 0 and not no_ref_images:
                uncond_ref = replace(uncond_ref, **cam_fields)
                uncond_null = replace(uncond_null, **cam_fields)

        # Broadcast split for model parallelism
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)
        if ref_guidance > 0 and not no_ref_images:
            _, uncond_ref, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncond_ref, None, None)
            _, uncond_null, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncond_null, None, None)

        # Return velocity function
        if ref_guidance > 0 and not no_ref_images:
            # Decomposed CFG: 4 forward passes
            # velocity = [ref,null] + cfg_scale*([null,text] - [null,null]) + ref_scale*([ref,text] - [null,text])
            def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                cond_v = self.denoise(noise, noise_x, timestep, condition)           # [ref, text]
                uncond_v = self.denoise(noise, noise_x, timestep, uncondition)       # [ref, null]
                uncond_ref_v = self.denoise(noise, noise_x, timestep, uncond_ref)    # [null, text]
                uncond_null_v = self.denoise(noise, noise_x, timestep, uncond_null)  # [null, null]
                velocity_pred = uncond_v + guidance * (uncond_ref_v - uncond_null_v) + ref_guidance * (cond_v - uncond_ref_v)
                return velocity_pred
        else:
            # Standard CFG: 2 forward passes
            def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                cond_v = self.denoise(noise, noise_x, timestep, condition)
                uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
                velocity_pred = uncond_v + guidance * (cond_v - uncond_v)
                return velocity_pred

        return velocity_fn

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> DenoisePrediction:
        """Denoise with warped video (channel concat) and reference image (token concat) conditioning.

        Args:
            noise: Noise tensor
            xt_B_C_T_H_W: Noisy latents [B, C, T, H, W]
            timesteps_B_T: Diffusion timesteps [B, T]
            condition: Condition object with conditional_latents (warped) and ref_latents (reference)

        Returns:
            Velocity prediction
        """
        # Apply frame_replace conditioning
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Frame replace: first few frames are ground truth
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            # Handle conditional frame timestep
            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )

        # Extract conditional latents (warped video - WARP SPECIFIC)
        conditional_latents = condition.conditional_latents

        # Extract reference latents (REF SPECIFIC)
        ref_latents = getattr(condition, 'ref_latents', None)
        ref_mask = getattr(condition, 'ref_mask', None)

        # Forward pass through network
        condition_dict = condition.to_dict()

        # Add warped conditional latents (channel concat in network)
        if conditional_latents is not None:
            condition_dict['conditional_latents_B_C_T_H_W'] = conditional_latents

        # Add reference latents (token concat in network)
        if ref_latents is not None:
            condition_dict['ref_latents_B_N_C_T_H_W'] = ref_latents
            condition_dict['ref_mask_B_N'] = ref_mask
            # Add ref_time_indices for dynamic RoPE positioning
            ref_time_indices = getattr(condition, 'ref_time_indices', None)
            if ref_time_indices is not None:
                condition_dict['ref_time_indices_B_N'] = ref_time_indices

        # Add warp mask (WARP MASK SPECIFIC)
        warp_mask = getattr(condition, 'warp_mask', None)
        if warp_mask is not None:
            condition_dict['warp_mask_B_1_T_H_W'] = warp_mask

        # Add camera parameters (CAMERA SPECIFIC)
        main_extrinsics = getattr(condition, 'main_extrinsics', None)
        if main_extrinsics is not None:
            condition_dict['main_extrinsics_B_T_3_4'] = main_extrinsics
            condition_dict['main_intrinsics_B_T_3_3'] = getattr(condition, 'main_intrinsics', None)
        ref_extrinsics = getattr(condition, 'ref_extrinsics', None)
        if ref_extrinsics is not None:
            condition_dict['ref_extrinsics_B_N_3_4'] = ref_extrinsics
            condition_dict['ref_intrinsics_B_N_3_3'] = getattr(condition, 'ref_intrinsics', None)
        cam_mask = getattr(condition, 'cam_mask', None)
        if cam_mask is not None:
            condition_dict['cam_mask_B'] = cam_mask

        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            **condition_dict,
        ).float()

        # Replace ground truth frames in output if configured
        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple = None,
        n_sample: int = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        uncond_ref_dropout: bool = False,
        no_ref_images: bool = False,
        ref_guidance: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples with optional CFG on reference images.

        This overrides the parent method to pass uncond_ref_dropout to get_velocity_fn_from_batch.

        Args:
            uncond_ref_dropout: If True, use zero ref_latents for unconditional path in CFG.
                               This enables classifier-free guidance on reference images.
            no_ref_images: If True, use zero ref_latents for BOTH condition and uncondition.
                          This completely disables reference image conditioning.
            ref_guidance: If > 0, use decomposed CFG with separate ref guidance scale.
        """
        import tqdm
        from cosmos_predict2._src.imaginaire.flags import INTERNAL
        from cosmos_predict2._src.imaginaire.utils import misc
        from cosmos_predict2._src.imaginaire.utils.context_parallel import (
            broadcast_split_tensor,
            cat_outputs_cp,
            find_split,
        )
        from einops import rearrange

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps,
            device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )

        timesteps = self.sample_scheduler.timesteps
        # Pass uncond_ref_dropout, no_ref_images, and ref_guidance to get_velocity_fn_from_batch
        velocity_fn = self.get_velocity_fn_from_batch(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            uncond_ref_dropout=uncond_ref_dropout,
            no_ref_images=no_ref_images,
            ref_guidance=ref_guidance,
        )
        use_spatial_split = False
        if self.net.is_context_parallel_enabled:
            cp_size = len(torch.distributed.get_process_group_ranks(self.get_context_parallel_group()))
            n_views = noise.shape[2] // self.get_num_video_latent_frames()
            state_t = noise.shape[2] // n_views
            use_spatial_split = cp_size > state_t or state_t % cp_size != 0
            after_split_shape = None
            if use_spatial_split:
                after_split_shape = find_split(noise.shape, cp_size, view_factor=n_views)
                after_split_shape = torch.Size([after_split_shape[0] * n_views, *after_split_shape[1:]])
                noise = rearrange(noise, "b c t h w -> b c (t h w)")
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())
            if use_spatial_split:
                noise = rearrange(noise, "b c (t h w) -> b c t h w", t=after_split_shape[0], h=after_split_shape[1])
        latents = noise

        if INTERNAL:
            timesteps_iter = timesteps
        else:
            timesteps_iter = tqdm.tqdm(timesteps, desc="Generating samples", total=len(timesteps))

        for _, t in enumerate(timesteps_iter):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            if use_spatial_split:
                latents = rearrange(latents, "b c t h w -> b c (t h w)")
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())
            if use_spatial_split:
                latents = rearrange(latents, "b c (t h w) -> b c t h w", t=state_shape[1], h=state_shape[2])

        return latents
