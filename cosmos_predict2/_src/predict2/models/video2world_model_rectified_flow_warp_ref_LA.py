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

"""Extended Video2World model with warp + reference + LA (Lookahead) frame conditioning.

This model supports:
1. Warped video conditioning (channel concat)
2. Reference image conditioning (token concat)
3. LA frame conditioning (token concat at actual temporal position) - NEW

LA frame is:
- A single GT frame from beyond the generated window (latent t=20~24)
- Encoded via 3D VAE
- Token-concatenated with RoPE at its actual temporal position
- Optionally dropped (zeroed) with la_dropout probability during training
"""

from dataclasses import replace
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import Text2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow_warp_ref import (
    Video2WorldModelRectifiedFlowWarpRef,
)
from cosmos_predict2._src.predict2.models.warp_ref_LA_conditioner import (
    Video2WorldConditionWarpRefLA,
    encode_la_frame,
)
from cosmos_predict2._src.predict2.models.warp_ref_conditioner import (
    encode_reference_frames,
)


class Video2WorldModelRectifiedFlowWarpRefLA(Video2WorldModelRectifiedFlowWarpRef):
    """Extended Video2World model with warp + reference + LA frame conditioning.

    Extends Video2WorldModelRectifiedFlowWarpRef to add LA frame:
    - LA frame encoded via 3D VAE -> la_latents [B, 1, C, 1, H, W]
    - Token-concatenated with RoPE at actual temporal position t (20~24)
    - LA dropout: with config.la_dropout probability, zero out la_latents during training
    """

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldConditionWarpRefLA]:
        """Get data and condition, encoding warped video, reference frames, and LA frame.

        Args:
            data_batch: Batch of data containing all parent fields plus:
                - la_frame: [B, 1, C, H, W] LA frame in uint8 [0, 255]
                - la_time_indices: [B, 1] latent time indices (20~24)

        Returns:
            Tuple of (raw_state, latent_state, condition) where condition includes
            conditional_latents (warped), ref_latents (reference), and la_latents (LA)
        """
        # Get data and condition from parent (includes warped + ref)
        raw_state, latent_state, parent_condition = super().get_data_and_condition(data_batch)

        # Convert parent condition to LA variant
        parent_kwargs = parent_condition.to_dict(skip_underscore=False)
        condition = Video2WorldConditionWarpRefLA(**parent_kwargs)

        # =========================================
        # Encode LA frame (LA SPECIFIC)
        # =========================================
        if "la_frame" in data_batch and "la_time_indices" in data_batch:
            la_frame = data_batch["la_frame"]  # [B, 1, C, H, W] uint8
            la_frame = la_frame.to(device=latent_state.device)

            # Encode LA frame via 3D VAE
            la_latents = encode_la_frame(self.tokenizer, la_frame)

            # Get LA time indices
            la_time_indices = data_batch["la_time_indices"].to(device=latent_state.device)

            # LA mask: from data_batch if available, else default to 1.0 (active)
            if "la_mask" in data_batch:
                la_mask = data_batch["la_mask"].to(device=latent_state.device)
            else:
                la_mask = torch.ones(la_frame.shape[0], 1, device=latent_state.device)

            # LA dropout: zero out latents with configured probability during training
            if self.training and self.config.la_dropout > 0:
                if torch.rand(1).item() < self.config.la_dropout:
                    la_latents = torch.zeros_like(la_latents)
                    la_mask = torch.zeros_like(la_mask)
                    log.debug("LA dropout applied: zeroed la_latents and la_mask")

            log.debug(
                f"Encoded LA frame: input={la_frame.shape}, "
                f"latents={la_latents.shape}, time_indices={la_time_indices[0]}"
            )

            # Build replace kwargs
            la_replace_kwargs = dict(
                la_latents=la_latents,
                la_time_indices=la_time_indices,
                la_mask=la_mask,
            )

            # Add LA camera parameters if available
            if "la_extrinsics" in data_batch:
                la_replace_kwargs["la_extrinsics"] = data_batch["la_extrinsics"].to(device=latent_state.device)
                la_replace_kwargs["la_intrinsics"] = data_batch["la_intrinsics"].to(device=latent_state.device)
                log.debug(f"Extracted LA camera params: extrinsics={la_replace_kwargs['la_extrinsics'].shape}")

            condition = replace(condition, **la_replace_kwargs)
        else:
            log.debug("No LA frame in data_batch")

        return raw_state, latent_state, condition

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> DenoisePrediction:
        """Denoise with warped video, reference image, and LA frame conditioning.

        Extends parent to also pass LA data to the network.
        """
        # Apply frame_replace conditioning (same as parent)
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Noisy conditioning: when num_conditional_frames >= noisy_conditioning_min_frames,
            # with noisy_conditioning_prob probability, add noisy_conditioning_scale noise
            # to the conditional frames going INTO the network.
            # The velocity replacement after the network still uses clean GT.
            if self.training and self.config.noisy_conditioning_prob > 0:
                noise_prob = self.config.noisy_conditioning_prob
                noise_scale = self.config.noisy_conditioning_scale
                min_frames = self.config.noisy_conditioning_min_frames
                # Per-sample num_conditional_frames = sum of mask along T dim
                num_cond_per_sample = condition.condition_video_input_mask_B_C_T_H_W[:, 0, :, 0, 0].sum(dim=1)  # [B]
                for b in range(xt_B_C_T_H_W.shape[0]):
                    if num_cond_per_sample[b].item() >= min_frames and torch.rand(1).item() < noise_prob:
                        cond_noise = noise_scale * torch.randn_like(condition_state_in_B_C_T_H_W[b])
                        condition_state_in_B_C_T_H_W[b] = condition_state_in_B_C_T_H_W[b] + cond_noise

            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

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

        # Extract conditional latents (warped video)
        conditional_latents = condition.conditional_latents

        # Extract reference latents
        ref_latents = getattr(condition, 'ref_latents', None)
        ref_mask = getattr(condition, 'ref_mask', None)

        # Extract LA latents (NEW)
        la_latents = getattr(condition, 'la_latents', None)
        la_time_indices = getattr(condition, 'la_time_indices', None)

        # Forward pass through network
        condition_dict = condition.to_dict()

        # Remove raw camera keys from to_dict() - they'll be remapped to network param names below
        for _cam_key in ['main_extrinsics', 'main_intrinsics', 'ref_extrinsics', 'ref_intrinsics',
                         'la_extrinsics', 'la_intrinsics', 'cam_mask', 'la_mask']:
            condition_dict.pop(_cam_key, None)

        # Add warped conditional latents (channel concat in network)
        if conditional_latents is not None:
            condition_dict['conditional_latents_B_C_T_H_W'] = conditional_latents

        # Add reference latents (token concat in network)
        if ref_latents is not None:
            condition_dict['ref_latents_B_N_C_T_H_W'] = ref_latents
            condition_dict['ref_mask_B_N'] = ref_mask
            ref_time_indices = getattr(condition, 'ref_time_indices', None)
            if ref_time_indices is not None:
                condition_dict['ref_time_indices_B_N'] = ref_time_indices

        # Add LA latents (token concat in network) (NEW)
        if la_latents is not None:
            condition_dict['la_latents_B_N_C_T_H_W'] = la_latents
            if la_time_indices is not None:
                condition_dict['la_time_indices_B_N'] = la_time_indices
            la_mask = getattr(condition, 'la_mask', None)
            if la_mask is not None:
                condition_dict['la_mask_B_N'] = la_mask

        # Add warp mask (WARP MASK SPECIFIC)
        warp_mask = getattr(condition, 'warp_mask', None)
        if warp_mask is not None:
            condition_dict['warp_mask_B_1_T_H_W'] = warp_mask

        # Add camera parameters (CAMERA SPECIFIC)
        # Remap condition field names to network forward parameter names
        main_extrinsics = getattr(condition, 'main_extrinsics', None)
        if main_extrinsics is not None:
            condition_dict['main_extrinsics_B_T_3_4'] = main_extrinsics
            condition_dict['main_intrinsics_B_T_3_3'] = getattr(condition, 'main_intrinsics', None)
        ref_extrinsics = getattr(condition, 'ref_extrinsics', None)
        if ref_extrinsics is not None:
            condition_dict['ref_extrinsics_B_N_3_4'] = ref_extrinsics
            condition_dict['ref_intrinsics_B_N_3_3'] = getattr(condition, 'ref_intrinsics', None)
        la_extrinsics = getattr(condition, 'la_extrinsics', None)
        if la_extrinsics is not None:
            condition_dict['la_extrinsics_B_N_3_4'] = la_extrinsics
            condition_dict['la_intrinsics_B_N_3_3'] = getattr(condition, 'la_intrinsics', None)
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

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        uncond_ref_dropout: bool = False,
        no_ref_images: bool = False,
        ref_guidance: float = 0.0,
    ) -> Callable:
        """Override to preserve LA latents in condition/uncondition.

        LA frame is always present in both cond and uncond paths (no LA dropout).
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

        # Convert to Video2WorldConditionWarpRefLA
        condition = Video2WorldConditionWarpRefLA(**base_condition.to_dict(skip_underscore=False))
        uncondition = Video2WorldConditionWarpRefLA(**base_uncondition.to_dict(skip_underscore=False))

        # Update data type
        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        # Get x0 and full condition with warped + ref + LA latents
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
        else:
            raise ValueError('Warped conditional latents do not exist.')

        # Add reference latents (REF SPECIFIC) - same logic as parent
        if condition_full.ref_latents is not None and condition_full.ref_mask is not None:
            zero_ref_latents = torch.zeros_like(condition_full.ref_latents)
            zero_ref_mask = torch.zeros_like(condition_full.ref_mask)
            ref_time_indices = condition_full.ref_time_indices

            if no_ref_images:
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
            elif ref_guidance > 0:
                condition = replace(
                    condition,
                    ref_latents=condition_full.ref_latents,
                    ref_mask=condition_full.ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                uncondition = replace(
                    uncondition,
                    ref_latents=condition_full.ref_latents,
                    ref_mask=condition_full.ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                uncond_ref = replace(
                    condition,
                    ref_latents=zero_ref_latents,
                    ref_mask=zero_ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                uncond_null = replace(
                    uncondition,
                    ref_latents=zero_ref_latents,
                    ref_mask=zero_ref_mask,
                    ref_time_indices=ref_time_indices,
                )
            else:
                condition = replace(
                    condition,
                    ref_latents=condition_full.ref_latents,
                    ref_mask=condition_full.ref_mask,
                    ref_time_indices=ref_time_indices,
                )
                if uncond_ref_dropout:
                    uncondition = replace(
                        uncondition,
                        ref_latents=zero_ref_latents,
                        ref_mask=zero_ref_mask,
                        ref_time_indices=ref_time_indices,
                    )
                else:
                    uncondition = replace(
                        uncondition,
                        ref_latents=condition_full.ref_latents,
                        ref_mask=condition_full.ref_mask,
                        ref_time_indices=ref_time_indices,
                    )
        else:
            raise ValueError('Reference latents do not exist.')

        # Add LA latents (LA SPECIFIC - always present, no dropout)
        if condition_full.la_latents is not None:
            la_latents = condition_full.la_latents
            la_time_indices = condition_full.la_time_indices

            condition = replace(condition, la_latents=la_latents, la_time_indices=la_time_indices)
            uncondition = replace(uncondition, la_latents=la_latents, la_time_indices=la_time_indices)

            if ref_guidance > 0 and not no_ref_images:
                uncond_ref = replace(uncond_ref, la_latents=la_latents, la_time_indices=la_time_indices)
                uncond_null = replace(uncond_null, la_latents=la_latents, la_time_indices=la_time_indices)

            log.debug(f"Added LA latents: shape={la_latents.shape}, time_indices={la_time_indices[0]}")
        else:
            raise ValueError('LA latents do not exist. Ensure la_frame and la_time_indices are in data_batch.')

        # Add warp_mask to condition/uncondition (always same, no dropout)
        if condition_full.warp_mask is not None:
            condition = replace(condition, warp_mask=condition_full.warp_mask)
            uncondition = replace(uncondition, warp_mask=condition_full.warp_mask)
            if ref_guidance > 0 and not no_ref_images:
                uncond_ref = replace(uncond_ref, warp_mask=condition_full.warp_mask)
                uncond_null = replace(uncond_null, warp_mask=condition_full.warp_mask)

        # Add camera params (including cam_mask)
        cam_fields = {}
        for field in ['main_extrinsics', 'main_intrinsics', 'ref_extrinsics', 'ref_intrinsics',
                       'la_extrinsics', 'la_intrinsics', 'cam_mask']:
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

        # =========================================
        # Validate ALL conditions are populated
        # =========================================
        def _validate_condition(cond, label):
            required_fields = {
                # Warp conditioning
                'conditional_latents': 'warped_video latents',
                # Reference conditioning
                'ref_latents': 'reference image latents',
                'ref_mask': 'reference mask',
                'ref_time_indices': 'reference time indices',
                # LA conditioning
                'la_latents': 'LA frame latents',
                'la_time_indices': 'LA time indices',
                # Video conditioning
                'gt_frames': 'GT frames (x0)',
                'condition_video_input_mask_B_C_T_H_W': 'conditional frame mask',
                # Camera conditioning
                'main_extrinsics': 'main camera extrinsics',
                'main_intrinsics': 'main camera intrinsics',
                'ref_extrinsics': 'reference camera extrinsics',
                'ref_intrinsics': 'reference camera intrinsics',
                'la_extrinsics': 'LA camera extrinsics',
                'la_intrinsics': 'LA camera intrinsics',
                'cam_mask': 'camera mask',
                # Warp mask
                'warp_mask': 'warp validity mask',
            }
            missing = []
            for field, desc in required_fields.items():
                val = getattr(cond, field, None)
                if val is None:
                    missing.append(f"{field} ({desc})")
                elif isinstance(val, torch.Tensor) and val.numel() == 0:
                    missing.append(f"{field} ({desc}) [empty tensor]")
            if missing:
                raise ValueError(
                    f"[{label}] Missing required conditions:\n  " + "\n  ".join(missing)
                )
            # Log shapes for debugging
            log.info(f"[{label}] All conditions validated:")
            for field in required_fields:
                val = getattr(cond, field, None)
                if isinstance(val, torch.Tensor):
                    log.info(f"  {field}: {list(val.shape)} {val.dtype}")
                else:
                    log.info(f"  {field}: {val}")

        _validate_condition(condition, "condition")
        _validate_condition(uncondition, "uncondition")

        # Return velocity function
        if ref_guidance > 0 and not no_ref_images:
            def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                cond_v = self.denoise(noise, noise_x, timestep, condition)
                uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
                uncond_ref_v = self.denoise(noise, noise_x, timestep, uncond_ref)
                uncond_null_v = self.denoise(noise, noise_x, timestep, uncond_null)
                velocity_pred = uncond_v + guidance * (uncond_ref_v - uncond_null_v) + ref_guidance * (cond_v - uncond_ref_v)
                return velocity_pred
        else:
            def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                cond_v = self.denoise(noise, noise_x, timestep, condition)
                uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
                velocity_pred = uncond_v + guidance * (cond_v - uncond_v)
                return velocity_pred

        return velocity_fn
