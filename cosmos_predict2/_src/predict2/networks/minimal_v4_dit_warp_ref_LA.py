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

"""Modified MiniTrainDIT for warp + reference + Lookahead (LA) frame conditioning.

This module extends MiniTrainDITWarpRef to support:
1. Warped video: Channel concat (same as before)
2. Reference images: Token concat with offset RoPE (same as before)
3. LA frame: Token concat with RoPE at actual temporal position (NEW)

Key additions from minimal_v4_dit_warp_ref.py:
- la_embedder: Separate PatchEmbed for LA frame (same architecture as ref_embedder)
- prepare_la_embedded_sequence(): Embeds LA frame and generates RoPE at actual position t
- Token concat order: [main, la, ref] matching temporal RoPE order (main=0-19, la=20-24, ref=50+)
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.amp as amp
from einops import rearrange, repeat
from torch import nn

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    PatchEmbed,
    SACConfig,
)
from cosmos_predict2._src.predict2.networks.minimal_v4_dit_warp_ref import MiniTrainDITWarpRef


class MiniTrainDITWarpRefLA(MiniTrainDITWarpRef):
    """
    Modified MiniTrainDIT for warp + reference + LA frame conditioning.

    Extends MiniTrainDITWarpRef to add:
    - la_embedder: Separate PatchEmbed for LA frame (17 channels = 16 latent + 1 padding mask)
    - LA frame tokens concatenated with main + ref tokens in temporal dim
    - RoPE at actual temporal position t (20~24) for LA tokens
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_fwd_count = 0
        self._debug_cam_count = 0

        log.info(
            f"MiniTrainDITWarpRefLA initialized: "
            f"la_embedder={self.ref_in_channels + 1}ch (same as ref_embedder)"
        )

    def build_ref_embedder(self):
        """Override to also build la_embedder alongside ref_embedder.

        Called during parent's __init__, before init_weights().
        """
        super().build_ref_embedder()
        self.build_la_embedder()

    def build_la_embedder(self):
        """Build separate patch embedder for LA frame.

        la_embedder uses ref_in_channels + 1 (for padding mask), same as ref_embedder.
        """
        la_in_channels = self.ref_in_channels + 1  # 16 + 1 = 17
        self.la_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=la_in_channels,
            out_channels=self.model_channels,
        )
        self._la_embedder_initialized = False

    def init_weights(self):
        """Override to also initialize la_embedder."""
        super().init_weights()
        self.la_embedder.init_weights()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to also handle la_embedder initialization from x_embedder.

        Same logic as ref_embedder: when la_embedder is missing from checkpoint,
        initialize from x_embedder weights.

        Weight mapping (x_embedder K dims -> la_embedder 68 dims):
        - x_embedder[:, :64] (latent) -> la_embedder[:, :64]
        - x_embedder[:, -4:] (padding, last 4ch) -> la_embedder[:, 64:68]
        """
        # Find keys in state_dict
        la_key = None
        x_key = None
        for key in state_dict.keys():
            if key.endswith("la_embedder.proj.1.weight"):
                la_key = key
            if key.endswith("x_embedder.proj.1.weight"):
                x_key = key

        # Detect if la_embedder was loaded from checkpoint or is still random init.
        # Uses _dcp_missing_keys (set by checkpointer from DCP metadata) for authoritative detection.
        la_needs_init = False
        if la_key is not None:
            result = self._check_embedder_missing_in_checkpoint("la_embedder")
            if result is not None:
                la_needs_init = result
            else:
                # Fallback for non-DCP loading (e.g., .pt checkpoint): la_key exists → assume loaded
                log.info(f"[load_state_dict] la_embedder in state_dict (non-DCP), assuming loaded from checkpoint")
        else:
            la_needs_init = True
            log.info(f"[load_state_dict] la_embedder key not in state_dict, needs init")

        # Initialize la_embedder from x_embedder if needed
        if la_needs_init and x_key is not None:
            x_weight = state_dict[x_key]
            x_in_dim = x_weight.shape[1]

            expected_la_key = x_key.replace("x_embedder", "la_embedder") if la_key is None else la_key
            current_la_weight = self.la_embedder.proj[1].weight
            la_in_dim = current_la_weight.shape[1]

            log.info(f"[load_state_dict] la_embedder missing in checkpoint, initializing from x_embedder")
            log.info(f"[load_state_dict] x_embedder dims: {x_in_dim}, la_embedder dims: {la_in_dim}")

            if x_in_dim >= 68 and la_in_dim == 68:
                la_weight = current_la_weight.clone()
                la_weight[:, :64] = x_weight[:, :64]  # latent
                la_weight[:, 64:68] = x_weight[:, -4:]  # padding (last 4ch)
                state_dict[expected_la_key] = la_weight
                log.info(
                    f"[load_state_dict] SUCCESS: Created la_embedder from x_embedder({x_in_dim}->68), "
                    f"latent[0:64], padding[-4:->64:68]"
                )
                self._la_embedder_initialized = True
            elif x_in_dim == 68 and la_in_dim == 68:
                state_dict[expected_la_key] = x_weight.clone()
                log.info(f"[load_state_dict] SUCCESS: Copied x_embedder(68) directly to la_embedder(68)")
                self._la_embedder_initialized = True
            else:
                log.warning(
                    f"[load_state_dict] Cannot init la_embedder: unexpected dims x={x_in_dim}, la={la_in_dim}"
                )
        elif not la_needs_init:
            log.info(f"[load_state_dict] la_embedder found in checkpoint, loading normally")
            self._la_embedder_initialized = True

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def init_la_embedder_from_x_embedder(self):
        """Initialize la_embedder weights from x_embedder (fallback for forward-time init).

        Same weight mapping logic as ref_embedder init.
        """
        if self._la_embedder_initialized:
            return

        try:
            from torch.distributed._tensor import DTensor

            x_weight = self.x_embedder.proj[1].weight
            la_weight = self.la_embedder.proj[1].weight

            if isinstance(x_weight, DTensor):
                x_weight_local = x_weight.to_local()
            else:
                x_weight_local = x_weight.data

            if isinstance(la_weight, DTensor):
                la_weight_local = la_weight.to_local()
            else:
                la_weight_local = la_weight.data

            x_in_dim = x_weight_local.shape[1]
            la_in_dim = la_weight_local.shape[1]

            log.info(f"[la_embedder init] x_embedder dims: {x_in_dim}, la_embedder dims: {la_in_dim}")

            if x_in_dim >= 68 and la_in_dim == 68:
                la_weight_local[:, :64].copy_(x_weight_local[:, :64])
                la_weight_local[:, 64:68].copy_(x_weight_local[:, -4:])
                log.info(
                    f"[la_embedder init] SUCCESS: Copied x_embedder({x_in_dim}) -> la_embedder(68), "
                    f"latent[0:64], padding[-4:->64:68]"
                )
            else:
                log.warning(
                    f"[la_embedder init] FAILED: Unexpected dims x={x_in_dim}, la={la_in_dim}. "
                    f"Using zero init."
                )
                la_weight_local.zero_()

        except Exception as e:
            log.error(f"[la_embedder init] FAILED with exception: {e}. Using zero init.")
            try:
                la_weight = self.la_embedder.proj[1].weight
                if hasattr(la_weight, 'to_local'):
                    la_weight.to_local().zero_()
                else:
                    la_weight.data.zero_()
            except Exception as e2:
                log.error(f"[la_embedder init] Zero init also failed: {e2}")

        self._la_embedder_initialized = True

    def prepare_la_embedded_sequence(
        self,
        la_latents_B_N_C_T_H_W: torch.Tensor,
        la_time_indices_B_N: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare embedded sequence for LA frame token.

        Args:
            la_latents_B_N_C_T_H_W: LA latents [B, 1, C, 1, H, W]
            la_time_indices_B_N: [B, 1] actual latent time indices (20~24)
            fps: FPS tensor (optional)

        Returns:
            Tuple of:
                - la_tokens: [B, 1, H', W', D] embedded LA tokens
                - la_rope: [B, H' * W', 1, 1, D] per-batch RoPE for LA tokens
        """
        B, N, C, T, H, W = la_latents_B_N_C_T_H_W.shape
        device = la_latents_B_N_C_T_H_W.device
        dtype = la_latents_B_N_C_T_H_W.dtype
        assert N == 1, f"Expected exactly 1 LA frame, got {N}"

        # Add padding mask channel (zeros for LA)
        padding_mask = torch.zeros(B, N, 1, T, H, W, device=device, dtype=dtype)
        la_with_mask = torch.cat([la_latents_B_N_C_T_H_W, padding_mask], dim=2)

        # Reshape for patch embedding: [B * 1, C+1, 1, H, W]
        la_reshaped = rearrange(la_with_mask, "b n c t h w -> (b n) c t h w")

        # Cast to embedder weight dtype
        embedder_dtype = self.la_embedder.proj[1].weight.dtype
        la_reshaped = la_reshaped.to(dtype=embedder_dtype)

        # Apply LA embedder: [B, 1, H', W', D]
        la_embedded = self.la_embedder(la_reshaped)

        _, T_lat, H_lat, W_lat, D = la_embedded.shape
        assert T_lat == 1, f"Expected T_lat=1 for single-frame LA, got {T_lat}"

        # Reshape to [B, 1, H', W', D]
        la_tokens = rearrange(
            la_embedded,
            "(b n) t h w d -> b (n t) h w d",
            b=B,
            n=N,
        )

        # Generate RoPE at actual temporal position t
        la_rope = self._generate_la_rope_embedding(
            H_lat, W_lat, la_time_indices_B_N,
        )

        return la_tokens, la_rope

    def _generate_la_rope_embedding(
        self,
        H_latent: int,
        W_latent: int,
        la_time_indices_B_N: torch.Tensor,
    ) -> torch.Tensor:
        """Generate per-batch RoPE embeddings for LA frame at actual temporal position.

        LA frame uses its actual latent time index (20~24) for the temporal axis.
        Spatial axis uses [0:H, 0:W] (same as main video).

        Each batch item gets its own RoPE based on its la_time_indices,
        so different samples in a batch correctly receive different RoPE positions.

        Args:
            H_latent: Latent height (after patchify)
            W_latent: Latent width (after patchify)
            la_time_indices_B_N: [B, 1] actual latent time indices (20~24)

        Returns:
            Per-batch RoPE embedding for LA frame [B, H * W, 1, 1, D]
        """
        pos_emb = self.pos_embedder
        head_dim = self.model_channels // self.num_heads
        dim_h = head_dim // 6 * 2
        dim_w = dim_h
        dim_t = head_dim - 2 * dim_h

        device = pos_emb.seq.device
        dtype = pos_emb.seq.dtype

        # Compute NTK-scaled theta values
        h_theta = 10000.0 * pos_emb.h_ntk_factor
        w_theta = 10000.0 * pos_emb.w_ntk_factor
        t_theta = 10000.0 * pos_emb.t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** pos_emb.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** pos_emb.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** pos_emb.dim_temporal_range.float())

        # Create spatial indices (same as main video: no offset)
        la_h_indices = torch.arange(H_latent, device=device, dtype=dtype)
        la_w_indices = torch.arange(W_latent, device=device, dtype=dtype)

        # Compute spatial half embeddings (shared across batch)
        half_emb_h = torch.outer(la_h_indices, h_spatial_freqs)
        half_emb_w = torch.outer(la_w_indices, w_spatial_freqs)

        B = la_time_indices_B_N.shape[0]
        all_ropes = []
        for b in range(B):
            # LA uses actual time index (20~24) per batch item
            la_time = la_time_indices_B_N[b, 0].to(device=device, dtype=dtype)
            la_rope_times = la_time.unsqueeze(0)  # [1]

            half_emb_t = torch.outer(la_rope_times, temporal_freqs)

            # Combine into full embedding: [1, H, W, D]
            em_T_H_W_D = torch.cat(
                [
                    repeat(half_emb_t, "t d -> t h w d", h=H_latent, w=W_latent),
                    repeat(half_emb_h, "h d -> t h w d", t=1, w=W_latent),
                    repeat(half_emb_w, "w d -> t h w d", t=1, h=H_latent),
                ] * 2,
                dim=-1,
            )

            # [H*W, 1, 1, D]
            rope_b = rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()
            all_ropes.append(rope_b)

        # Stack to [B, H*W, 1, 1, D]
        return torch.stack(all_ropes, dim=0)

    def _debug_verify_la_embedder(self):
        """Verify la_embedder weights were properly initialized from x_embedder.

        Compares latent channels [0:64] of la_embedder with x_embedder.
        Logs cosine similarity and max absolute difference.
        """
        try:
            from torch.distributed._tensor import DTensor

            x_w = self.x_embedder.proj[1].weight
            la_w = self.la_embedder.proj[1].weight
            ref_w = self.ref_embedder.proj[1].weight

            x_local = x_w.to_local().float() if isinstance(x_w, DTensor) else x_w.data.float()
            la_local = la_w.to_local().float() if isinstance(la_w, DTensor) else la_w.data.float()
            ref_local = ref_w.to_local().float() if isinstance(ref_w, DTensor) else ref_w.data.float()

            x_in = x_local.shape[1]
            la_in = la_local.shape[1]
            ref_in = ref_local.shape[1]

            log.info(f"[DEBUG la_embedder] weight shapes: x_embedder=[*, {x_in}], la_embedder=[*, {la_in}], ref_embedder=[*, {ref_in}]")

            # Compare latent channels [0:64]
            x_latent = x_local[:, :64].reshape(-1)
            la_latent = la_local[:, :64].reshape(-1)
            ref_latent = ref_local[:, :64].reshape(-1)

            cos_la_x = torch.nn.functional.cosine_similarity(la_latent.unsqueeze(0), x_latent.unsqueeze(0)).item()
            cos_ref_x = torch.nn.functional.cosine_similarity(ref_latent.unsqueeze(0), x_latent.unsqueeze(0)).item()
            cos_la_ref = torch.nn.functional.cosine_similarity(la_latent.unsqueeze(0), ref_latent.unsqueeze(0)).item()

            maxdiff_la_x = (la_latent - x_latent).abs().max().item()
            maxdiff_la_ref = (la_latent - ref_latent).abs().max().item()

            log.info(
                f"[DEBUG la_embedder] latent[:64] cosine_sim: la-x={cos_la_x:.6f}, ref-x={cos_ref_x:.6f}, la-ref={cos_la_ref:.6f}"
            )
            log.info(
                f"[DEBUG la_embedder] latent[:64] max_abs_diff: la-x={maxdiff_la_x:.2e}, la-ref={maxdiff_la_ref:.2e}"
            )

            if cos_la_x > 0.999:
                log.info("[DEBUG la_embedder] PASS: la_embedder latent channels match x_embedder")
            else:
                log.warning(f"[DEBUG la_embedder] WARN: la_embedder latent channels diverged from x_embedder (cos={cos_la_x:.6f})")

            # Compare padding channels (last 4ch of x_embedder vs last 4ch of la_embedder)
            if x_in >= 68 and la_in == 68:
                x_pad = x_local[:, -4:].reshape(-1)
                la_pad = la_local[:, -4:].reshape(-1)
                pad_diff = (la_pad - x_pad).abs().max().item()
                log.info(f"[DEBUG la_embedder] padding channel max_diff (x[-4:] vs la[64:68]): {pad_diff:.2e}")

        except Exception as e:
            log.warning(f"[DEBUG la_embedder] verification failed: {e}")

    def _debug_log_forward(
        self,
        B: int, T_main: int, T_la: int, T_ref: int, H: int, W: int,
        la_time_indices_B_N, ref_time_indices_B_N, ref_mask_B_N,
        rope_emb, x_tokens,
    ):
        """Log token shapes and RoPE temporal positions per batch for debugging."""
        HW = H * W
        total_T = T_main + T_la + T_ref

        log.info(
            f"[DEBUG forward #{self._debug_fwd_count}] "
            f"tokens: [B={B}, T={total_T}, H={H}, W={W}] = "
            f"main({T_main}) + la({T_la}) + ref({T_ref})"
        )
        log.info(
            f"[DEBUG forward #{self._debug_fwd_count}] "
            f"x_tokens shape={list(x_tokens.shape)}, "
            f"rope shape={list(rope_emb.shape)}, "
            f"expected rope L={total_T * HW}"
        )

        # Per-batch RoPE temporal summary
        # RoPE is [B, L, 1, 1, D] where L = (T_main + T_la + T_ref) * H * W
        # To extract temporal value, sample first spatial pixel (h=0, w=0) of each temporal slot
        if rope_emb.ndim == 5:
            for b in range(min(B, 4)):  # Log up to 4 batch items
                # Main RoPE times: slots [0:T_main], pick first pixel (index 0, HW, 2*HW, ...)
                main_t_vals = []
                for t in range(min(T_main, 5)):  # first 5 + last 1
                    idx = t * HW
                    main_t_vals.append(f"t{t}={rope_emb[b, idx, 0, 0, 0].item():.4f}")
                if T_main > 5:
                    idx_last = (T_main - 1) * HW
                    main_t_vals.append(f"...t{T_main-1}={rope_emb[b, idx_last, 0, 0, 0].item():.4f}")

                # LA RoPE time
                la_t_vals = []
                if T_la > 0:
                    la_offset = T_main * HW
                    for t in range(T_la):
                        idx = la_offset + t * HW
                        la_t_vals.append(f"la{t}={rope_emb[b, idx, 0, 0, 0].item():.4f}")

                # Ref RoPE times
                ref_t_vals = []
                if T_ref > 0:
                    ref_offset = (T_main + T_la) * HW
                    for t in range(min(T_ref, 5)):
                        idx = ref_offset + t * HW
                        ref_t_vals.append(f"ref{t}={rope_emb[b, idx, 0, 0, 0].item():.4f}")

                log.info(
                    f"[DEBUG forward #{self._debug_fwd_count}] batch[{b}] "
                    f"rope_t(dim0): main=[{', '.join(main_t_vals)}] "
                    f"la=[{', '.join(la_t_vals)}] "
                    f"ref=[{', '.join(ref_t_vals)}]"
                )

                # Also log actual la_time_indices and ref_time_indices
                if la_time_indices_B_N is not None:
                    log.info(
                        f"[DEBUG forward #{self._debug_fwd_count}] batch[{b}] "
                        f"la_time_indices={la_time_indices_B_N[b].tolist()}"
                    )
                if ref_time_indices_B_N is not None and ref_mask_B_N is not None:
                    ref_ti = ref_time_indices_B_N[b].tolist()
                    ref_m = ref_mask_B_N[b].tolist()
                    log.info(
                        f"[DEBUG forward #{self._debug_fwd_count}] batch[{b}] "
                        f"ref_time_indices={ref_ti}, ref_mask={ref_m}"
                    )

        # Verify la_embedder weights on first call
        if self._debug_fwd_count == 0:
            self._debug_verify_la_embedder()

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        # Reference inputs
        ref_latents_B_N_C_T_H_W: Optional[torch.Tensor] = None,
        ref_mask_B_N: Optional[torch.Tensor] = None,
        ref_time_indices_B_N: Optional[torch.Tensor] = None,
        # LA inputs (NEW)
        la_latents_B_N_C_T_H_W: Optional[torch.Tensor] = None,
        la_time_indices_B_N: Optional[torch.Tensor] = None,
        la_mask_B_N: Optional[torch.Tensor] = None,
        # Camera parameters
        main_extrinsics_B_T_3_4: Optional[torch.Tensor] = None,
        main_intrinsics_B_T_3_3: Optional[torch.Tensor] = None,
        ref_extrinsics_B_N_3_4: Optional[torch.Tensor] = None,
        ref_intrinsics_B_N_3_3: Optional[torch.Tensor] = None,
        la_extrinsics_B_N_3_4: Optional[torch.Tensor] = None,
        la_intrinsics_B_N_3_3: Optional[torch.Tensor] = None,
        cam_mask_B: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with LA + reference token concatenation.

        Token concatenation order: [main_video, la_frame, ref_frames]
        RoPE positions: main=0-19, la=20-24 (actual position), ref=50+

        Args:
            All args from parent, plus:
            la_latents_B_N_C_T_H_W: LA latents [B, 1, 16, 1, H, W] (NEW)
            la_time_indices_B_N: LA time indices [B, 1] (20~24) (NEW)

        Returns:
            Model output (velocity field prediction) - only for main tokens
        """
        assert isinstance(data_type, DataType)

        # Prepare main video tokens
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        B, T_main, H, W, D = x_B_T_H_W_D.shape
        T_la = 0
        T_ref = 0

        # Inject camera embedding into main tokens (CAMERA SPECIFIC)
        if self.use_camera_embedding and main_extrinsics_B_T_3_4 is not None:
            # One-time verification: log CameraNet weight stats after checkpoint loading
            if not self._camera_net_verified:
                self._camera_net_verified = True
                try:
                    fp_w = self.camera_net.final_proj.weight
                    fp_b = self.camera_net.final_proj.bias
                    if hasattr(fp_w, 'to_local'):
                        fp_w = fp_w.to_local()
                        fp_b = fp_b.to_local()
                    fp_w_max = fp_w.abs().max().item()
                    fp_b_max = fp_b.abs().max().item()
                    cp_w = self.camera_net.camera_proj.weight
                    if hasattr(cp_w, 'to_local'):
                        cp_w = cp_w.to_local()
                    cp_w_norm = cp_w.norm().item()
                    scale_val = self.camera_net.scale.item() if not hasattr(self.camera_net.scale, 'to_local') else self.camera_net.scale.to_local().item()

                    is_fresh_init = fp_w_max < 1e-8 and fp_b_max < 1e-8
                    log.info(
                        f"[CameraNet verify] final_proj.weight max={fp_w_max:.8f}, "
                        f"final_proj.bias max={fp_b_max:.8f}, "
                        f"camera_proj.weight norm={cp_w_norm:.4f}, scale={scale_val:.4f}"
                    )
                    if is_fresh_init:
                        log.info("[CameraNet verify] final_proj is ZERO → fresh init confirmed (output=0, pretrained behavior preserved)")
                    else:
                        log.info("[CameraNet verify] final_proj is NON-ZERO → resumed from trained checkpoint")
                except Exception as e:
                    log.warning(f"[CameraNet verify] Could not verify weights: {e}")

            from cosmos_predict2._src.predict2.networks.camera_net import generate_raymap
            main_raymap = generate_raymap(
                main_extrinsics_B_T_3_4, main_intrinsics_B_T_3_3, self.video_size
            )
            main_cam_emb = self.camera_net(main_raymap.to(dtype=x_B_T_H_W_D.dtype))
            if cam_mask_B is not None:
                main_cam_emb = main_cam_emb * cam_mask_B[:, None, None, None, None]
            x_B_T_H_W_D = x_B_T_H_W_D + main_cam_emb

            # # ========== DEBUG: Log CameraNet weights and output every forward ==========
            # try:
            #     from torch.distributed._tensor import DTensor
            #
            #     def _to_local(p):
            #         return p.to_local() if isinstance(p, DTensor) else p.data
            #
            #     cam = self.camera_net
            #
            #     # encode_first Conv2d weight
            #     ef_w = _to_local(cam.encode_first[0].weight)
            #     ef_w_norm = ef_w.norm().item()
            #
            #     # encode_second Conv2d weight
            #     es_w = _to_local(cam.encode_second[0].weight)
            #     es_w_norm = es_w.norm().item()
            #
            #     # final_proj (zero-initialized, key indicator of learning)
            #     fp_w = _to_local(cam.final_proj.weight)
            #     fp_b = _to_local(cam.final_proj.bias)
            #     fp_w_norm = fp_w.norm().item()
            #     fp_w_max = fp_w.abs().max().item()
            #     fp_b_norm = fp_b.norm().item()
            #
            #     # camera_proj (Xavier-initialized)
            #     cp_w = _to_local(cam.camera_proj.weight)
            #     cp_w_norm = cp_w.norm().item()
            #
            #     # scale (learnable, init=1.0)
            #     scale_val = cam.scale.item() if not isinstance(cam.scale, DTensor) else _to_local(cam.scale).item()
            #
            #     # camera embedding output stats
            #     cam_emb_norm = main_cam_emb.norm().item()
            #     cam_emb_mean = main_cam_emb.mean().item()
            #     cam_emb_max = main_cam_emb.abs().max().item()
            #
            #     log.info(
            #         f"[DEBUG CAM] iter={self._debug_cam_count} "
            #         f"encode_first.w norm={ef_w_norm:.6f}, "
            #         f"encode_second.w norm={es_w_norm:.6f}, "
            #         f"final_proj.w norm={fp_w_norm:.8f} max={fp_w_max:.8f}, "
            #         f"final_proj.b norm={fp_b_norm:.8f}, "
            #         f"camera_proj.w norm={cp_w_norm:.6f}, "
            #         f"scale={scale_val:.6f}"
            #     )
            #     log.info(
            #         f"[DEBUG CAM] iter={self._debug_cam_count} "
            #         f"cam_emb output norm={cam_emb_norm:.6f}, "
            #         f"mean={cam_emb_mean:.8f}, "
            #         f"abs_max={cam_emb_max:.8f}"
            #     )
            # except Exception as e:
            #     log.warning(f"[DEBUG CAM] iter={self._debug_cam_count} logging failed: {e}")
            # self._debug_cam_count += 1
            # # ========== END DEBUG CAM ==========

        # Prepare LA tokens first (RoPE 20-24, right after main 0-19)
        if la_latents_B_N_C_T_H_W is not None and la_time_indices_B_N is not None:
            la_tokens, la_rope = self.prepare_la_embedded_sequence(
                la_latents_B_N_C_T_H_W,
                la_time_indices_B_N,
                fps=fps,
            )
            T_la = la_tokens.shape[1]  # 1

            # Inject camera embedding into LA tokens (CAMERA SPECIFIC)
            if self.use_camera_embedding and la_extrinsics_B_N_3_4 is not None:
                from cosmos_predict2._src.predict2.networks.camera_net import generate_raymap
                log.info(f"[DEBUG] la_extrinsics shape={la_extrinsics_B_N_3_4.shape}, la_intrinsics shape={la_intrinsics_B_N_3_3.shape if la_intrinsics_B_N_3_3 is not None else None}")
                B_la, N_la = la_extrinsics_B_N_3_4.shape[:2]
                la_raymap = generate_raymap(
                    la_extrinsics_B_N_3_4.reshape(B_la * N_la, 1, 3, 4),
                    la_intrinsics_B_N_3_3.reshape(B_la * N_la, 1, 3, 3),
                    self.video_size,
                )
                la_cam_emb = self.camera_net(la_raymap.to(dtype=la_tokens.dtype))
                la_cam_emb = la_cam_emb.reshape(B_la, N_la, H, W, D)
                if la_mask_B_N is not None:
                    la_cam_emb = la_cam_emb * la_mask_B_N[:, :, None, None, None]
                if cam_mask_B is not None:
                    la_cam_emb = la_cam_emb * cam_mask_B[:, None, None, None, None]
                la_tokens = la_tokens + la_cam_emb

            # Concatenate LA tokens after main tokens
            x_B_T_H_W_D = torch.cat([x_B_T_H_W_D, la_tokens], dim=1)

            # Per-batch RoPE: expand main_rope [L, 1, 1, D] -> [B, L, 1, 1, D],
            # then cat with la_rope [B, L_la, 1, 1, D] on dim=1
            rope_emb_L_1_1_D = rope_emb_L_1_1_D.unsqueeze(0).expand(B, -1, -1, -1, -1)
            rope_emb_L_1_1_D = torch.cat([rope_emb_L_1_1_D, la_rope], dim=1)

            if extra_pos_emb is not None:
                la_extra_pos = torch.zeros(B, T_la, H, W, D, device=extra_pos_emb.device, dtype=extra_pos_emb.dtype)
                extra_pos_emb = torch.cat([extra_pos_emb, la_extra_pos], dim=1)

        # Prepare reference tokens (RoPE 50+, after LA)
        if ref_latents_B_N_C_T_H_W is not None and ref_mask_B_N is not None:
            ref_tokens, ref_rope = self.prepare_ref_embedded_sequence(
                ref_latents_B_N_C_T_H_W,
                ref_mask_B_N,
                fps=fps,
                ref_time_indices_B_N=ref_time_indices_B_N,
                main_H_latent=H,
                main_W_latent=W,
            )
            T_ref = ref_tokens.shape[1]

            # Inject camera embedding into ref tokens (CAMERA SPECIFIC)
            if self.use_camera_embedding and ref_extrinsics_B_N_3_4 is not None:
                from cosmos_predict2._src.predict2.networks.camera_net import generate_raymap
                B_ref_cam, N_ref_cam = ref_extrinsics_B_N_3_4.shape[:2]
                ref_raymap = generate_raymap(
                    ref_extrinsics_B_N_3_4.reshape(B_ref_cam * N_ref_cam, 1, 3, 4),
                    ref_intrinsics_B_N_3_3.reshape(B_ref_cam * N_ref_cam, 1, 3, 3),
                    self.video_size,
                )
                ref_cam_emb = self.camera_net(ref_raymap.to(dtype=ref_tokens.dtype))
                ref_cam_emb = ref_cam_emb.reshape(B_ref_cam, N_ref_cam, H, W, D)
                ref_cam_emb = ref_cam_emb * ref_mask_B_N[:, :, None, None, None]
                if cam_mask_B is not None:
                    ref_cam_emb = ref_cam_emb * cam_mask_B[:, None, None, None, None]
                ref_tokens = ref_tokens + ref_cam_emb

            # Concatenate ref tokens after LA tokens
            x_B_T_H_W_D = torch.cat([x_B_T_H_W_D, ref_tokens], dim=1)

            # Per-batch RoPE: if still 4D (no LA tokens), expand to 5D first
            if rope_emb_L_1_1_D.ndim == 4:
                rope_emb_L_1_1_D = rope_emb_L_1_1_D.unsqueeze(0).expand(B, -1, -1, -1, -1)
            rope_emb_L_1_1_D = torch.cat([rope_emb_L_1_1_D, ref_rope], dim=1)

            if extra_pos_emb is not None:
                ref_extra_pos = torch.zeros(B, T_ref, H, W, D, device=extra_pos_emb.device, dtype=extra_pos_emb.dtype)
                extra_pos_emb = torch.cat([extra_pos_emb, ref_extra_pos], dim=1)

        # =========================================
        # Debug logging (first 2 forward calls only)
        # =========================================
        if self._debug_fwd_count < 2:
            self._debug_log_forward(
                B, T_main, T_la, T_ref, H, W,
                la_time_indices_B_N, ref_time_indices_B_N, ref_mask_B_N,
                rope_emb_L_1_1_D, x_B_T_H_W_D,
            )
            self._debug_fwd_count += 1

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb.shape

        # Process through blocks (all tokens treated identically)
        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )

            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_main = x_B_T_H_W_D[:, :T_main, :, :, :]
                x_reshaped = rearrange(x_main, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped)

        # Extract main tokens at output (discard ref + LA tokens)
        x_main_B_T_H_W_D = x_B_T_H_W_D[:, :T_main, :, :, :]

        x_B_T_H_W_O = self.final_layer(x_main_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        if intermediate_feature_ids:
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp
