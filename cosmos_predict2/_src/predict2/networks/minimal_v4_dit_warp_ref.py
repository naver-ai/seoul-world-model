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

"""Modified MiniTrainDIT for warp + reference image conditioning.

This module extends the base DiT to support:
1. Warped video: Channel concat (same as before)
2. Reference images: Token concat with offset RoPE

Key changes from minimal_v4_dit.py:
- BlockWarpRef: Modified Block with reference token concatenation in self-attention
- MiniTrainDITWarpRef: Modified MiniTrainDIT with ref_embedder and reference processing
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.amp as amp
import transformer_engine as te
from einops import rearrange, repeat
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torchvision import transforms

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.utils.kv_cache import KVCacheConfig

# Import everything we need from the base minimal_v4_dit
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
    Attention,
    FinalLayer,
    GPT2FeedForward,
    I2VCrossAttention,
    LearnablePosEmbAxis,
    PatchEmbed,
    SACConfig,
    CheckpointMode,
    Timesteps,
    TimestepEmbedding,
    VideoRopePosition3DEmb,
    VideoSize,
    WeightTrainingStat,
)


class BlockWarpRef(nn.Module):
    """
    Standard Block for TIC-FT style processing.

    When reference tokens are concatenated in the temporal dimension of x_B_T_H_W_D,
    they are processed IDENTICALLY to main tokens through all layers:
    - Same LayerNorm with AdaLN modulation
    - Same attention
    - Same cross-attention
    - Same MLP

    The only difference is in RoPE indices (handled externally before calling this block).
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.num_heads = num_heads
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)

        if image_context_dim is None:
            self.cross_attn = Attention(
                x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend
            )
        else:
            self.cross_attn = I2VCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                img_latent_dim=image_context_dim,
                qkv_format="bshd",
                backend=backend,
            )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.cp_size = None
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        self.cp_size = None if ranks is None else len(ranks)
        self.self_attn.set_context_parallel_group(
            process_group=process_group,
            ranks=ranks,
            stream=stream,
            cp_comm_type=cp_comm_type,
        )

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
    ) -> torch.Tensor:
        """Forward pass (TIC-FT style: all tokens processed identically).

        When ref tokens are concatenated in temporal dimension, they're processed
        exactly like main tokens. RoPE handles position differentiation.

        Args:
            x_B_T_H_W_D: Video tokens [B, T, H, W, D] (may include ref tokens in T)
            emb_B_T_D: Timestep embeddings [B, T, D] (extended for ref frames)
            crossattn_emb: Cross-attention embeddings (text)
            rope_emb_L_1_1_D: RoPE [L, 1, 1, D] (concatenated main + ref RoPE)
            adaln_lora_B_T_3D: AdaLN LoRA embeddings
            extra_per_block_pos_emb: Extra positional embeddings
            kv_cache_cfg: KV cache config

        Returns:
            Output tokens [B, T, H, W, D]
        """
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        # Self-attention
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )

        video_size = VideoSize(T=T, H=H, W=W)
        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        tokens_B_L_D = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        attn_output_B_L_D = self.self_attn(
            tokens_B_L_D,
            None,
            rope_emb=rope_emb_L_1_1_D,
            video_size=video_size,
            kv_cache_cfg=kv_cache_cfg,
        )
        result_B_T_H_W_D = rearrange(attn_output_B_L_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        # Cross-attention
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        result_B_T_H_W_D = rearrange(
            self.cross_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_cross_attn_B_T_1_1_D * result_B_T_H_W_D

        # MLP
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D

        return x_B_T_H_W_D


class MiniTrainDITWarpRef(WeightTrainingStat):
    """
    Modified MiniTrainDIT for warp + reference image conditioning.

    Supports:
    1. Warped video latents (channel concat) - handled by caller
    2. Reference images (token concat) - handled here with ref_embedder

    Reference images are:
    - Encoded via separate ref_embedder (same architecture as x_embedder)
    - Concatenated as tokens with offset RoPE (time offset to avoid collision)
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "transformer_engine",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: Optional[int] = None,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        sac_config: SACConfig = SACConfig(),
        n_dense_blocks: int = -1,
        natten_parameters: Union[dict, list] = None,
        use_wan_fp32_strategy: bool = False,
        # Reference image settings (NEW)
        ref_in_channels: int = 16,  # Original latent channels (before warp expansion)
        max_ref_frames: int = 5,
        ref_time_offset: int = 50,
        ref_time_stride: int = 5,
        # Camera embedding settings
        camera_embedding: bool = False,
        video_size: Tuple[int, int] = (704, 1280),
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim

        # Reference settings
        self.ref_in_channels = ref_in_channels
        self.max_ref_frames = max_ref_frames
        self.ref_time_offset = ref_time_offset
        self.ref_time_stride = ref_time_stride

        # Camera embedding settings
        self.use_camera_embedding = camera_embedding
        self.video_size = video_size
        if self.use_camera_embedding:
            from cosmos_predict2._src.predict2.networks.camera_net import CameraNet
            self.camera_net = CameraNet(
                in_channels=6,
                downscale_coef=8,
                out_channels=16,
                patch_size=[1, 2, 2],
                hidden_size=model_channels,
            )
            self._camera_net_verified = False  # For one-time forward verification
            log.info(f"CameraNet initialized: video_size={video_size}, hidden_size={model_channels}")

        self.build_patch_embed()
        self.build_ref_embedder()  # NEW
        self.build_pos_embed()

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        # Use BlockWarpRef instead of Block
        self.blocks = nn.ModuleList(
            [
                BlockWarpRef(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                    image_context_dim=None if extra_image_context_dim is None else model_channels,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            use_wan_fp32_strategy=self.use_wan_fp32_strategy,
        )

        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)
        if extra_image_context_dim is not None:
            self.img_context_proj = nn.Sequential(
                nn.Linear(extra_image_context_dim, model_channels, bias=True),
                nn.GELU(),
            )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

        self._is_context_parallel_enabled = False

        log.info(
            f"MiniTrainDITWarpRef initialized: "
            f"x_embedder={in_channels + (1 if concat_padding_mask else 0)}ch, "
            f"ref_embedder={ref_in_channels + 1}ch, "
            f"max_ref_frames={max_ref_frames}, RoPE: TIME_OFFSET(offset={ref_time_offset}, stride={ref_time_stride})"
        )

    def build_patch_embed(self):
        in_channels = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
        )

    def build_ref_embedder(self):
        """Build separate patch embedder for reference images.

        ref_embedder uses ref_in_channels + 1 (for padding mask).
        This should match the original pretrained x_embedder size.
        """
        ref_in_channels = self.ref_in_channels + 1  # Add padding mask channel
        self.ref_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=ref_in_channels,
            out_channels=self.model_channels,
        )
        # Flag to track if ref_embedder has been initialized from x_embedder
        self._ref_embedder_initialized = False

    def _check_embedder_missing_in_checkpoint(self, embedder_name: str) -> bool:
        """Check if an embedder key was missing from the DCP checkpoint.

        Uses _dcp_missing_keys set by the checkpointer after DCP loading.
        This is the authoritative way to know which keys were in the checkpoint,
        since DCP loads values in-place into model tensors (making data_ptr/snapshot
        approaches unreliable with FSDP + DTensor).

        Returns True if the embedder needs initialization from x_embedder.
        """
        dcp_missing = getattr(self, '_dcp_missing_keys', None)
        if dcp_missing is not None:
            # Check if any ref_embedder key is in the missing set
            is_missing = any(f"{embedder_name}.proj.1.weight" in k for k in dcp_missing)
            if is_missing:
                log.info(f"[load_state_dict] {embedder_name} confirmed MISSING from DCP checkpoint → needs re-init")
            else:
                log.info(f"[load_state_dict] {embedder_name} confirmed PRESENT in DCP checkpoint → loading normally")
            return is_missing
        else:
            log.warning(f"[load_state_dict] _dcp_missing_keys not available (non-DCP load?)")
            return None  # Unknown — caller should fall back

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to handle ref_embedder initialization when loading pretrained checkpoints.

        When loading a checkpoint that doesn't have ref_embedder weights (missing key),
        we initialize ref_embedder from x_embedder weights.

        Weight mapping (x_embedder K dims -> ref_embedder 68 dims):
        - x_embedder[:, :64] (latent) -> ref_embedder[:, :64]
        - x_embedder[:, -4:] (padding, last 4ch) -> ref_embedder[:, 64:68]
        """
        # Find the actual keys in state_dict (handle various prefixes)
        ref_key = None
        x_key = None

        for key in state_dict.keys():
            if key.endswith("ref_embedder.proj.1.weight"):
                ref_key = key
            if key.endswith("x_embedder.proj.1.weight"):
                x_key = key

        # Detect if ref_embedder was loaded from checkpoint or is still random init.
        # Uses _dcp_missing_keys (set by checkpointer from DCP metadata) for authoritative detection.
        ref_needs_init = False
        if ref_key is not None:
            result = self._check_embedder_missing_in_checkpoint("ref_embedder")
            if result is not None:
                ref_needs_init = result
            else:
                # Fallback for non-DCP loading (e.g., .pt checkpoint): ref_key exists → assume loaded
                log.info(f"[load_state_dict] ref_embedder in state_dict (non-DCP), assuming loaded from checkpoint")
        else:
            ref_needs_init = True
            log.info(f"[load_state_dict] ref_embedder key not in state_dict, needs init")

        # Initialize ref_embedder from x_embedder if needed
        if ref_needs_init and x_key is not None:
            x_weight = state_dict[x_key]
            x_in_dim = x_weight.shape[1]

            # Determine expected ref_embedder key by replacing x_embedder with ref_embedder
            expected_ref_key = x_key.replace("x_embedder", "ref_embedder") if ref_key is None else ref_key

            # Get current ref_embedder weight shape
            current_ref_weight = self.ref_embedder.proj[1].weight
            ref_in_dim = current_ref_weight.shape[1]

            log.info(f"[load_state_dict] ref_embedder missing in checkpoint, initializing from x_embedder")
            log.info(f"[load_state_dict] x_embedder dims: {x_in_dim}, ref_embedder dims: {ref_in_dim}")

            # Create ref_embedder weight from x_embedder
            if x_in_dim >= 68 and ref_in_dim == 68:
                # x_embedder: [0:64 latent][...middle...][last 4ch padding]
                # ref_embedder: [0:64 latent][64:68 padding]
                ref_weight = current_ref_weight.clone()
                ref_weight[:, :64] = x_weight[:, :64]  # latent
                ref_weight[:, 64:68] = x_weight[:, -4:]  # padding (last 4ch)
                state_dict[expected_ref_key] = ref_weight
                log.info(
                    f"[load_state_dict] SUCCESS: Created ref_embedder from x_embedder({x_in_dim}->68), "
                    f"latent[0:64], padding[-4:->64:68]"
                )
                self._ref_embedder_initialized = True
            elif x_in_dim == 68 and ref_in_dim == 68:
                # x_embedder matches ref_embedder exactly (original pretrained)
                state_dict[expected_ref_key] = x_weight.clone()
                log.info(f"[load_state_dict] SUCCESS: Copied x_embedder(68) directly to ref_embedder(68)")
                self._ref_embedder_initialized = True
            else:
                log.warning(
                    f"[load_state_dict] Cannot init ref_embedder: unexpected dims x={x_in_dim}, ref={ref_in_dim}. "
                    f"Will use forward-time init as fallback."
                )
        elif not ref_needs_init:
            # ref_embedder exists in checkpoint (resume case) - just load normally
            log.info(f"[load_state_dict] ref_embedder found in checkpoint, loading normally")
            self._ref_embedder_initialized = True
        elif x_key is None:
            log.warning(f"[load_state_dict] Neither ref_embedder nor x_embedder found in checkpoint")

        # =========================================
        # CameraNet checkpoint detection + re-init (CAMERA SPECIFIC)
        # =========================================
        if self.use_camera_embedding:
            # Detect if camera_net keys exist in the checkpoint.
            # _check_embedder_missing_in_checkpoint uses "camera_net.final_proj.weight" pattern.
            dcp_missing = getattr(self, '_dcp_missing_keys', None)
            cam_in_checkpoint = False
            if dcp_missing is not None:
                cam_missing = any("camera_net." in k for k in dcp_missing)
                cam_in_checkpoint = not cam_missing
                if cam_in_checkpoint:
                    log.info("[load_state_dict] CameraNet: PRESENT in DCP checkpoint → loading normally")
                else:
                    log.info("[load_state_dict] CameraNet: MISSING from DCP checkpoint → will re-init after load")
            else:
                # Non-DCP fallback: check if any camera_net key exists in state_dict
                # and has non-zero scale (scale is init to 1.0, so zero means DCP zeroed it out)
                has_cam_keys = any("camera_net." in k for k in state_dict.keys())
                if not has_cam_keys:
                    cam_in_checkpoint = False
                    log.info("[load_state_dict] CameraNet: no keys in state_dict → will re-init after load")
                else:
                    # Check scale value: should be 1.0 for fresh init, 0.0 means DCP zeroed it
                    scale_key = None
                    for key in state_dict.keys():
                        if "camera_net.scale" in key:
                            scale_key = key
                            break
                    if scale_key is not None:
                        scale_val = state_dict[scale_key]
                        is_zero = scale_val.abs().max().item() < 1e-8 if hasattr(scale_val, 'abs') else True
                        cam_in_checkpoint = not is_zero
                    else:
                        cam_in_checkpoint = False
                    if cam_in_checkpoint:
                        log.info("[load_state_dict] CameraNet: LOADED from checkpoint (scale is non-zero)")
                    else:
                        log.info("[load_state_dict] CameraNet: DCP zeroed out init values → will re-init after load")

            self._camera_net_needs_reinit = not cam_in_checkpoint

        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Re-initialize CameraNet if it was not in checkpoint
        # DCP partial loading may zero out keys not found in checkpoint,
        # destroying the proper init (scale=1.0, camera_proj=Xavier, etc.)
        if self.use_camera_embedding and self._camera_net_needs_reinit:
            self._reinit_camera_net()

        # Fix broken GN weights even when resuming from a checkpoint that contains CameraNet.
        # A previous bug in _reinit_camera_net() missed GroupNorm, leaving weight=0.
        # This caused a gradient deadlock (GN output=0 + final_proj weight=0 = no learning).
        # Detect and fix: only touch GN weights, preserve all other trained parameters.
        if self.use_camera_embedding and not self._camera_net_needs_reinit:
            self._fix_camera_net_gn_if_broken()

        # Verify CameraNet state after checkpoint loading (always, for both fresh init and resume)
        if self.use_camera_embedding:
            self._verify_camera_net_after_load()

        return result

    def _fix_camera_net_gn_if_broken(self):
        """Fix GroupNorm weight=0 bug in CameraNet loaded from an existing checkpoint.

        A previous bug left GN weight=0 after init, causing a gradient deadlock
        (GN output always 0 → final_proj never learns → no backward flow).
        This only fixes GN weights to 1.0; all other parameters are preserved.
        """
        try:
            from torch.distributed._tensor import DTensor

            def _get_local(param):
                return param.to_local() if isinstance(param, DTensor) else param.data

            cam_net = self.camera_net
            fixed = False
            for name, module_seq in [("encode_first", cam_net.encode_first),
                                     ("encode_second", cam_net.encode_second)]:
                for m in module_seq.modules():
                    if isinstance(m, torch.nn.GroupNorm):
                        w = _get_local(m.weight)
                        if w.abs().max().item() < 1e-8:
                            w.fill_(1.0)
                            _get_local(m.bias).zero_()
                            log.warning(
                                f"[CameraNet GN fix] {name}: GroupNorm weight was 0 → reset to 1.0"
                            )
                            fixed = True
            if not fixed:
                log.info("[CameraNet GN fix] GroupNorm weights OK, no fix needed")
        except Exception as e:
            log.error(f"[CameraNet GN fix] FAILED: {e}")

    def _reinit_camera_net(self):
        """Re-initialize CameraNet weights after checkpoint loading zeroed them out.

        DCP partial loading with FSDP may zero out parameters for modules not in
        the checkpoint. This restores the proper initialization:
        - encode_first/second: Kaiming (He) init
        - final_proj: ZERO init (critical: ensures output=0 at training start)
        - camera_proj: Xavier init
        - scale: 1.0
        """
        try:
            from torch.distributed._tensor import DTensor

            cam_net = self.camera_net
            log.info("[CameraNet re-init] Re-initializing CameraNet weights after DCP load...")

            def _get_local(param):
                return param.to_local() if isinstance(param, DTensor) else param.data

            # Re-init encode layers (Kaiming for Conv2d, default for GroupNorm)
            for name, module_seq in [("encode_first", cam_net.encode_first),
                                     ("encode_second", cam_net.encode_second)]:
                for m in module_seq.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        w = _get_local(m.weight)
                        torch.nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            _get_local(m.bias).zero_()
                    elif isinstance(m, torch.nn.GroupNorm):
                        _get_local(m.weight).fill_(1.0)
                        _get_local(m.bias).zero_()
                log.info(f"[CameraNet re-init] {name}: Kaiming(Conv2d) + default(GroupNorm) init applied")

            # Re-init final_proj (ZERO - critical for pretrained compatibility)
            fp_w = _get_local(cam_net.final_proj.weight)
            fp_b = _get_local(cam_net.final_proj.bias)
            fp_w.zero_()
            fp_b.zero_()
            log.info("[CameraNet re-init] final_proj: ZERO init applied (output=0 at start)")

            # Re-init camera_proj (Xavier)
            cp_w = _get_local(cam_net.camera_proj.weight)
            torch.nn.init.xavier_uniform_(cp_w)
            if cam_net.camera_proj.bias is not None:
                _get_local(cam_net.camera_proj.bias).zero_()
            log.info(f"[CameraNet re-init] camera_proj: Xavier init applied (norm={cp_w.norm().item():.4f})")

            # Re-init scale to 1.0
            scale = _get_local(cam_net.scale)
            scale.fill_(1.0)
            log.info(f"[CameraNet re-init] scale: set to 1.0")

            log.info("[CameraNet re-init] SUCCESS: CameraNet properly re-initialized")

        except Exception as e:
            log.error(f"[CameraNet re-init] FAILED: {e}. CameraNet may not train correctly!")

    def _verify_camera_net_after_load(self):
        """Log CameraNet weight stats right after load_state_dict for verification."""
        try:
            from torch.distributed._tensor import DTensor

            def _get_local(param):
                return param.to_local() if isinstance(param, DTensor) else param.data

            cam_net = self.camera_net
            fp_w = _get_local(cam_net.final_proj.weight)
            fp_b = _get_local(cam_net.final_proj.bias)
            fp_w_max = fp_w.abs().max().item()
            fp_b_max = fp_b.abs().max().item()

            cp_w = _get_local(cam_net.camera_proj.weight)
            cp_w_norm = cp_w.norm().item()

            scale_val = _get_local(cam_net.scale).item()

            is_zero = fp_w_max < 1e-8 and fp_b_max < 1e-8

            log.info(
                f"[CameraNet verify] final_proj.weight max={fp_w_max:.8f}, "
                f"final_proj.bias max={fp_b_max:.8f}, "
                f"camera_proj.weight norm={cp_w_norm:.4f}, scale={scale_val:.4f}"
            )
            if is_zero:
                log.info("[CameraNet verify] final_proj is ZERO → fresh init confirmed (output=0, pretrained behavior preserved)")
            else:
                log.info(
                    f"[CameraNet verify] final_proj is NON-ZERO → resumed from trained checkpoint "
                    f"(final_proj learned: w_max={fp_w_max:.6f}, b_max={fp_b_max:.6f})"
                )
        except Exception as e:
            log.warning(f"[CameraNet verify] Could not verify weights after load: {e}")

    def init_ref_embedder_from_x_embedder(self):
        """Initialize ref_embedder weights from x_embedder (latent + padding parts only).

        This method copies the pretrained weights from x_embedder to ref_embedder.
        Should be called once after pretrained weights are loaded into x_embedder.

        Weight mapping:
        - x_embedder (K dims): [0:64 latent][...middle...][last 4ch padding]
        - ref_embedder (68 dims): [0:64 latent][64:68 padding]

        We copy:
        - x_embedder[:, :64] (latent) -> ref_embedder[:, :64]
        - x_embedder[:, -4:] (padding, last 4ch) -> ref_embedder[:, 64:68]

        On failure, falls back to zero initialization.
        """
        if self._ref_embedder_initialized:
            return

        try:
            from torch.distributed._tensor import DTensor

            x_weight = self.x_embedder.proj[1].weight
            ref_weight = self.ref_embedder.proj[1].weight

            # Handle DTensor (FSDP sharded weights)
            if isinstance(x_weight, DTensor):
                x_weight_local = x_weight.to_local()
            else:
                x_weight_local = x_weight.data

            if isinstance(ref_weight, DTensor):
                ref_weight_local = ref_weight.to_local()
            else:
                ref_weight_local = ref_weight.data

            x_in_dim = x_weight_local.shape[1]
            ref_in_dim = ref_weight_local.shape[1]

            log.info(f"[ref_embedder init] x_embedder dims: {x_in_dim}, ref_embedder dims: {ref_in_dim}")

            # ========== DEBUG: Log x_embedder weights before copy ==========
            log.info(f"[DEBUG] x_embedder weight shape: {x_weight_local.shape}")
            log.info(f"[DEBUG] x_embedder weight[:5, :8] (first 5 out, first 8 in):\n{x_weight_local[:5, :8]}")
            log.info(f"[DEBUG] x_embedder weight norm per channel section:")
            log.info(f"  - latent[0:64] norm: {x_weight_local[:, :64].norm().item():.6f}")
            if x_in_dim > 68:
                log.info(f"  - middle[64:{x_in_dim-4}] norm: {x_weight_local[:, 64:x_in_dim-4].norm().item():.6f}")
            log.info(f"  - padding[-4:] norm: {x_weight_local[:, -4:].norm().item():.6f}")

            log.info(f"[DEBUG] ref_embedder weight shape before init: {ref_weight_local.shape}")
            log.info(f"[DEBUG] ref_embedder weight[:5, :8] before init:\n{ref_weight_local[:5, :8]}")
            log.info(f"[DEBUG] ref_embedder weight norm before init: {ref_weight_local.norm().item():.6f}")
            # ========== END DEBUG ==========

            if x_in_dim >= 68 and ref_in_dim == 68:
                ref_weight_local[:, :64].copy_(x_weight_local[:, :64])
                ref_weight_local[:, 64:68].copy_(x_weight_local[:, -4:])
                log.info(
                    f"[ref_embedder init] SUCCESS: Copied x_embedder({x_in_dim}) -> ref_embedder(68), "
                    f"latent[0:64], padding[-4:->64:68]"
                )
            else:
                log.warning(
                    f"[ref_embedder init] FAILED: Unexpected dims x={x_in_dim}, ref={ref_in_dim}. "
                    f"Using zero init for ref_embedder."
                )
                ref_weight_local.zero_()

            # ========== DEBUG: Log ref_embedder weights after copy ==========
            log.info(f"[DEBUG] ref_embedder weight[:5, :8] after init:\n{ref_weight_local[:5, :8]}")
            log.info(f"[DEBUG] ref_embedder weight norm after init:")
            log.info(f"  - latent[0:64] norm: {ref_weight_local[:, :64].norm().item():.6f}")
            log.info(f"  - padding[64:68] norm: {ref_weight_local[:, 64:68].norm().item():.6f}")
            # ========== END DEBUG ==========

        except Exception as e:
            log.error(f"[ref_embedder init] FAILED with exception: {e}. Using zero init for ref_embedder.")
            try:
                ref_weight = self.ref_embedder.proj[1].weight
                if hasattr(ref_weight, 'to_local'):
                    ref_weight.to_local().zero_()
                else:
                    ref_weight.data.zero_()
            except Exception as e2:
                log.error(f"[ref_embedder init] Zero init also failed: {e2}")

        self._ref_embedder_initialized = True

    def build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def init_weights(self):
        self.x_embedder.init_weights()
        self.ref_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare embedded sequence for main video tokens."""
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def prepare_ref_embedded_sequence(
        self,
        ref_latents_B_N_C_T_H_W: torch.Tensor,
        ref_mask_B_N: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        ref_time_indices_B_N: Optional[torch.Tensor] = None,
        main_H_latent: Optional[int] = None,
        main_W_latent: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare embedded sequence for reference image tokens.

        Always processes all max_ref_frames with zero padding for consistency
        across all ranks in distributed training (FSDP).

        Returns tokens in [B, T_ref, H, W, D] format for easy temporal concat with main tokens.

        Args:
            ref_latents_B_N_C_T_H_W: Reference latents [B, max_ref, C, 1, H, W]
            ref_mask_B_N: Reference mask [B, max_ref] indicating valid frames
            fps: FPS tensor
            ref_time_indices_B_N: [B, max_ref] latent time indices for RoPE positioning
            main_H_latent: Main token's H dimension (for spatial shift offset calculation)
            main_W_latent: Main token's W dimension (for spatial shift offset calculation)

        Returns:
            Tuple of:
                - ref_tokens: [B, max_ref, H', W', D] embedded reference tokens (T_ref = max_ref)
                - ref_rope: [B, max_ref * H' * W', 1, 1, D] per-batch RoPE for reference tokens
        """
        B, max_ref, C, T, H, W = ref_latents_B_N_C_T_H_W.shape
        device = ref_latents_B_N_C_T_H_W.device
        dtype = ref_latents_B_N_C_T_H_W.dtype

        # Use ALL reference frames (including zero-padded ones) for consistent
        # tensor shapes across all ranks in distributed training.
        # Zero-padded frames will produce near-zero tokens after embedding.

        # Add padding mask channel (zeros for reference)
        padding_mask = torch.zeros(B, max_ref, 1, T, H, W, device=device, dtype=dtype)

        # Concatenate: [B, max_ref, C+1, 1, H, W]
        ref_with_mask = torch.cat([ref_latents_B_N_C_T_H_W, padding_mask], dim=2)

        # Reshape for patch embedding: [B * max_ref, C+1, 1, H, W]
        ref_reshaped = rearrange(ref_with_mask, "b n c t h w -> (b n) c t h w")

        # Cast to embedder weight dtype (handles float32 input with bfloat16 weights)
        embedder_dtype = self.ref_embedder.proj[1].weight.dtype
        ref_reshaped = ref_reshaped.to(dtype=embedder_dtype)

        # ========== DEBUG: Log ref_latents input to ref_embedder ==========
        log.info(f"[DEBUG REF] ref_latents input shape: {ref_reshaped.shape}")
        log.info(f"[DEBUG REF] ref_latents input norm: {ref_reshaped.norm().item():.6f}")
        log.info(f"[DEBUG REF] ref_latents input mean: {ref_reshaped.mean().item():.8f}")
        log.info(f"[DEBUG REF] ref_latents input min: {ref_reshaped.min().item():.8f}")
        log.info(f"[DEBUG REF] ref_latents input max: {ref_reshaped.max().item():.8f}")
        is_zero_input = ref_reshaped.abs().max().item() < 1e-6
        log.info(f"[DEBUG REF] ref_latents is all zeros: {is_zero_input}")
        # ========== END DEBUG ==========

        # Apply reference embedder: [B * max_ref, 1, H', W', D]
        ref_embedded = self.ref_embedder(ref_reshaped)

        # ========== DEBUG: Log ref_embedded output ==========
        log.info(f"[DEBUG REF] ref_embedded output shape: {ref_embedded.shape}")
        log.info(f"[DEBUG REF] ref_embedded output norm: {ref_embedded.norm().item():.6f}")
        log.info(f"[DEBUG REF] ref_embedded output mean: {ref_embedded.mean().item():.8f}")
        log.info(f"[DEBUG REF] ref_embedded output min: {ref_embedded.min().item():.8f}")
        log.info(f"[DEBUG REF] ref_embedded output max: {ref_embedded.max().item():.8f}")
        is_zero_output = ref_embedded.abs().max().item() < 1e-6
        log.info(f"[DEBUG REF] ref_embedded is all zeros: {is_zero_output}")
        # ========== END DEBUG ==========

        # Get latent spatial dimensions
        _, T_lat, H_lat, W_lat, D = ref_embedded.shape
        assert T_lat == 1, f"Expected T_lat=1 for single-frame ref, got {T_lat}"

        # Reshape to [B, max_ref, H', W', D] for temporal concat with main tokens
        # Each ref frame becomes one "time step" in the temporal dimension
        ref_tokens = rearrange(
            ref_embedded,
            "(b n) t h w d -> b (n t) h w d",  # n*t becomes temporal dim (t=1, so just n)
            b=B,
            n=max_ref
        )

        # Generate RoPE for ALL reference frames (max_ref) with dynamic time indices
        ref_rope = self._generate_ref_rope_embedding(
            max_ref, H_lat, W_lat, fps, ref_time_indices_B_N,
        )

        return ref_tokens, ref_rope

    def _generate_ref_rope_embedding(
        self,
        num_ref_frames: int,
        H_latent: int,
        W_latent: int,
        fps: Optional[torch.Tensor] = None,
        ref_time_indices_B_N: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate per-batch RoPE embeddings for reference frames using Time Offset Mode.

        Time Offset Mode:
        - Time axis: ref_time_offset + ref_time_indices * ref_time_stride
        - Spatial axis: No offset [0:H, 0:W]

        Each batch item gets its own RoPE based on its ref_time_indices,
        so different samples in a batch correctly receive different RoPE positions.

        Args:
            num_ref_frames: Number of reference frames
            H_latent: Latent height (after patchify)
            W_latent: Latent width (after patchify)
            fps: FPS tensor (optional)
            ref_time_indices_B_N: [B, max_ref] latent time indices from dataset (0-23 range).

        Returns:
            Per-batch RoPE embedding for reference frames [B, num_ref * H * W, 1, 1, D]
        """
        if num_ref_frames == 0:
            return None

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

        # Spatial indices (same for all batch items)
        ref_h_indices = torch.arange(H_latent, device=device, dtype=dtype)
        ref_w_indices = torch.arange(W_latent, device=device, dtype=dtype)
        half_emb_h = torch.outer(ref_h_indices, h_spatial_freqs)
        half_emb_w = torch.outer(ref_w_indices, w_spatial_freqs)

        # Generate per-batch RoPE
        if ref_time_indices_B_N is not None:
            B = ref_time_indices_B_N.shape[0]
        else:
            B = 1

        all_ropes = []
        for b in range(B):
            # TIME OFFSET MODE per batch item
            if ref_time_indices_B_N is not None:
                latent_indices = ref_time_indices_B_N[b, :num_ref_frames].to(device=device, dtype=dtype)
                ref_rope_times = self.ref_time_offset + latent_indices * self.ref_time_stride
            else:
                ref_rope_times = torch.tensor(
                    [self.ref_time_offset + i * self.ref_time_stride for i in range(num_ref_frames)],
                    device=device,
                    dtype=dtype,
                )

            half_emb_t = torch.outer(ref_rope_times, temporal_freqs)

            em_T_H_W_D = torch.cat(
                [
                    repeat(half_emb_t, "t d -> t h w d", h=H_latent, w=W_latent),
                    repeat(half_emb_h, "h d -> t h w d", t=num_ref_frames, w=W_latent),
                    repeat(half_emb_w, "w d -> t h w d", t=num_ref_frames, h=H_latent),
                ] * 2,
                dim=-1,
            )

            # [num_ref*H*W, 1, 1, D]
            rope_b = rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()
            all_ropes.append(rope_b)

        # Stack to [B, num_ref*H*W, 1, 1, D]
        return torch.stack(all_ropes, dim=0)

    def unpatchify(self, x_B_T_H_W_M):
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

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
        # Reference inputs (NEW)
        ref_latents_B_N_C_T_H_W: Optional[torch.Tensor] = None,
        ref_mask_B_N: Optional[torch.Tensor] = None,
        ref_time_indices_B_N: Optional[torch.Tensor] = None,
        # Camera parameters (CAMERA SPECIFIC)
        main_extrinsics_B_T_3_4: Optional[torch.Tensor] = None,
        main_intrinsics_B_T_3_3: Optional[torch.Tensor] = None,
        ref_extrinsics_B_N_3_4: Optional[torch.Tensor] = None,
        ref_intrinsics_B_N_3_3: Optional[torch.Tensor] = None,
        cam_mask_B: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with TIC-FT style reference token concatenation.

        Reference tokens are concatenated in temporal dimension ONCE at input,
        processed identically through all blocks, and split at output.
        RoPE time indices are computed from ref_time_indices: offset + indices * stride.

        Args:
            x_B_C_T_H_W: Main video latents (with channel-concatenated warp) [B, C, T, H, W]
            timesteps_B_T: Diffusion timesteps
            crossattn_emb: Text embeddings
            fps: Frame rate
            padding_mask: Padding mask
            data_type: Data type (VIDEO or IMAGE)
            intermediate_feature_ids: IDs for intermediate features
            img_context_emb: Image context embeddings
            ref_latents_B_N_C_T_H_W: Reference latents [B, max_ref, 16, 1, H, W] (NEW)
            ref_mask_B_N: Reference mask [B, max_ref] (NEW)
            ref_time_indices_B_N: [B, max_ref] latent time indices for dynamic RoPE (NEW)

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
        T_ref = 0

        # Inject camera embedding into main tokens (CAMERA SPECIFIC)
        if self.use_camera_embedding and main_extrinsics_B_T_3_4 is not None:
            # One-time verification: log CameraNet weight stats after checkpoint loading
            if not self._camera_net_verified:
                self._camera_net_verified = True
                try:
                    fp_w = self.camera_net.final_proj.weight
                    fp_b = self.camera_net.final_proj.bias
                    # Handle DTensor (FSDP)
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
            )  # [B, T_pixel, 6, H_pixel, W_pixel]
            main_cam_emb = self.camera_net(main_raymap.to(dtype=x_B_T_H_W_D.dtype))  # [B, T_latent, H, W, D]

            # One-time: log raymap and embedding stats
            if not getattr(self, '_camera_raymap_logged', False):
                self._camera_raymap_logged = True
                log.info(
                    f"[CameraNet forward] main_extrinsics: {main_extrinsics_B_T_3_4.shape}, "
                    f"main_raymap: {main_raymap.shape}, main_cam_emb: {main_cam_emb.shape}"
                )
                log.info(
                    f"[CameraNet forward] raymap stats: min={main_raymap.min().item():.4f}, "
                    f"max={main_raymap.max().item():.4f}, mean={main_raymap.mean().item():.6f}, "
                    f"std={main_raymap.std().item():.4f}"
                )
                log.info(
                    f"[CameraNet forward] cam_emb stats: min={main_cam_emb.min().item():.6f}, "
                    f"max={main_cam_emb.max().item():.6f}, "
                    f"norm={main_cam_emb.norm().item():.6f}"
                )
                # Verify: frame0 extrinsic should be near-identity (relative coords)
                ext0 = main_extrinsics_B_T_3_4[0, 0]  # [3, 4]
                R0 = ext0[:3, :3]
                t0 = ext0[:3, 3]
                log.info(
                    f"[CameraNet forward] frame0 R diag=[{R0[0,0]:.4f}, {R0[1,1]:.4f}, {R0[2,2]:.4f}], "
                    f"t=[{t0[0]:.4f}, {t0[1]:.4f}, {t0[2]:.4f}] (expect ~identity for relative coords)"
                )

            # Apply camera dropout mask
            if cam_mask_B is not None:
                main_cam_emb = main_cam_emb * cam_mask_B[:, None, None, None, None]

            x_B_T_H_W_D = x_B_T_H_W_D + main_cam_emb

        # Prepare reference tokens (always use all max_ref_frames for FSDP consistency)
        if ref_latents_B_N_C_T_H_W is not None and ref_mask_B_N is not None:
            ref_tokens, ref_rope = self.prepare_ref_embedded_sequence(
                ref_latents_B_N_C_T_H_W,
                ref_mask_B_N,
                fps=fps,
                ref_time_indices_B_N=ref_time_indices_B_N,
                main_H_latent=H,  # Pass main token's spatial dims for spatial shift calculation
                main_W_latent=W,
            )
            T_ref = ref_tokens.shape[1]  # max_ref frames

            # Inject camera embedding into ref tokens (CAMERA SPECIFIC)
            if self.use_camera_embedding and ref_extrinsics_B_N_3_4 is not None:
                from cosmos_predict2._src.predict2.networks.camera_net import generate_raymap
                B_ref, N_ref = ref_extrinsics_B_N_3_4.shape[:2]
                # Process each ref as T=1 video: [B*N, 1, 3, 4] / [B*N, 1, 3, 3]
                ref_raymap = generate_raymap(
                    ref_extrinsics_B_N_3_4.reshape(B_ref * N_ref, 1, 3, 4),
                    ref_intrinsics_B_N_3_3.reshape(B_ref * N_ref, 1, 3, 3),
                    self.video_size,
                )  # [B*N, 1, 6, H_pixel, W_pixel]
                ref_cam_emb = self.camera_net(ref_raymap.to(dtype=ref_tokens.dtype))  # [B*N, 1, H, W, D]
                ref_cam_emb = ref_cam_emb.reshape(B_ref, N_ref, H, W, D)
                # Zero out invalid ref camera embeddings
                ref_cam_emb = ref_cam_emb * ref_mask_B_N[:, :, None, None, None]
                # Apply camera dropout mask
                if cam_mask_B is not None:
                    ref_cam_emb = ref_cam_emb * cam_mask_B[:, None, None, None, None]
                ref_tokens = ref_tokens + ref_cam_emb

            # TIC-FT STYLE: TEMPORAL CONCATENATION AT INPUT
            x_B_T_H_W_D = torch.cat([x_B_T_H_W_D, ref_tokens], dim=1)

            # Per-batch RoPE: expand main_rope [L, 1, 1, D] -> [B, L, 1, 1, D],
            # then cat with ref_rope [B, L_ref, 1, 1, D] on dim=1
            rope_emb_L_1_1_D = rope_emb_L_1_1_D.unsqueeze(0).expand(B, -1, -1, -1, -1)
            rope_emb_L_1_1_D = torch.cat([rope_emb_L_1_1_D, ref_rope], dim=1)

            # Extend extra_pos_emb if present
            if extra_pos_emb is not None:
                # Zero-pad for ref frames (they don't need extra positional embedding)
                ref_extra_pos = torch.zeros(B, T_ref, H, W, D, device=extra_pos_emb.device, dtype=extra_pos_emb.dtype)
                extra_pos_emb = torch.cat([extra_pos_emb, ref_extra_pos], dim=1)

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

        # Store for logging
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
                # Only extract main tokens for intermediate features
                x_main = x_B_T_H_W_D[:, :T_main, :, :, :]
                x_reshaped = rearrange(x_main, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped)

        # Extract main tokens at output (discard ref tokens)
        x_main_B_T_H_W_D = x_B_T_H_W_D[:, :T_main, :, :, :]

        x_B_T_H_W_O = self.final_layer(x_main_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        if intermediate_feature_ids:
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp

    def enable_selective_checkpoint(self, sac_config: SACConfig, blocks: nn.ModuleList):
        if sac_config.mode == CheckpointMode.NONE:
            return self

        log.info(
            f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks"
        )
        _context_fn = sac_config.get_context_fn()
        for block_id, block in blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(
                    block,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                )
                blocks.register_module(block_id, block)
        self.register_module(
            "final_layer",
            ptd_checkpoint_wrapper(
                self.final_layer,
                context_fn=_context_fn,
                preserve_rng_state=False,
            ),
        )
        return self

    def fully_shard(self, mesh):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False)
        if self.use_camera_embedding:
            fully_shard(self.camera_net, mesh=mesh, reshard_after_forward=True)

    def disable_context_parallel(self):
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled
