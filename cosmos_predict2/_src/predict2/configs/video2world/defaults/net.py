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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit_warp_ref_LA import MinimalV1LVGDiTWarpRefLA
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

# 2B base DiT (used by the base reason_embeddings experiment chain that the
# fine-tune inherits from).
COSMOS_V1_2B_NET_MININET: LazyDict = L(MinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)

# 2B Warp+Ref+LA Network (registered name: cosmos_v1_2B_warp_ref_LA)
COSMOS_V1_2B_NET_WARP_REF_LA: LazyDict = L(MinimalV1LVGDiTWarpRefLA)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
    max_ref_frames=5,
    ref_time_offset=50,
    ref_time_stride=5,
)

# 2B Warp+Ref+LA+Camera Network (registered name: cosmos_v1_2B_warp_ref_LA_cam)
COSMOS_V1_2B_NET_WARP_REF_LA_CAM = copy.deepcopy(COSMOS_V1_2B_NET_WARP_REF_LA)
COSMOS_V1_2B_NET_WARP_REF_LA_CAM.camera_embedding = True


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B_warp_ref_LA", node=COSMOS_V1_2B_NET_WARP_REF_LA)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B_warp_ref_LA_cam", node=COSMOS_V1_2B_NET_WARP_REF_LA_CAM)
