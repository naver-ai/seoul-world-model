# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/nv-tlabs/GEN3C
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

"""Bilinear splatting and forward warping functions."""

import torch

from .projection import project_points_warp


def bilinear_splatting_warp(
    frame: torch.Tensor,
    mask: torch.Tensor,
    depth: torch.Tensor,
    flow: torch.Tensor,
    depth_weight_scale: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear splatting for forward warping."""
    b, c, h, w = frame.shape
    device = frame.device
    dtype = frame.dtype

    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    grid = torch.cat([x, y], dim=1)

    trans_pos = flow + grid
    trans_pos_offset = trans_pos + 1
    trans_pos_floor = torch.floor(trans_pos_offset).long()
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()

    trans_pos_offset = torch.stack([
        torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
        torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)
    ], dim=1)
    trans_pos_floor = torch.stack([
        torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
        torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)
    ], dim=1)
    trans_pos_ceil = torch.stack([
        torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
        torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)
    ], dim=1)

    prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                     (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                     (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                     (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
    prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                     (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

    clamped_depth = torch.clamp(depth, min=0)
    log_depth = torch.log1p(clamped_depth)
    exponent = log_depth / (log_depth.max() + 1e-7) * depth_weight_scale
    exponent = torch.clamp(exponent, max=80.0)
    depth_weights = torch.exp(exponent) + 1e-7

    weight_nw = torch.moveaxis(prox_weight_nw * mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_sw = torch.moveaxis(prox_weight_sw * mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_ne = torch.moveaxis(prox_weight_ne * mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_se = torch.moveaxis(prox_weight_se * mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

    warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=dtype, device=device)
    warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=dtype, device=device)

    frame_cl = torch.moveaxis(frame, [0, 1, 2, 3], [0, 3, 1, 2])
    batch_indices = torch.arange(b, device=device, dtype=torch.long)[:, None, None]

    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), frame_cl * weight_nw, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), frame_cl * weight_sw, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), frame_cl * weight_ne, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), frame_cl * weight_se, accumulate=True
    )

    warped_weights.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), weight_nw, accumulate=True
    )
    warped_weights.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), weight_sw, accumulate=True
    )
    warped_weights.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), weight_ne, accumulate=True
    )
    warped_weights.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), weight_se, accumulate=True
    )

    warped_frame = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
    warped_weights = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
    warped_frame = warped_frame[:, :, 1:-1, 1:-1]
    warped_weights = warped_weights[:, :, 1:-1, 1:-1]
    warped_weights = torch.nan_to_num(warped_weights, nan=1000.0)

    valid_mask = warped_weights > 0
    warped_frame = torch.where(valid_mask, warped_frame / warped_weights, torch.tensor(0.0, device=device, dtype=dtype))
    warped_frame = torch.clamp(warped_frame, min=0, max=1)

    return warped_frame, valid_mask.float()


def forward_warp_frame(
    frame: torch.Tensor,
    world_points: torch.Tensor,
    target_w2c: torch.Tensor,
    target_intrinsic: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward warp frame using world points to target view."""
    b, c, h, w = frame.shape
    device = frame.device
    dtype = frame.dtype

    if mask is None:
        mask = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)

    projected = project_points_warp(world_points, target_w2c, target_intrinsic)

    trans_depth = projected[:, :, :, 2:3, 0]
    trans_coords = projected[:, :, :, :2, 0] / (trans_depth + 1e-7)
    trans_depth = trans_depth.permute(0, 3, 1, 2)

    mask = mask * (trans_depth > 0).float()

    trans_coords = trans_coords.permute(0, 3, 1, 2)

    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    grid = torch.cat([x, y], dim=1)

    flow = trans_coords - grid

    warped_frame, warped_mask = bilinear_splatting_warp(frame, mask, trans_depth, flow)

    return warped_frame, warped_mask
