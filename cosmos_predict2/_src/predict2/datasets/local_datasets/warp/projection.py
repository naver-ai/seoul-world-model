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

"""3D projection utility functions for warping operations."""

import torch


def unproject_points_warp(
    depth: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Unproject depth map to 3D world points."""
    b, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype

    x1d = torch.arange(0, w, device=device, dtype=dtype)[None]
    y1d = torch.arange(0, h, device=device, dtype=dtype)[:, None]
    x2d = x1d.repeat([h, 1])
    y2d = y1d.repeat([1, w])
    ones_2d = torch.ones(size=(h, w), device=device, dtype=dtype)
    pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]

    intrinsic_inv = torch.linalg.inv(intrinsic.float()).to(dtype)
    c2w = torch.linalg.inv(w2c.float()).to(dtype)

    intrinsic_inv_4d = intrinsic_inv[:, None, None]
    depth_4d = depth[:, 0][:, :, :, None, None]
    ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])

    unnormalized_pos = torch.matmul(intrinsic_inv_4d, pos_vectors_homo)
    cam_points = depth_4d * unnormalized_pos

    cam_points_homo = torch.cat([cam_points, ones_4d], dim=3)
    c2w_4d = c2w[:, None, None]
    world_points_homo = torch.matmul(c2w_4d, cam_points_homo)
    world_points = world_points_homo[:, :, :, :3, 0]

    return world_points


def project_points_warp(
    world_points: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Project 3D world points to 2D pixel coordinates."""
    b, h, w, _ = world_points.shape
    device = world_points.device
    dtype = world_points.dtype

    world_points = world_points.unsqueeze(-1)
    ones_4d = torch.ones((b, h, w, 1, 1), device=device, dtype=dtype)
    world_points_homo = torch.cat([world_points, ones_4d], dim=3)

    w2c_4d = w2c[:, None, None]
    cam_points_homo = torch.matmul(w2c_4d, world_points_homo)
    cam_points = cam_points_homo[:, :, :, :3]

    intrinsic_4d = intrinsic[:, None, None]
    projected_points = torch.matmul(intrinsic_4d, cam_points)

    return projected_points
