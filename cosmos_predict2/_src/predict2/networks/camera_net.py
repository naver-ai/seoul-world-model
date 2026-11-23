# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Camera-pose conditioning module.

Independent NAVER implementation, motivated by the camera-conditioning design of
Hunyuan-GameCraft (Plucker raymap -> pixel-unshuffle -> 1x1 conv encoder -> patch
projection). No Hunyuan-GameCraft source code is included. The module converts camera
extrinsics and intrinsics into Plucker ray features, then projects those features to
the token layout consumed by the video DiT.
"""

from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn as nn

from cosmos_predict2._src.imaginaire.modules.camera import Camera


def _project_to_rotation(matrix: torch.Tensor) -> torch.Tensor:
    """Return the nearest proper rotation matrix for each input matrix."""
    left, _, right_t = torch.linalg.svd(matrix.double())
    rotation = left @ right_t
    orientation = torch.linalg.det(rotation)
    left = left.clone()
    left[..., :, -1] *= orientation.unsqueeze(-1)
    return (left @ right_t).to(dtype=matrix.dtype)


def generate_raymap(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Build a [B, T, 6, H, W] Plucker raymap from camera matrices."""
    if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsics must have shape [B, T, 3, 4], got {tuple(extrinsics.shape)}")
    if intrinsics.shape[:2] != extrinsics.shape[:2] or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must have shape [B, T, 3, 3], got {tuple(intrinsics.shape)}")

    batch, frames = extrinsics.shape[:2]
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    # fp32 like training: inference runs under a bf16 autocast
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        poses = extrinsics.reshape(-1, 3, 4).float()
        rotations = _project_to_rotation(poses[:, :, :3])
        poses = torch.cat((rotations, poses[:, :, 3:]), dim=-1)
        calibrations = intrinsics.reshape(-1, 3, 3).float()
        rays = Camera.get_plucker_rays(poses, calibrations, (height, width))

    return rays.reshape(batch, frames, height, width, 6).permute(0, 1, 4, 2, 3).contiguous()


class CameraNet(nn.Module):
    """Encode Plucker raymaps as camera-conditioning tokens."""

    def __init__(
        self,
        in_channels: int = 6,
        downscale_coef: int = 8,
        out_channels: int = 16,
        patch_size: Sequence[int] | None = None,
        hidden_size: int = 2048,
    ) -> None:
        super().__init__()
        if downscale_coef <= 0:
            raise ValueError("downscale_coef must be positive")

        patch = tuple(patch_size or (1, 2, 2))
        if len(patch) != 3 or any(value <= 0 for value in patch):
            raise ValueError(f"patch_size must contain three positive integers, got {patch}")

        self.downscale_coef = downscale_coef
        expanded_channels = in_channels * downscale_coef**2
        first_channels = expanded_channels // 2
        second_channels = first_channels // 2

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_coef)
        self.encode_first = nn.Sequential(
            nn.Conv2d(expanded_channels, first_channels, kernel_size=1),
            nn.GroupNorm(2, first_channels),
            nn.ReLU(inplace=True),
        )
        self.encode_second = nn.Sequential(
            nn.Conv2d(first_channels, second_channels, kernel_size=1),
            nn.GroupNorm(2, second_channels),
            nn.ReLU(inplace=True),
        )
        self.final_proj = nn.Conv2d(second_channels, out_channels, kernel_size=1)
        self.camera_proj = nn.Conv3d(
            out_channels,
            hidden_size,
            kernel_size=patch,
            stride=patch,
        )
        self.scale = nn.Parameter(torch.ones(1))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize a neutral camera branch suitable for pretrained DiTs."""
        for encoder in (self.encode_first, self.encode_second):
            convolution = encoder[0]
            nn.init.kaiming_normal_(convolution.weight, mode="fan_out", nonlinearity="relu")
            if convolution.bias is not None:
                nn.init.zeros_(convolution.bias)

        nn.init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            nn.init.zeros_(self.final_proj.bias)

        nn.init.xavier_uniform_(self.camera_proj.weight)
        if self.camera_proj.bias is not None:
            nn.init.zeros_(self.camera_proj.bias)

    @staticmethod
    def _pairwise_temporal_mean(
        tensor: torch.Tensor,
        batch: int,
        frames: int,
    ) -> tuple[torch.Tensor, int]:
        """Average adjacent frames, carrying an unmatched final frame forward."""
        feature_shape = tensor.shape[1:]
        sequence = tensor.reshape(batch, frames, *feature_shape)
        paired_frames = frames // 2

        if paired_frames:
            paired = 0.5 * (
                sequence[:, : paired_frames * 2 : 2]
                + sequence[:, 1 : paired_frames * 2 : 2]
            )
        else:
            paired = sequence[:, :0]

        if frames % 2:
            paired = torch.cat((paired, sequence[:, -1:]), dim=1)

        output_frames = paired.shape[1]
        return paired.reshape(batch * output_frames, *feature_shape), output_frames

    def forward(self, raymap: torch.Tensor) -> torch.Tensor:
        """Return camera tokens with shape [B, T', H', W', hidden_size]."""
        if raymap.ndim != 5:
            raise ValueError(f"raymap must have shape [B, T, C, H, W], got {tuple(raymap.shape)}")

        batch, frames, channels, height, width = raymap.shape
        factor = self.downscale_coef
        if height % factor or width % factor:
            raise ValueError(
                f"raymap spatial size {(height, width)} must be divisible by downscale_coef={factor}"
            )

        features = self.pixel_unshuffle(raymap.reshape(batch * frames, channels, height, width))
        features = self.encode_first(features)
        features, frames = self._pairwise_temporal_mean(features, batch, frames)
        features = self.encode_second(features)
        features, frames = self._pairwise_temporal_mean(features, batch, frames)
        features = self.final_proj(features)

        reduced_height, reduced_width = height // factor, width // factor
        projected_channels = features.shape[1]
        features = features.reshape(
            batch,
            frames,
            projected_channels,
            reduced_height,
            reduced_width,
        ).permute(0, 2, 1, 3, 4)
        features = self.camera_proj(features)
        return features.permute(0, 2, 3, 4, 1) * self.scale
