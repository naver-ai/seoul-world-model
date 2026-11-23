# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Inference visualization callback for creating concatenated comparison videos.

Creates side-by-side comparison videos with (horizontal layout):
- Column 1: Generated video
- Column 2: Camera trajectory 2D visualization (bird's eye view, animated, ego-centric)
- Column 3: Ground truth video (stretched)
"""

import os
from typing import Optional

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from einops import rearrange

from cosmos_predict2._src.imaginaire.utils import distributed, log


def create_trajectory_frame(
    camera_positions: np.ndarray,
    current_frame_idx: int,
    gt_frame_indices: np.ndarray,
    frame_size: tuple[int, int] = (720, 360),  # H, W - narrower for horizontal layout
    padding: int = 50,
    ref_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a single frame of the camera trajectory visualization (ego-centric, stylized).

    The trajectory is displayed in ego-centric coordinates with modern styling:
    - Gradient colored path showing progress
    - Glowing current position indicator
    - Sleek dark theme with accent colors
    - Reference panorama positions (orange squares) at their actual locations

    Args:
        camera_positions: [T, 3] array of camera positions (x, y, z)
        current_frame_idx: Current frame index to highlight
        gt_frame_indices: [12] array of GT frame indices
        frame_size: (H, W) output frame size
        padding: Padding around the trajectory
        ref_positions: Optional [N, 3] array of reference panorama positions (x, y, z)

    Returns:
        RGB image array [H, W, 3] uint8
    """
    H, W = frame_size
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)

    # Use X-Z plane for bird's eye view (top-down)
    x_coords = camera_positions[:, 0]
    z_coords = camera_positions[:, 2]

    # Convert to ego-centric coordinates
    x_ego = x_coords - x_coords[0]
    z_ego = z_coords - z_coords[0]

    # Calculate bounds
    x_min, x_max = x_ego.min(), x_ego.max()
    z_min, z_max = z_ego.min(), z_ego.max()

    # Calculate ranges and use the longer side for square aspect
    x_range_raw = x_max - x_min
    z_range_raw = z_max - z_min
    max_range = max(x_range_raw, z_range_raw)
    margin = max(max_range * 0.15, 2.0)
    square_range = max(max_range + 2 * margin, 5.0)

    # Center both axes
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2
    ax.set_xlim(x_center - square_range / 2, x_center + square_range / 2)
    ax.set_ylim(z_center - square_range / 2, z_center + square_range / 2)

    # Dark background with subtle gradient effect
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Draw subtle grid
    ax.grid(True, color='#21262d', linestyle='-', linewidth=0.5, alpha=0.5)

    # Create gradient colored trajectory for traveled path
    if current_frame_idx > 0:
        points = np.array([x_ego[:current_frame_idx + 1], z_ego[:current_frame_idx + 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color gradient from cyan to magenta
        colors = np.linspace(0, 1, len(segments))
        cmap = LinearSegmentedColormap.from_list('trajectory', ['#00d4ff', '#7c3aed', '#ff0080'])

        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(colors)
        lc.set_linewidth(3)
        lc.set_alpha(0.9)
        ax.add_collection(lc)

    # Draw future trajectory as dotted line
    if current_frame_idx < len(x_ego) - 1:
        ax.plot(x_ego[current_frame_idx:], z_ego[current_frame_idx:],
                color='#3d4450', linewidth=2, linestyle='--', alpha=0.6)

    # Mark reference panorama positions with orange squares (if provided)
    if ref_positions is not None and len(ref_positions) > 0:
        # ref_positions: [N, 3] in same coordinate system as camera_positions
        # Convert to ego-centric (same as camera positions)
        ref_x_ego = ref_positions[:, 0] - camera_positions[0, 0]
        ref_z_ego = ref_positions[:, 2] - camera_positions[0, 2]

        # Outer glow for reference markers
        ax.scatter(ref_x_ego, ref_z_ego, c='#f97316', s=80, zorder=3, alpha=0.3,
                   marker='s', edgecolors='none')
        # Inner reference markers
        ax.scatter(ref_x_ego, ref_z_ego, c='#f97316', s=40, zorder=4, alpha=0.9,
                   marker='s', edgecolors='#ea580c', linewidths=1.5, label='Ref')

    # Mark start position with special marker
    ax.scatter([x_ego[0]], [z_ego[0]], c='#00d4ff', s=150, zorder=6,
               marker='o', edgecolors='#0891b2', linewidths=2, alpha=0.9, label='Start')

    # Current position with glowing effect
    current_x = x_ego[current_frame_idx]
    current_z = z_ego[current_frame_idx]

    # Outer glow rings
    for size, alpha in [(300, 0.1), (200, 0.2), (120, 0.3)]:
        ax.scatter([current_x], [current_z], c='#ff0080', s=size, zorder=6, alpha=alpha)

    # Core marker
    ax.scatter([current_x], [current_z], c='#ff0080', s=80, zorder=7,
               edgecolors='white', linewidths=2, label='Now')

    # Draw direction indicator (arrow)
    if current_frame_idx < len(camera_positions) - 1:
        look_ahead = min(5, len(x_ego) - current_frame_idx - 1)
        if look_ahead > 0:
            next_x = x_ego[current_frame_idx + look_ahead]
            next_z = z_ego[current_frame_idx + look_ahead]
            dx = next_x - current_x
            dz = next_z - current_z
            length = np.sqrt(dx**2 + dz**2)
            if length > 0.01:
                scale = 1.5 / length
                ax.annotate('', xy=(current_x + dx * scale, current_z + dz * scale),
                           xytext=(current_x, current_z),
                           arrowprops=dict(arrowstyle='->', color='#ff0080',
                                          lw=2, mutation_scale=15))

    # Styling
    ax.tick_params(colors='#8b949e', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(1)

    # Progress indicator
    progress = (current_frame_idx + 1) / len(camera_positions) * 100
    ax.set_title(f'Camera Path  {current_frame_idx + 1}/{len(camera_positions)}  ({progress:.0f}%)',
                 color='#e6edf3', fontsize=10, pad=8, fontweight='bold')

    # Minimal axis labels
    ax.set_xlabel('X', color='#8b949e', fontsize=9)
    ax.set_ylabel('Z', color='#8b949e', fontsize=9)

    # Compact legend
    legend = ax.legend(loc='upper right', fontsize=7, facecolor='#161b22',
                      edgecolor='#30363d', labelcolor='#e6edf3', framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0.5)

    # Convert to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]  # RGBA -> RGB

    plt.close(fig)

    # Resize to target size
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    return img


def create_trajectory_video(
    camera_positions: np.ndarray,
    gt_frame_indices: np.ndarray,
    num_frames: int,
    frame_size: tuple[int, int] = (720, 360),
    ref_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create the full camera trajectory visualization video.

    Args:
        camera_positions: [T, 3] array of camera positions
        gt_frame_indices: [12] array of GT frame indices
        num_frames: Number of frames in the output video
        frame_size: (H, W) output frame size
        ref_positions: Optional [N, 3] array of reference panorama positions

    Returns:
        Video array [T, H, W, 3] uint8
    """
    frames = []
    for i in range(num_frames):
        frame = create_trajectory_frame(
            camera_positions,
            current_frame_idx=i,
            gt_frame_indices=gt_frame_indices,
            frame_size=frame_size,
            ref_positions=ref_positions,
        )
        frames.append(frame)

    return np.stack(frames, axis=0)  # [T, H, W, 3]


def stretch_gt_frames_to_video(
    gt_frames: np.ndarray,
    gt_frame_indices: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    """Stretch 12 GT frames to match video length by holding each frame.

    Each GT frame is held until the next GT frame's index.

    Args:
        gt_frames: [12, H, W, 3] array of GT frames
        gt_frame_indices: [12] array of GT frame indices
        num_frames: Number of frames in the output video

    Returns:
        Video array [T, H, W, 3] uint8
    """
    output_frames = []

    for frame_idx in range(num_frames):
        gt_idx = np.searchsorted(gt_frame_indices, frame_idx, side='right') - 1
        gt_idx = max(0, min(gt_idx, len(gt_frames) - 1))
        output_frames.append(gt_frames[gt_idx])

    return np.stack(output_frames, axis=0)  # [T, H, W, 3]


def create_inference_visualization(
    generated_video: torch.Tensor,
    camera_positions: torch.Tensor,
    gt_frames: torch.Tensor,
    gt_frame_indices: torch.Tensor,
    output_path: str,
    fps: int = 10,
    ref_positions: Optional[torch.Tensor] = None,
) -> str:
    """Create concatenated inference visualization video (horizontal layout).

    Layout: [Generated Video | Camera Trajectory | GT Video]

    Args:
        generated_video: [C, T, H, W] or [T, H, W, C] generated video tensor (uint8 or float)
        camera_positions: [T, 3] camera positions
        gt_frames: [12, C, H, W] or [12, H, W, C] GT frames tensor
        gt_frame_indices: [12] GT frame indices
        output_path: Path to save the output video
        fps: Frames per second
        ref_positions: Optional [N, 3] tensor of reference panorama positions

    Returns:
        Path to the saved video
    """
    # Convert tensors to numpy (convert bfloat16 to float32 first for numpy compatibility)
    if isinstance(generated_video, torch.Tensor):
        generated_video = generated_video.float().cpu().numpy()
    if isinstance(camera_positions, torch.Tensor):
        camera_positions = camera_positions.float().cpu().numpy()
    if isinstance(gt_frames, torch.Tensor):
        gt_frames = gt_frames.float().cpu().numpy()
    if isinstance(gt_frame_indices, torch.Tensor):
        gt_frame_indices = gt_frame_indices.long().cpu().numpy()
    if ref_positions is not None and isinstance(ref_positions, torch.Tensor):
        ref_positions = ref_positions.float().cpu().numpy()

    # Handle different tensor formats
    # Generated video: expect [C, T, H, W] -> [T, H, W, C]
    if generated_video.ndim == 4:
        if generated_video.shape[0] == 3:  # [C, T, H, W]
            generated_video = rearrange(generated_video, 'c t h w -> t h w c')

    # GT frames: expect [12, C, H, W] -> [12, H, W, C]
    if gt_frames.ndim == 4:
        if gt_frames.shape[1] == 3:  # [12, C, H, W]
            gt_frames = rearrange(gt_frames, 'n c h w -> n h w c')

    # Normalize to uint8 if needed
    if generated_video.dtype == np.float32 or generated_video.dtype == np.float64:
        if generated_video.max() <= 1.0:
            generated_video = (generated_video * 255).clip(0, 255).astype(np.uint8)
        else:
            generated_video = generated_video.clip(0, 255).astype(np.uint8)

    if gt_frames.dtype == np.float32 or gt_frames.dtype == np.float64:
        if gt_frames.max() <= 1.0:
            gt_frames = (gt_frames * 255).clip(0, 255).astype(np.uint8)
        else:
            gt_frames = gt_frames.clip(0, 255).astype(np.uint8)

    num_frames, H, W, C = generated_video.shape

    # Trajectory width is smaller (about 1/3 of video width)
    traj_width = W // 3

    # Create trajectory visualization video (narrower)
    log.info(f"Creating trajectory visualization ({num_frames} frames)...")
    trajectory_video = create_trajectory_video(
        camera_positions,
        gt_frame_indices,
        num_frames=num_frames,
        frame_size=(H, traj_width),
        ref_positions=ref_positions,
    )

    # Stretch GT frames to video length
    log.info(f"Stretching GT frames to video length...")
    gt_video = stretch_gt_frames_to_video(
        gt_frames,
        gt_frame_indices,
        num_frames=num_frames,
    )

    # Concatenate horizontally: generated | trajectory | gt
    log.info(f"Concatenating videos horizontally...")
    concat_video = np.concatenate([generated_video, trajectory_video, gt_video], axis=2)

    # Save video using OpenCV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (concat_video.shape[2], concat_video.shape[1]))

    for frame in concat_video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    log.info(f"Saved inference visualization to: {output_path}")

    return output_path


def save_inference_comparison(
    generated_video: torch.Tensor,
    data_batch: dict,
    output_dir: str,
    sample_idx: int,
    guidance: float,
    fps: int = 10,
) -> Optional[str]:
    """Save inference comparison video if required data is available.

    Args:
        generated_video: [C, T, H, W] generated video tensor
        data_batch: Data batch dictionary containing camera_positions, gt_frames, gt_frame_indices
        output_dir: Output directory
        sample_idx: Sample index for filename
        guidance: Guidance value for filename
        fps: Frames per second

    Returns:
        Path to saved video or None if data not available
    """
    # Check if required fields are available
    required_fields = ['camera_positions', 'gt_frames', 'gt_frame_indices']
    for field in required_fields:
        if field not in data_batch:
            log.debug(f"Skipping inference visualization: {field} not in data_batch")
            return None

    camera_positions = data_batch['camera_positions']
    gt_frames = data_batch['gt_frames']
    gt_frame_indices = data_batch['gt_frame_indices']

    # Optional: reference positions (for custom dataset)
    ref_positions = data_batch.get('ref_positions', None)
    ref_frame_indices = data_batch.get('ref_frame_indices', None)

    # Handle batched data (take first item)
    if camera_positions.ndim == 3:  # [B, T, 3]
        camera_positions = camera_positions[0]
    if gt_frames.ndim == 5:  # [B, 12, C, H, W]
        gt_frames = gt_frames[0]
    if gt_frame_indices.ndim == 2:  # [B, 12]
        gt_frame_indices = gt_frame_indices[0]
    if ref_positions is not None and ref_positions.ndim == 3:  # [B, N, 3]
        ref_positions = ref_positions[0]
    if ref_frame_indices is not None and ref_frame_indices.ndim == 2:  # [B, N]
        ref_frame_indices = ref_frame_indices[0]

    # If ref_positions not provided but ref_frame_indices available,
    # compute ref_positions from camera_positions at those frame indices
    if ref_positions is None and ref_frame_indices is not None:
        if isinstance(ref_frame_indices, torch.Tensor):
            ref_frame_indices_np = ref_frame_indices.long().cpu().numpy()
        else:
            ref_frame_indices_np = ref_frame_indices
        if isinstance(camera_positions, torch.Tensor):
            camera_positions_np = camera_positions.float().cpu().numpy()
        else:
            camera_positions_np = camera_positions
        ref_positions = camera_positions_np[ref_frame_indices_np]
        ref_positions = torch.from_numpy(ref_positions).float()

    output_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_guidance_{guidance:.1f}_comparison.mp4")

    return create_inference_visualization(
        generated_video=generated_video,
        camera_positions=camera_positions,
        gt_frames=gt_frames,
        gt_frame_indices=gt_frame_indices,
        ref_positions=ref_positions,
        output_path=output_path,
        fps=fps,
    )
