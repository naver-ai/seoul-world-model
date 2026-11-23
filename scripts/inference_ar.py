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

"""Autoregressive (AR) inference script for multi-chunk video generation.

Generates long videos by chaining multiple 77-frame chunks:
- Chunk 0: 1 conditional frame (reference image) + 76 generated
- Chunk 1+: 5 conditional frames (last 5 from prev generation) + 72 generated

The overlap of 5 pixel frames (= 2 latent frames) provides temporal continuity.

This script loads the model/checkpoint directly (no trainer) and runs its own
AR generation loop.

Usage:
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12600 \\
      -m scripts.inference_ar \\
      --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \\
      -- experiment=predict2_generate \\
      checkpoint.load_path=/path/to/model.pt

Prefer the wrapper script: `DATA_DIR=... CKPT=... bash sh/inference/generate.sh`.
"""

import argparse
import importlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import cv2
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from loguru import logger as logging

from cosmos_predict2._src.imaginaire.config import pretty_print_overrides
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.utils.guardrail import Guardrails
from cosmos_oss.init import init_environment, is_rank0


_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, "
    "poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, "
    "choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, "
    "outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, "
    "visual noise, and flickering. Overall, the video is of poor quality."
)


def collate_single(sample: dict) -> dict:
    """Add batch dimension to a single sample dict."""
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        elif isinstance(v, (int, float)):
            batch[k] = v
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def _forward_warp_from_sample(sample: dict):
    """Reproduce the model's forward-warp condition from a sample's ingredients
    ([-1,1] src frames + depth + poses), so it can be visualized/saved. Returns a
    [C, T, H, W] uint8 tensor, or None if the sample is not in forward-warp mode."""
    if "src_frames" not in sample:
        return None
    from cosmos_predict2._src.predict2.models.gen3c_warping.forward_warp_utils_pytorch import forward_warp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sf = sample["src_frames"].float()      # [T,3,H,W] in [-1,1]
    sd = sample["src_depths"].float()      # [T,1,H,W]
    sw = sample["src_w2cs"].float()        # [T,4,4]
    si = sample["src_intrinsics"].float()  # [T,3,3]
    tw = sample["target_w2cs"].float()     # [T,4,4]
    T, C, H, W = sf.shape
    outs = []
    for i in range(0, T, 32):
        j = min(i + 32, T)
        w, _, _, _ = forward_warp(
            frame1=sf[i:j].to(dev), mask1=torch.ones(j - i, 1, H, W, device=dev),
            depth1=sd[i:j].to(dev), transformation1=sw[i:j].to(dev), transformation2=tw[i:j].to(dev),
            intrinsic1=si[i:j].to(dev), intrinsic2=si[i:j].to(dev), is_image=True, is_depth=True)
        outs.append(w.cpu())
    warped = torch.cat(outs)  # [T,3,H,W] in [-1,1]
    u8 = ((warped + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)  # [T,3,H,W]
    return u8.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]


@torch.no_grad()
def generate_chunk(model, data_batch, guidance, num_steps, seed, overlap, chunk_idx,
                   prev_generated_video=None, la_mode="normal", first_cond_frame=None,
                   drop_ref=False, drop_cam=False):
    """Generate a single AR chunk.

    Args:
        model: The loaded model.
        data_batch: Batched data dict on GPU.
        guidance: CFG guidance scale.
        num_steps: Number of sampling steps.
        seed: Random seed (offset by chunk_idx internally).
        overlap: Number of overlap frames from previous chunk.
        chunk_idx: Current chunk index.
        prev_generated_video: [1, C, T, H, W] float [-1,1] from previous chunk, or None.
        la_mode: LA variant — "normal", "off", "sink-first", "sink-retrieved".
            sink-first/sink-retrieved: replace first conditional frame with LA image (chunk 1+),
            disable separate LA conditioning. No RoPE duplication.
        first_cond_frame: [1, C, H, W] uint8 from chunk 0's first GT frame (for sink-first).
        drop_ref: If True, zero out ref_frames and ref_mask to disable ref conditioning.
        drop_cam: If True, zero out cam_mask to disable all camera embeddings.

    Returns:
        generated_video: [1, C, T, H, W] float [-1, 1]
    """
    from einops import repeat

    # --- Save LA image for sink modes (before zeroing) ---
    _sink_la_img = None
    if la_mode == "sink-first" and chunk_idx > 0 and first_cond_frame is not None:
        _sink_la_img = first_cond_frame.clone()  # [1, C, H, W] uint8
    elif la_mode == "sink-retrieved" and chunk_idx > 0:
        _sink_la_img = data_batch["la_frame"][:, 0, :, :, :].clone()  # [B, C, H, W] uint8

    # --- LA mode handling ---
    if la_mode == "off":
        data_batch["la_frame"] = torch.zeros_like(data_batch["la_frame"])
        data_batch["la_mask"] = torch.zeros_like(data_batch["la_mask"])
        logging.info("  LA mode=off: zeroed la_frame and la_mask")

    elif la_mode in ("sink-first", "sink-retrieved"):
        # Disable separate LA conditioning for all chunks
        data_batch["la_frame"] = torch.zeros_like(data_batch["la_frame"])
        data_batch["la_mask"] = torch.zeros_like(data_batch["la_mask"])
        if chunk_idx == 0:
            logging.info(f"  LA mode={la_mode}: chunk 0 — no LA")
        else:
            logging.info(f"  LA mode={la_mode}: chunk {chunk_idx} — will replace first cond frame after overlap injection")

    if drop_ref:
        data_batch["ref_frames"] = torch.zeros_like(data_batch["ref_frames"])
        data_batch["ref_mask"] = torch.zeros_like(data_batch["ref_mask"])
        logging.info("  Ref dropout: zeroed ref_frames and ref_mask at tensor level (pre-VAE)")
    if drop_cam:
        data_batch["cam_mask"] = torch.zeros_like(data_batch["cam_mask"])
        logging.info("  Cam dropout: zeroed cam_mask (all camera embeddings disabled)")

    # For chunk 1+: inject previous generation's last `overlap` frames
    if chunk_idx > 0 and prev_generated_video is not None:
        prev_frames = prev_generated_video[0, :, -overlap:, :, :]  # [C, overlap, H, W]
        prev_frames_uint8 = ((prev_frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        data_batch["video"][0, :, :overlap, :, :] = prev_frames_uint8
        logging.info(f"  Injected {overlap} frames from previous chunk into video[:, :, :{overlap}]")

    # For sink modes chunk 1+: replace first conditional frame with LA image
    # Must be AFTER overlap injection so it overwrites the injected frame 0
    if _sink_la_img is not None:
        data_batch["video"][0, :, 0, :, :] = _sink_la_img[0]
        logging.info(f"  LA mode={la_mode}: replaced video frame 0 with LA image")

    # Convert num_conditional_frames from pixel space to latent space.
    # The dataset provides pixel-space counts (e.g. 5 pixel overlap frames),
    # but set_video_condition operates on latent x0, so it needs latent-space counts.
    # Without this conversion, 5 pixel frames would be interpreted as 5 LATENT frames
    # (~20 pixel frames), causing the model to reproduce the warped video for ~2 seconds.
    if "num_conditional_frames" in data_batch:
        pixel_cond = data_batch["num_conditional_frames"]
        latent_cond = model.tokenizer.get_latent_num_frames(pixel_cond)
        data_batch["num_conditional_frames"] = latent_cond
        logging.info(f"  num_conditional_frames: {pixel_cond} pixel → {latent_cond} latent")

    # Compute text embeddings
    text_encoder_config = getattr(model.config, "text_encoder_config", None)
    if text_encoder_config is not None and text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(
            data_batch, model.input_caption_key
        )
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(
            text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
        )

    # Negative prompt
    negative_prompt_emb = model.text_encoder.compute_text_embeddings_online(
        data_batch={"ai_caption": [_DEFAULT_NEGATIVE_PROMPT], "images": None},
        input_caption_key="ai_caption",
    )
    B = data_batch["t5_text_embeddings"].shape[0]
    if negative_prompt_emb.shape != data_batch["t5_text_embeddings"].shape:
        data_batch["neg_t5_text_embeddings"] = misc.to(
            repeat(negative_prompt_emb, "... -> b ...", b=B), **model.tensor_kwargs,
        )
    else:
        data_batch["neg_t5_text_embeddings"] = misc.to(
            negative_prompt_emb, **model.tensor_kwargs
        )
    data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]

    # Compute state_shape from raw video dimensions (before normalization).
    # generate_samples_from_batch calls _normalize/_augment internally,
    # so we must NOT call them here to avoid double normalization.
    _T, _H, _W = data_batch["video"].shape[-3:]
    state_shape = [
        model.config.state_ch,
        model.tokenizer.get_latent_num_frames(_T),
        _H // model.tokenizer.spatial_compression_factor,
        _W // model.tokenizer.spatial_compression_factor,
    ]

    sample_seed = seed + chunk_idx
    # autocast needed: the model runs outside any trainer precision context
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        sample_latent = model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            state_shape=state_shape,
            n_sample=B,
            num_steps=num_steps,
            is_negative_prompt=True,
            seed=sample_seed,
            no_la=False,
            no_ref_images=False,
        )

    logging.info(
        f"  Latent: shape={list(sample_latent.shape)}, "
        f"range=[{sample_latent.min().item():.3f}, {sample_latent.max().item():.3f}]"
    )

    generated_video = model.decode(sample_latent)  # [1, C, T, H, W] float [-1, 1]
    logging.info(
        f"  Decoded: shape={list(generated_video.shape)}, "
        f"range=[{generated_video.min().item():.3f}, {generated_video.max().item():.3f}]"
    )

    return generated_video


# ============================================================================
# Visualization helpers
# ============================================================================

def build_map_video(dataset, chunk_metadata_list, total_frames, video_height=704):
    """Build per-frame 2D bird's-eye map video.

    Shows trajectory, current position, panorama markers colored by role
    (ref / LA / both / inactive), connection lines, and frame/chunk indicators.

    Args:
        dataset: CustomDatasetARLACam instance.
        chunk_metadata_list: List of dicts from dataset.get_chunk_viz_metadata().
        total_frames: Total frames in the final concatenated video.
        video_height: Map canvas size (square).

    Yields:
        numpy [video_height, video_height, 3] uint8 frames (RGB), one at a time.
    """
    import cv2
    from collections import Counter

    S = video_height
    track_2d, pano_2d = dataset.get_all_positions_2d()

    # World-to-pixel mapping
    all_pts = np.concatenate(
        [track_2d] + [p.reshape(1, 2) for p in pano_2d.values()], axis=0
    )
    center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
    span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max(), 1e-6) * 1.35

    def to_px(pt):
        n = (pt - center) / span + 0.5
        return int(n[0] * (S - 1)), int((1 - n[1]) * (S - 1))

    track_px = [to_px(p) for p in track_2d]
    pano_px = {idx: to_px(p) for idx, p in pano_2d.items()}

    T_chunk = dataset.num_main_frames
    stride = dataset.stride

    def chunk_for_frame(f):
        return 0 if f < T_chunk else 1 + (f - T_chunk) // stride

    # -- colour palette (RGB) --
    BG_COLOR = (18, 18, 32)
    GRID_COLOR = (32, 32, 48)
    TRACK_DIM = (48, 48, 58)
    TRACK_VISITED = (195, 195, 210)
    CHUNK_COLOR = (0, 200, 80)
    REF_COLOR = (80, 170, 255)
    LA_COLOR = (255, 160, 50)
    BOTH_COLOR = (195, 120, 255)
    INACTIVE_COLOR = (55, 55, 65)
    CUR_COLOR = (20, 255, 140)
    TEXT_HI = (240, 240, 245)
    TEXT_LO = (155, 155, 165)

    def _blend(canvas, overlay, alpha):
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    def _text_bg(canvas, text, pos, font, scale, color, thickness=1, alpha=0.70):
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        ov = canvas.copy()
        cv2.rectangle(ov, (x - 6, y - th - 6), (x + tw + 6, y + bl + 4),
                       (12, 12, 20), -1)
        _blend(canvas, ov, alpha)
        cv2.putText(canvas, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    def _glow(canvas, pt, color, rings):
        for r, a in rings:
            ov = canvas.copy()
            cv2.circle(ov, pt, r, color, -1, cv2.LINE_AA)
            _blend(canvas, ov, a)

    # Pre-render static background (grid + full trajectory)
    bg = np.full((S, S, 3), BG_COLOR[0], dtype=np.uint8)
    bg[:, :] = BG_COLOR
    step = S // 12
    for i in range(1, 12):
        cv2.line(bg, (i * step, 0), (i * step, S - 1), GRID_COLOR, 1)
        cv2.line(bg, (0, i * step), (S - 1, i * step), GRID_COLOR, 1)
    for i in range(len(track_px) - 1):
        cv2.line(bg, track_px[i], track_px[i + 1], TRACK_DIM, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for f in range(total_frames):
        canvas = bg.copy()
        ci = min(chunk_for_frame(f), len(chunk_metadata_list) - 1)
        meta = chunk_metadata_list[ci]
        tf = min(f, len(track_px) - 1)  # track frame

        # 1) Chunk segment (green)
        ts, te = meta["track_start"], min(meta["track_end"], len(track_px))
        for i in range(ts, te - 1):
            cv2.line(canvas, track_px[i], track_px[i + 1], CHUNK_COLOR, 3, cv2.LINE_AA)

        # 2) Visited path (white)
        for i in range(tf):
            cv2.line(canvas, track_px[i], track_px[i + 1], TRACK_VISITED, 2, cv2.LINE_AA)

        # 3) Ref target positions on track
        for ref in meta["refs"]:
            ri = min(ref["abs_frame"], len(track_px) - 1)
            cv2.circle(canvas, track_px[ri], 5, REF_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, track_px[ri], 5, (255, 255, 255), 1, cv2.LINE_AA)

        # 4) LA target position on track (diamond)
        la_ti = min(meta["la_abs_frame"], len(track_px) - 1)
        lp = track_px[la_ti]
        d = 7
        diamond = np.array([
            [lp[0], lp[1] - d], [lp[0] + d, lp[1]],
            [lp[0], lp[1] + d], [lp[0] - d, lp[1]],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [diamond], LA_COLOR)
        cv2.polylines(canvas, [diamond], True, (255, 255, 255), 1, cv2.LINE_AA)

        # 5) Connection lines (with glow)
        cur = track_px[tf]
        active_ref_panos = {r["pano_idx"] for r in meta["refs"]}
        ref_pano_counts = Counter(r["pano_idx"] for r in meta["refs"])
        la_pano = meta["la_pano_idx"]

        drawn = set()
        for ref in meta["refs"]:
            pidx = ref["pano_idx"]
            if pidx not in drawn and pidx in pano_px:
                ov = canvas.copy()
                cv2.line(ov, cur, pano_px[pidx], REF_COLOR, 4, cv2.LINE_AA)
                _blend(canvas, ov, 0.25)
                cv2.line(canvas, cur, pano_px[pidx], REF_COLOR, 1, cv2.LINE_AA)
                drawn.add(pidx)
        if la_pano in pano_px:
            ov = canvas.copy()
            cv2.line(ov, cur, pano_px[la_pano], LA_COLOR, 4, cv2.LINE_AA)
            _blend(canvas, ov, 0.25)
            cv2.line(canvas, cur, pano_px[la_pano], LA_COLOR, 1, cv2.LINE_AA)

        # 6) Panorama markers
        for pidx, ppx in pano_px.items():
            is_ref = pidx in active_ref_panos
            is_la = pidx == la_pano
            if is_ref and is_la:
                color, radius = BOTH_COLOR, 11
            elif is_ref:
                color, radius = REF_COLOR, 10
            elif is_la:
                color, radius = LA_COLOR, 10
            else:
                color, radius = INACTIVE_COLOR, 5
            # Glow for active markers
            if is_ref or is_la:
                _glow(canvas, ppx, color, [(radius + 6, 0.10), (radius + 3, 0.18)])
            cv2.circle(canvas, ppx, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, ppx, radius + 1, (170, 170, 180), 1, cv2.LINE_AA)
            # Pano index
            cv2.putText(canvas, str(pidx), (ppx[0] + radius + 4, ppx[1] + 4),
                        font, 0.32, TEXT_LO, 1, cv2.LINE_AA)
            # Ref count if >1
            if pidx in ref_pano_counts and ref_pano_counts[pidx] > 1:
                cv2.putText(canvas, f"x{ref_pano_counts[pidx]}",
                            (ppx[0] + radius + 4, ppx[1] - 8),
                            font, 0.32, REF_COLOR, 1, cv2.LINE_AA)

        # 7) Current position (glow + marker)
        _glow(canvas, cur, CUR_COLOR, [(20, 0.07), (14, 0.14), (10, 0.28)])
        cv2.circle(canvas, cur, 6, CUR_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, cur, 7, (255, 255, 255), 2, cv2.LINE_AA)

        # 8) Text overlay
        _text_bg(canvas, f"Chunk {ci}/{len(chunk_metadata_list)-1}",
                 (12, 28), font, 0.60, TEXT_HI)
        _text_bg(canvas, f"Frame {f}/{total_frames-1}",
                 (12, 56), font, 0.48, TEXT_LO)

        # 9) Legend box
        yl = S - 100
        ov = canvas.copy()
        cv2.rectangle(ov, (6, yl - 12), (110, S - 6), (12, 12, 20), -1)
        _blend(canvas, ov, 0.75)
        cv2.rectangle(canvas, (6, yl - 12), (110, S - 6), (50, 50, 60), 1)
        legend = [
            (CUR_COLOR, "Current"),
            (REF_COLOR, "Ref"),
            (LA_COLOR, "LA"),
            (INACTIVE_COLOR, "Pano"),
        ]
        for i, (c, label) in enumerate(legend):
            y = yl + i * 19
            cv2.circle(canvas, (20, y), 5, c, -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (32, y + 5), font, 0.37, TEXT_LO, 1, cv2.LINE_AA)

        yield canvas


def build_ref_la_video(chunk_ref_images, chunk_la_images, total_width, la_mode="normal", gen_h=704, gen_w=1280):
    """Pre-render ref/LA image rows per chunk (memory-efficient).

    Row 2: [Ref 1 | Ref 2 | Ref 3]
    Row 3: [Ref 4 | Ref 5 | LA]

    Args:
        chunk_ref_images: list of [5, C, H, W] uint8 tensors per chunk.
        chunk_la_images:  list of [1, C, H, W] uint8 tensors per chunk.
        total_width: target row width (must match row 1).
        la_mode: LA mode string for label display.

    Returns:
        list of numpy [2*cell_h, total_width, 3] uint8 arrays, one per chunk.
    """
    import cv2

    cell_w = total_width // 3
    cell_h = round(cell_w * gen_h / gen_w)

    REF_ACCENT = (80, 170, 255)
    LA_ACCENT = (255, 160, 50)

    def _render_cell(img_chw, label, accent):
        img_np = img_chw.permute(1, 2, 0).numpy()
        img = cv2.resize(img_np, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        # Accent border
        cv2.rectangle(img, (0, 0), (cell_w - 1, cell_h - 1), accent, 2)
        # Label bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc, th = 0.55, 1
        (tw, txh), bl = cv2.getTextSize(label, font, sc, th)
        bar_h = txh + bl + 16
        ov = img.copy()
        cv2.rectangle(ov, (2, 2), (cell_w - 3, bar_h), (12, 12, 20), -1)
        cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)
        # Accent tag
        cv2.rectangle(img, (6, 6), (12, bar_h - 4), accent, -1)
        cv2.putText(img, label, (18, txh + 8), font, sc,
                    (255, 255, 255), th, cv2.LINE_AA)
        return img

    # Pre-render combined rows per chunk
    num_chunks = len(chunk_ref_images)
    chunk_rows = []
    for ci in range(num_chunks):
        refs = chunk_ref_images[ci]
        la = chunk_la_images[ci]
        row2 = np.concatenate([
            _render_cell(refs[0], f"Ref 1", REF_ACCENT),
            _render_cell(refs[1], f"Ref 2", REF_ACCENT),
            _render_cell(refs[2], f"Ref 3", REF_ACCENT),
        ], axis=1)
        row3 = np.concatenate([
            _render_cell(refs[3], f"Ref 4", REF_ACCENT),
            _render_cell(refs[4], f"Ref 5", REF_ACCENT),
            _render_cell(la[0],   f"LA ({la_mode})" if la_mode != "normal" else "LA",  LA_ACCENT),
        ], axis=1)
        combined = np.concatenate([row2, row3], axis=0)  # [2*cell_h, 3*cell_w, 3]
        # Fix width to exactly total_width
        if combined.shape[1] < total_width:
            pad = np.zeros((combined.shape[0], total_width - combined.shape[1], 3), dtype=np.uint8)
            combined = np.concatenate([combined, pad], axis=1)
        elif combined.shape[1] > total_width:
            combined = combined[:, :total_width, :]
        chunk_rows.append(combined)

    return chunk_rows


# ============================================================================
# Streaming video writer
# ============================================================================

class _StreamingVideoWriter:
    """Write video frames one at a time via ffmpeg pipe — O(1) memory."""

    def __init__(self, path, fps, width, height, crf=28):
        import subprocess
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
                "-r", str(fps), "-i", "-",
                "-vcodec", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb_hwc):
        """Write one RGB frame — numpy uint8 [H, W, 3]."""
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb_hwc).tobytes())

    def close(self):
        self._proc.stdin.close()
        self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# OOD trajectory helpers
# ============================================================================

def _ensure_gen_metadata(traj_dir: str):
    """Generate gen_metadata.json if missing (same logic as inference_ood_traj.sh)."""
    import json as _json
    meta_path = os.path.join(traj_dir, "gen_metadata.json")
    if os.path.exists(meta_path):
        return

    ext_dir = os.path.join(traj_dir, "gen_track", "extrinsics")
    num_frames = len([f for f in os.listdir(ext_dir) if f.startswith("frame_") and f.endswith(".npy")])

    traj_name = os.path.basename(traj_dir)
    traj_type = "_".join(traj_name.split("_")[2:])
    start_pano = os.path.basename(os.path.dirname(traj_dir))

    first_window = 77
    subsequent_window = 72
    overlap = 5
    stride = first_window - overlap

    if num_frames <= first_window:
        num_windows = 1
    else:
        num_windows = 1 + max(0, (num_frames - first_window + stride - 1) // stride)

    headings = []
    traj_json = os.path.join(traj_dir, "trajectory.json")
    if os.path.exists(traj_json):
        with open(traj_json) as f:
            tdata = _json.load(f)
        headings = [fr.get("heading", 0) for fr in tdata.get("frames", [])]

    chunks = []
    for ci in range(num_windows):
        if ci == 0:
            fs, fe = 0, min(first_window - 1, num_frames - 1)
        else:
            fs = first_window + (ci - 1) * stride
            fe = min(fs + subsequent_window - 1, num_frames - 1)

        hc = 0.0
        tt = "straight"
        if headings and fs < len(headings) and fe < len(headings):
            hc = headings[fe] - headings[fs]
            while hc > 180: hc -= 360
            while hc < -180: hc += 360
            if hc > 15: tt = "right_turn"
            elif hc < -15: tt = "left_turn"

        chunks.append({
            "chunk_idx": ci,
            "frame_start": fs,
            "frame_end": fe,
            "num_frames": fe - fs + 1,
            "heading_change_deg": round(hc, 1),
            "turn_type": tt,
        })

    meta = {
        "sequence": traj_name,
        "start_pano_no": start_pano,
        "trajectory_type": traj_type,
        "num_windows": num_windows,
        "total_frames": num_frames,
        "window_structure": {
            "first_window_frames": first_window,
            "subsequent_window_frames": subsequent_window,
            "history_overlap": overlap,
        },
        "turn_threshold_deg": 15.0,
        "chunks": chunks,
    }

    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)
    logging.info(f"Generated gen_metadata.json: {num_windows} windows, {num_frames} frames")


# ============================================================================
# Per-sample AR generation
# ============================================================================

def run_single_sample(model, dataset, output_dir, args, seed, la_mode,
                      per_chunk_captions=None, *, guardrails):
    """Run AR inference for a single sample (all chunks).

    Args:
        model: Loaded model on GPU.
        dataset: Dataset instance (initialized with correct sample_idx).
        output_dir: Directory to save outputs.
        args: Parsed CLI arguments.
        seed: Random seed.
        la_mode: LA mode string.
        per_chunk_captions: Optional list of captions, one per chunk.
            If provided, dataset.fixed_caption is overridden per chunk.
        guardrails: Cosmos-Guardrail1 (always on): every chunk's caption is checked before it is
            used, and the generated video before it is saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = len(dataset)
    overlap = dataset.overlap_frames

    logging.info(f"AR: {num_chunks} chunks, overlap={overlap}, seed={seed}")

    all_generated_frames = []
    chunk_ref_images = []
    chunk_la_images = []
    chunk_warped_videos = []  # the ACTUAL warp condition the model used (forward-warp mode)
    prev_generated_video = None
    first_cond_frame = None

    for chunk_idx in range(num_chunks):
        # Per-chunk caption override
        if per_chunk_captions is not None and chunk_idx < len(per_chunk_captions):
            dataset.fixed_caption = per_chunk_captions[chunk_idx]
            logging.info(f"Caption for chunk {chunk_idx}: {per_chunk_captions[chunk_idx][:80]}...")

        logging.info(f"\n{'='*60}")
        logging.info(f"CHUNK {chunk_idx}/{num_chunks-1}")
        logging.info(f"{'='*60}")

        sample = dataset[chunk_idx]
        # Cosmos-Guardrail1 text guardrail on the caption this chunk is conditioned on (raises before generating)
        guardrails.check_prompt(sample[model.input_caption_key])
        chunk_ref_images.append(sample["ref_frames"].clone())

        # Save first conditioning GT frame for sink-first mode
        if chunk_idx == 0:
            # video shape: [C, T, H, W] → frame 0: [C, H, W] → [1, C, H, W]
            first_cond_frame = sample["video"][:, 0, :, :].unsqueeze(0)

        # Record the LA image that will actually be fed to the model (for visualization.mp4)
        if la_mode == "off":
            chunk_la_images.append(torch.zeros_like(sample["la_frame"]))
        elif la_mode in ("sink-first", "sink-retrieved") and chunk_idx == 0:
            chunk_la_images.append(torch.zeros_like(sample["la_frame"]))
        elif la_mode == "sink-first" and chunk_idx > 0:
            chunk_la_images.append(first_cond_frame.clone())  # [1, C, H, W] for viz
        else:
            # normal or sink-retrieved (same image)
            chunk_la_images.append(sample["la_frame"].clone())

        data_batch = collate_single(sample)
        data_batch = misc.to(data_batch, device="cuda")

        generated_video = generate_chunk(
            model, data_batch, args.guidance, args.num_steps,
            seed, overlap, chunk_idx, prev_generated_video,
            la_mode=la_mode, first_cond_frame=first_cond_frame,
            drop_ref=args.no_ref, drop_cam=args.no_cam,
        )

        prev_generated_video = generated_video.clone()

        # Reproduce the ACTUAL warp condition (forward-warp mode) from this chunk's
        # ingredients, so the saved warped_video/comparison reflect the real trajectory.
        chunk_warped_videos.append(_forward_warp_from_sample(sample))

        # Convert to uint8
        gen_uint8 = ((generated_video[0] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)  # [C, T, H, W]

        # Accumulate unique frames
        if chunk_idx == 0:
            all_generated_frames.append(gen_uint8.cpu())
        else:
            all_generated_frames.append(gen_uint8[:, overlap:, :, :].cpu())

        # Save per-chunk camera/ref/LA data for later re-visualization
        if hasattr(dataset, "save_chunk_data"):
            npz_path = dataset.save_chunk_data(chunk_idx, str(output_dir))
            logging.info(f"  Saved {npz_path}")

        torch.cuda.empty_cache()

    # Cosmos-Guardrail1 video guardrail (content safety filter, then face pixelation) on the frames that
    # get saved. Runs after the whole AR loop, so the chunks above were generated and conditioned on the
    # unfiltered frames. An unsafe video raises here and nothing is saved.
    chunk_lens = [f.shape[1] for f in all_generated_frames]
    checked = guardrails.check_video(torch.cat(all_generated_frames, dim=1))
    all_generated_frames = [f.contiguous() for f in torch.split(checked, chunk_lens, dim=1)]

    # ---- Save generation metadata (panorama selection, camera info) ----
    if hasattr(dataset, "get_generation_metadata"):
        metadata = dataset.get_generation_metadata()
        metadata["generation_params"] = {
            "guidance": args.guidance,
            "num_steps": args.num_steps,
            "seed": seed,
            "fps": args.fps,
            "la_mode": la_mode,
            "drop_ref": args.no_ref,
            "drop_cam": args.no_cam,
            "save_width": args.save_width,
            "guardrails": "Cosmos-Guardrail1 (text: Blocklist + Qwen3Guard; video: safety filter + face blur)",
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved generation metadata: {metadata_path}")

    # ---- Build final outputs (memory-efficient streaming) ----

    total_frames = sum(f.shape[1] for f in all_generated_frames)
    gen_h, gen_w = all_generated_frames[0].shape[2], all_generated_frames[0].shape[3]
    num_main_frames = dataset.num_main_frames
    stride_val = dataset.stride
    logging.info(f"\nFinal: {total_frames} frames ({total_frames / args.fps:.1f}s)")

    # Datasets without panorama viz metadata (e.g. MarsForwardWarpDataset) only get
    # the concatenated final video; the map/comparison overlay is panorama-specific.
    if not hasattr(dataset, "get_chunk_viz_metadata"):
        logging.info("No panorama viz metadata; writing output.mp4 + visualization.mp4.")
        final_video = torch.cat(all_generated_frames, dim=1).cpu()  # [C, total, H, W]
        save_img_or_video(final_video, str(output_dir / "output"), fps=args.fps)
        logging.info(f"Saved: {output_dir / 'output.mp4'}")
        # visualization.mp4 = [warped condition | generated], if the dataset exposes warped_video
        overlap = dataset.overlap_frames
        warped_chunks = []
        for ci in range(num_chunks):
            if ci < len(chunk_warped_videos) and chunk_warped_videos[ci] is not None:
                w = chunk_warped_videos[ci]  # ACTUAL forward-warp the model used
            else:
                s = dataset[ci]
                if "warped_video" not in s:
                    warped_chunks = None
                    break
                w = s["warped_video"]  # [C, T, H, W] uint8
            warped_chunks.append(w if ci == 0 else w[:, overlap:, :, :])
        if warped_chunks:
            import numpy as _np
            warped_full = torch.cat(warped_chunks, dim=1).cpu()
            tmin = min(warped_full.shape[1], final_video.shape[1])
            gh, gw = final_video.shape[2], final_video.shape[3]
            top = torch.cat([warped_full[:, :tmin], final_video[:, :tmin]], dim=3)   # [C,T,gh,2gw]
            top_np = top.permute(1, 2, 3, 0).numpy().astype(_np.uint8)               # [T,gh,2gw,3]
            # ref/LA rows below (reuse helper): row = [refs... | LA], width matches top
            ref_la_rows = build_ref_la_video(chunk_ref_images, chunk_la_images, gw * 2,
                                             la_mode=la_mode, gen_h=gh, gen_w=gw)
            chunk_lens = [f.shape[1] for f in all_generated_frames]

            def _chunk_of(fi):
                acc = 0
                for ci2, L in enumerate(chunk_lens):
                    if fi < acc + L:
                        return ci2
                    acc += L
                return len(chunk_lens) - 1

            out_frames = [
                _np.concatenate([top_np[fi], ref_la_rows[min(_chunk_of(fi), len(ref_la_rows) - 1)]], axis=0)
                for fi in range(tmin)
            ]
            comp = torch.from_numpy(_np.stack(out_frames)).permute(3, 0, 1, 2)       # [C,T,H,W]
            save_img_or_video(comp, str(output_dir / "visualization"), fps=args.fps)
            logging.info(f"Saved: {output_dir / 'visualization.mp4'} ([warped | generated] + ref/LA)")
        else:
            logging.info("visualization.mp4 skipped: forward-warp computes warped_video inside the model "
                         "(not exposed to the saving loop).")
        return

    # Layout: row1 = [generated | map | warped], row2-3 = [ref/LA]
    map_size = gen_h
    total_width = gen_w + map_size + gen_w

    # Pre-compute per-chunk metadata (small)
    chunk_metadata_list = [dataset.get_chunk_viz_metadata(ci) for ci in range(num_chunks)]
    ref_la_rows = build_ref_la_video(
        chunk_ref_images, chunk_la_images, total_width, la_mode=la_mode,
        gen_h=gen_h, gen_w=gen_w,
    )
    cell_w = total_width // 3
    cell_h = round(cell_w * gen_h / gen_w)
    comparison_h = gen_h + cell_h * 2

    # Map frame generator (yields one numpy [H, H, 3] at a time)
    map_gen = build_map_video(dataset, chunk_metadata_list, total_frames, video_height=gen_h)

    # Helper: global frame index → (chunk_index, offset within chunk)
    def frame_to_chunk(f):
        first_len = all_generated_frames[0].shape[1]
        if f < first_len:
            return 0, f
        remaining = f - first_len
        return 1 + remaining // stride_val, remaining % stride_val

    # Stream all three videos frame-by-frame via ffmpeg pipe
    if args.save_width and args.save_width != gen_w:
        save_w = args.save_width
        save_h = round(gen_h * save_w / gen_w)
        save_h -= save_h % 2  # ensure even for video encoding
        logging.info(f"Output will be downscaled: {gen_w}x{gen_h} -> {save_w}x{save_h}")
    else:
        save_w, save_h = gen_w, gen_h

    logging.info(f"Streaming output/warped/visualization videos ({total_frames} frames)...")
    cur_warped_ci = -1
    cur_warped = None

    with _StreamingVideoWriter(output_dir / "output.mp4", args.fps, save_w, save_h, crf=18) as w_final, \
         _StreamingVideoWriter(output_dir / "warped_video.mp4", args.fps, save_w, save_h, crf=18) as w_warped, \
         _StreamingVideoWriter(output_dir / "visualization.mp4", args.fps, total_width, comparison_h, crf=28) as w_comp:

        for f, map_frame in enumerate(map_gen):
            ci, offset = frame_to_chunk(f)

            # Generated frame [C, H, W] → [H, W, C]
            gen_hwc = all_generated_frames[ci][:, offset, :, :].permute(1, 2, 0).numpy()

            # Warped frame (load one chunk at a time, ~200 MB)
            if ci != cur_warped_ci:
                cur_warped = None
                if ci < len(chunk_warped_videos) and chunk_warped_videos[ci] is not None:
                    # forward-warp mode: the ACTUAL warp the model computed for this chunk
                    w = chunk_warped_videos[ci]
                else:
                    s = dataset[ci]
                    w = s["warped_video"] if "warped_video" in s else s["video"]
                cur_warped = w if ci == 0 else w[:, overlap:, :, :]
                cur_warped_ci = ci
            warped_hwc = cur_warped[:, offset, :, :].permute(1, 2, 0).numpy()

            # Uniform downscale if --save-width specified
            if save_w != gen_w:
                gen_hwc_save = cv2.resize(gen_hwc, (save_w, save_h), interpolation=cv2.INTER_LANCZOS4)
                warped_hwc_save = cv2.resize(warped_hwc, (save_w, save_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                gen_hwc_save = gen_hwc
                warped_hwc_save = warped_hwc
            w_final.write(gen_hwc_save)
            w_warped.write(warped_hwc_save)

            # Compose comparison frame (always at generation resolution)
            row1 = np.concatenate([gen_hwc, map_frame, warped_hwc], axis=1)
            ref_la_row = ref_la_rows[min(ci, len(ref_la_rows) - 1)]
            comp_frame = np.concatenate([row1, ref_la_row], axis=0)
            w_comp.write(comp_frame)

        del cur_warped

    logging.info(f"Saved: {output_dir / 'output.mp4'} ({save_w}x{save_h})")
    logging.info(f"Saved: {output_dir / 'warped_video.mp4'} ({save_w}x{save_h})")
    logging.info(f"Saved visualization: [{total_frames}, {comparison_h}, {total_width}] -> {output_dir / 'visualization.mp4'}")


# ============================================================================
# Main
# ============================================================================

def main():
    init_environment()

    parser = argparse.ArgumentParser(description="AR inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=None,
                        help="Number of AR chunks (overrides dataset config)")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Dataset base directory (overrides dataset config)")
    parser.add_argument("--track-dir", type=str, default=None,
                        help="Track subdirectory name (overrides dataset config)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (overrides dataset fixed_caption)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="JSON file with named prompts. Each sample picks one uniformly at random. "
                             "Use --prompt-file-exclude to skip keys (default: exclude 'base').")
    parser.add_argument("--prompt-file-exclude", type=str, nargs="*", default=["base"],
                        help="Keys to exclude from --prompt-file (default: ['base'])")
    parser.add_argument("--no-la", action="store_true",
                        help="Drop LA frame (zero out la_frame in data batch)")
    parser.add_argument("--la-mode", type=str, default="normal",
                        choices=["normal", "off", "sink-first", "sink-retrieved"],
                        help="LA attention sink variant: normal (default), off (no LA), "
                             "sink-first (chunk0 GT frame0 as attention sink at t=0), "
                             "sink-retrieved (dataset LA image as attention sink at t=0)")
    parser.add_argument("--no-ref", action="store_true",
                        help="Drop reference frames (zero out ref_frames and ref_mask)")
    parser.add_argument("--no-cam", action="store_true",
                        help="Drop camera embeddings (zero out cam_mask for main+LA+ref)")
    parser.add_argument("--use-gt-track", action="store_true",
                        help="Use gen_track_gt instead of gen_track (for qual evaluation)")
    parser.add_argument("--save-width", type=int, default=None,
                        help="Uniform downscale output videos to this width (height auto, e.g. --save-width 720)")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="Subsequence index for validation AR (overrides dataset sample_idx)")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to generate in one run (model loaded once)")
    parser.add_argument("--sample-start", type=int, default=0,
                        help="Starting sample index for multi-sample mode (default: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples whose output directories already exist")
    parser.add_argument("--traj-list", type=str, default=None,
                        help="File listing OOD trajectory directories (one per line). "
                             "Model loaded once, each GPU processes different trajectories.")

    args = parser.parse_args()

    # Backward compat: --no-la maps to la_mode="off"
    la_mode = args.la_mode
    if args.no_la:
        la_mode = "off"

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()

    overrides = list(args.opts) if args.opts else []
    overrides.append("checkpoint.load_training_state=False")
    overrides.append("job.group=ar_inference")
    # Inference keeps a full (unsharded) model on each GPU; multi-GPU runs distribute
    # trajectories/samples across ranks (data parallel), never the model itself. This
    # matches upstream inference (load_model_from_checkpoint defaults to enable_fsdp=False,
    # i.e. fsdp_shard_size=1). The model default (8) is a training value, so force 1 —
    # otherwise single-GPU runs try to build an 8-way FSDP mesh and fail.
    overrides.append("model.config.fsdp_shard_size=1")

    if args.output_dir:
        # Videos go straight into args.output_dir. Inference writes no log/config
        # files; this only isolates the framework's (unused) job dir so nothing
        # clutters the output directory.
        os.environ["IMAGINAIRE_OUTPUT_ROOT"] = os.path.join(args.output_dir, ".logs")

    config = override(config, overrides)

    # Override dataset parameters before instantiation
    if args.base_dir:
        config.dataloader_val.dataset.base_dir = args.base_dir
        config.dataloader_train.dataset.base_dir = args.base_dir
    if args.track_dir:
        config.dataloader_val.dataset.track_dir = args.track_dir
        config.dataloader_train.dataset.track_dir = args.track_dir
    if args.num_chunks:
        try:
            config.dataloader_val.dataset.num_chunks = args.num_chunks
            config.dataloader_train.dataset.num_chunks = args.num_chunks
        except Exception:
            pass  # Dataset may not have num_chunks (e.g. validation AR with fixed 2 chunks)
    # Load prompt file (per-sample random prompts)
    prompt_pool = None
    if args.prompt_file:
        import json as _json
        with open(args.prompt_file) as _f:
            _all_prompts = _json.load(_f)
        exclude_keys = set(args.prompt_file_exclude or [])
        prompt_pool = {k: v for k, v in _all_prompts.items() if k not in exclude_keys}
        prompt_pool_keys = sorted(prompt_pool.keys())
        if is_rank0():
            logging.info(f"Prompt pool: {len(prompt_pool)} prompts from {args.prompt_file} "
                         f"(excluded: {exclude_keys})")
            for k in prompt_pool_keys:
                v = prompt_pool[k]
                if isinstance(v, dict):
                    phases = ", ".join(v.keys())
                    logging.info(f"  [{k}] phased ({phases})")
                else:
                    logging.info(f"  [{k}] {v[:80]}...")
    if args.prompt:
        config.dataloader_val.dataset.fixed_caption = args.prompt
        config.dataloader_train.dataset.fixed_caption = args.prompt
    if args.use_gt_track:
        config.dataloader_val.dataset.use_gt_track = True
        config.dataloader_train.dataset.use_gt_track = True
    if args.sample_idx is not None:
        config.dataloader_val.dataset.sample_idx = args.sample_idx
        config.dataloader_train.dataset.sample_idx = args.sample_idx

    if is_rank0():
        output_dir = Path(args.output_dir) if args.output_dir else Path(config.job.path_local)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("=" * 60)
        logging.info("AUTOREGRESSIVE (AR) INFERENCE")
        logging.info("=" * 60)
        logging.info(f"Output: {output_dir}")
        logging.info(f"Guidance: {args.guidance}, Steps: {args.num_steps}")
        logging.info(f"Chunks: {args.num_chunks if args.num_chunks else 'from config'}")
        logging.info(f"Seed: {args.seed if args.seed else 'random'}")
        logging.info(f"LA mode: {la_mode}")
        if args.use_gt_track:
            logging.info("Track: gen_track_gt (GT qual mode)")
        logging.info("=" * 60)

    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True)
            + "\n" + pretty_print_overrides(overrides, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"Config saved to: {config.job.path_local}/config.yaml")
        return

    # Cosmos-Guardrail1, always on (no switch). Load it before the model so a missing checkpoint fails
    # fast, and check the prompts known up front before anything is generated: the dataset prompt
    # (after --prompt) and every --prompt-file entry. Each chunk's caption is checked again in
    # run_single_sample (prompts that passed are cached), which covers prompts a dataset brings itself.
    guardrails = Guardrails()
    _ds_cfg = config.dataloader_val.dataset
    upfront_prompts = [getattr(_ds_cfg, key, None) for key in ("fixed_caption", "prompt")]
    if prompt_pool is not None:
        for _entry in prompt_pool.values():
            upfront_prompts += list(_entry.values()) if isinstance(_entry, dict) else [_entry]
    for _p in upfront_prompts:
        if isinstance(_p, str) and _p:
            guardrails.check_prompt(_p)

    # Save dataset config before freeze (for multi-sample re-instantiation)
    dataset_cfg = deepcopy(config.dataloader_val.dataset)

    # ---- Load model and checkpoint directly (no trainer) ----
    config.validate()
    config.freeze()

    from cosmos_predict2._src.imaginaire.lazy_config import instantiate
    from cosmos_predict2._src.predict2.utils.model_loader import (
        create_model_from_consolidated_checkpoint_with_fsdp,
    )

    # Initialize the Megatron model-parallel state. The trainer used to do this in
    # its constructor; without it the dataloader sampler's get_data_parallel_world_size()
    # fails. All parallel sizes are 1 for single-GPU inference, but the data-parallel
    # group must still exist.
    import inspect as _inspect

    from megatron.core import parallel_state

    if not parallel_state.is_initialized():
        _mp = config.model_parallel
        _mp_kwargs = dict(
            pipeline_model_parallel_size=_mp.pipeline_model_parallel_size,
            tensor_model_parallel_size=_mp.tensor_model_parallel_size,
            context_parallel_size=_mp.context_parallel_size,
        )
        if "create_gloo_process_groups" in _inspect.signature(parallel_state.initialize_model_parallel).parameters:
            _mp_kwargs["create_gloo_process_groups"] = False
        parallel_state.initialize_model_parallel(**_mp_kwargs)
        parallel_state.sequence_parallel = _mp.sequence_parallel

    # Backend flags the trainer used to set on our behalf.
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    # Load the consolidated checkpoint (model.pt). Training writes sharded DCP
    # checkpoints; convert one to a single .pt with scripts/convert_distcp_to_pt.py
    # before running inference.
    if not (isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt")):
        raise ValueError(
            f"CKPT must be a consolidated .pt checkpoint (model.pt), got: {config.checkpoint.load_path}\n"
            "Convert a DCP 'iter_XXXXXXXXX' directory with:\n"
            "  python scripts/convert_distcp_to_pt.py <iter_dir>/model <iter_dir>"
        )
    memory_format = config.trainer.memory_format
    model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    model = model.to("cuda", memory_format=memory_format)
    model.on_train_start(memory_format)

    dataloader_val = instantiate(config.dataloader_val)

    model.eval()

    # ---- AR generation ----
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)

    if args.traj_list is not None:
        # Traj-list mode: iterate through OOD trajectory directories, model loaded once
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        root_output = Path(args.output_dir) if args.output_dir else Path(config.job.path_local)
        root_output.mkdir(parents=True, exist_ok=True)

        # Read trajectory directories from file
        with open(args.traj_list) as f:
            all_traj_dirs = [line.strip() for line in f if line.strip()]

        if rank == 0:
            logging.info(f"Traj-list mode: {len(all_traj_dirs)} trajectories from {args.traj_list}")

        # Filter: skip already completed (resume) and ensure gen_metadata.json
        trajs_to_run = []
        skipped = 0
        for traj_dir in all_traj_dirs:
            traj_path = Path(traj_dir)
            # Output dir name: pano_id/traj_name (e.g. 99137000024/traj_2_straight_left90)
            pano_id = traj_path.parent.name
            traj_name = traj_path.name
            out_name = f"{pano_id}__{traj_name}"
            out_dir = root_output / out_name

            if args.resume and (out_dir / "output.mp4").exists():
                if rank == 0:
                    logging.info(f"{out_name} — SKIP (exists)")
                skipped += 1
                continue

            # Ensure gen_metadata.json exists
            gen_meta_path = traj_path / "gen_metadata.json"
            if not gen_meta_path.exists():
                _ensure_gen_metadata(str(traj_path))

            trajs_to_run.append((traj_dir, out_name))

        if not trajs_to_run:
            if rank == 0:
                logging.info("All trajectories already completed. Nothing to do.")
        else:
            # Distribute across ranks (round-robin)
            my_trajs = trajs_to_run[rank::world_size]
            max_per_rank = (len(trajs_to_run) + world_size - 1) // world_size
            n_real = len(my_trajs)
            # Pad for FSDP sync
            while len(my_trajs) < max_per_rank:
                my_trajs.append(my_trajs[-1])

            if rank == 0:
                logging.info(f"{len(trajs_to_run)} trajs across {world_size} GPUs "
                             f"({max_per_rank}/GPU, {skipped} skipped)")

            done, failed = 0, 0
            for step_i, (traj_dir, out_name) in enumerate(my_trajs):
                is_real = step_i < n_real
                if is_real:
                    out_dir = root_output / out_name
                else:
                    out_dir = root_output / f".pad_rank{rank}"

                logging.info(f"\n{'='*60}")
                logging.info(f"[GPU {rank}] [{step_i + 1}/{max_per_rank}] "
                             f"{'TRAJ' if is_real else 'PAD'} {out_name}")
                logging.info(f"{'='*60}")
                try:
                    # Read num_chunks from gen_metadata.json
                    gen_meta_path = Path(traj_dir) / "gen_metadata.json"
                    with open(gen_meta_path) as f:
                        gen_meta = json.load(f)
                    num_chunks = gen_meta["num_windows"]

                    dataset_cfg.base_dir = traj_dir
                    dataset_cfg.num_chunks = num_chunks
                    dataset = instantiate(dataset_cfg)
                    run_single_sample(model, dataset, out_dir, args, seed, la_mode, guardrails=guardrails)
                    if is_real:
                        done += 1
                    else:
                        import shutil
                        shutil.rmtree(out_dir, ignore_errors=True)
                    del dataset
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"FAILED [GPU {rank}]: {out_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    if is_real:
                        failed += 1

            logging.info(f"\n{'='*60}")
            logging.info(f"[GPU {rank}] Done: {done}, Failed: {failed}")
            logging.info(f"Output: {root_output}")
            logging.info(f"{'='*60}")

    elif args.n_samples is not None and args.n_samples != 0:
        # Multi-sample mode: distribute samples across GPUs, model loaded once
        # n_samples < 0 → infinite mode (generate until Ctrl+C)
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        root_output = Path(args.output_dir) if args.output_dir else Path(config.job.path_local)
        root_output.mkdir(parents=True, exist_ok=True)

        infinite = args.n_samples < 0
        if not infinite:
            max_steps = (args.n_samples + world_size - 1) // world_size
        else:
            max_steps = None

        if rank == 0:
            if infinite:
                logging.info(f"Infinite mode: {world_size} GPUs, Ctrl+C to stop")
            else:
                logging.info(f"{args.n_samples} samples across {world_size} GPUs ({max_steps}/GPU)")

        done, failed, skipped = 0, 0, 0
        step = 0

        try:
            while max_steps is None or step < max_steps:
                si = args.sample_start + step * world_size + rank
                is_pad = not infinite and (step * world_size + rank >= args.n_samples)

                # Per-sample random prompt from prompt pool
                chosen_prompt_key = None
                per_chunk_captions = None
                if prompt_pool is not None:
                    import random as _rnd
                    _rng = _rnd.Random(si)  # deterministic per sample index
                    chosen_prompt_key = _rng.choice(prompt_pool_keys)
                    chosen_entry = prompt_pool[chosen_prompt_key]
                    if isinstance(chosen_entry, dict):
                        # Phased prompts: chunk0=onset, chunk1=sustain
                        per_chunk_captions = [
                            chosen_entry.get("onset", chosen_entry.get("sustain", "")),
                            chosen_entry.get("sustain", chosen_entry.get("onset", "")),
                        ]
                        dataset_cfg.fixed_caption = per_chunk_captions[0]
                    else:
                        dataset_cfg.fixed_caption = chosen_entry

                # Build sample name (include prompt key if available)
                if chosen_prompt_key is not None:
                    sample_name = f"sample_{si:04d}_{chosen_prompt_key}"
                else:
                    sample_name = f"sample_{si:04d}"
                sample_dir = root_output / sample_name

                # Resume: use metadata.json as completion marker
                # Also check for glob pattern sample_{si:04d}_* to handle prompt-keyed dirs
                already_done = False
                if args.resume:
                    if (sample_dir / "metadata.json").exists():
                        already_done = True
                    else:
                        import glob as _glob
                        _matches = _glob.glob(str(root_output / f"sample_{si:04d}_*" / "metadata.json"))
                        if _matches:
                            already_done = True

                # Synchronized skip: all ranks must agree (FSDP requires matched forward calls)
                if dist.is_initialized():
                    skip_flag = torch.tensor([1 if already_done else 0], device="cuda")
                    dist.all_reduce(skip_flag, op=dist.ReduceOp.MIN)
                    all_skip = skip_flag.item() == 1
                else:
                    all_skip = already_done

                if all_skip:
                    logging.info(f"[GPU {rank}] step {step} — {sample_name} SKIP")
                    skipped += 1
                    step += 1
                    continue

                if is_pad:
                    out_dir = root_output / f".pad_rank{rank}"
                elif args.n_samples == 1 and world_size == 1:
                    # single clip: write videos straight into the chosen output dir
                    out_dir = root_output
                else:
                    out_dir = sample_dir

                logging.info(f"\n{'='*60}")
                logging.info(f"[GPU {rank}] step {step} — "
                             f"{'PAD' if is_pad else 'SAMPLE'} {sample_name}")
                logging.info(f"{'='*60}")
                try:
                    dataset_cfg.sample_idx = si
                    dataset = instantiate(dataset_cfg)
                    run_single_sample(model, dataset, out_dir, args, seed, la_mode,
                                      per_chunk_captions=per_chunk_captions, guardrails=guardrails)
                    if not is_pad:
                        done += 1
                    else:
                        import shutil
                        shutil.rmtree(out_dir, ignore_errors=True)
                    del dataset
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"FAILED [GPU {rank}]: {sample_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    if not is_pad:
                        failed += 1

                step += 1
        except KeyboardInterrupt:
            logging.info(f"\n[GPU {rank}] Stopped by user.")

        logging.info(f"\n{'='*60}")
        logging.info(f"[GPU {rank}] Done: {done}, Failed: {failed}, Skipped: {skipped}")
        logging.info(f"Output: {root_output}")
        logging.info(f"{'='*60}")
    else:
        # Single-sample mode (backward compatible)
        dataset = dataloader_val.dataset
        output_dir = Path(config.job.path_local) / "ar_output"
        run_single_sample(model, dataset, output_dir, args, seed, la_mode, guardrails=guardrails)


if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
