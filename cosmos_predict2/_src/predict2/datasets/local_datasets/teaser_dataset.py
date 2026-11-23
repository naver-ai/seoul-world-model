# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
"""Teaser dataset: regenerate the released teaser video from the sample folder.

The sample folder holds the posed frames of the teaser route in the TrajectoryDataset
layout (so the same folder also works with ``TrajectoryDataset``), the images the model
was conditioned on, and ``teaser.json``, which says which file plays which role:

    data_dir/
        images/<name>.png        route_00..route_11 look forward along the route;
                                 view<K>_<direction> are 8 views of nearby panorama K
        depth/<name>.npy         depth map (H, W), same size as the image
        intrinsics/<name>.npy    camera intrinsics K (3, 3) for the image
        extrinsics/<name>.npy    world-to-camera pose (3, 4)
        tokens/start.png         first frame of the video (video size)
        tokens/chunk<C>_ref<K>.png, tokens/chunk<C>_lookahead.png   reference / look-ahead images
        teaser.json

Rules (those the teaser was generated with):
  * camera path: the 12 route poses placed evenly on the (77 + 72)-frame timeline,
    rotation SLERP and linear translation between them, shifted sideways by
    ``lateral_offset`` metres, cut into two 77-frame chunks with a 5-frame overlap;
  * warp condition, per target frame of a chunk: the nearest of the chunk's
    ``warp_references`` within ``ref_warp_distance``, else the nearest of the chunk's 6
    route views, forward-warped on the CPU with its depth;
  * resize, as in training (rendered 720x1280, video 704x1280): the image is centre-cropped
    to the video aspect ratio and resized, while depth and K are scaled from the whole frame
    with no crop offset, so image and depth rows drift apart by up to 8 px towards the top
    and bottom edges. Kept so the teaser reproduces; ``TrajectoryDataset`` aligns them;
  * reference / look-ahead images and the start frame: the files listed in teaser.json.
    As in training, the reference and look-ahead camera poses are taken from the camera
    path, not from where the images were captured: frames 12, 28, 44, 60, 76 and, for the
    look-ahead, the chunk's last frame (the teaser pipeline's stand-in for the look-ahead
    time). The warp uses each source frame's own pose and depth.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataloader.utils import make_relative_extrinsics
from cosmos_predict2._src.predict2.datasets.local_datasets.warp import (
    extrinsic_3x4_to_4x4,
    forward_warp_frame,
    unproject_points_warp,
    w2c_to_c2w,
)

TEASER_FORMAT = "swm_teaser/1"


class TeaserDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        fixed_caption: "str | None" = None,   # None -> the prompt in teaser.json
        video_size: tuple[int, int] = (704, 1280),  # (H, W)
        warp_device: str = "cpu",
        sample_idx: int = 0,                  # accepted+ignored (single clip)
        **kwargs,
    ):
        super().__init__()
        path = os.path.join(data_dir, "teaser.json")
        assert os.path.isfile(path), f"{path} not found (TEASER=1 needs the sample folder)"
        with open(path) as f:
            self.plan = json.load(f)
        assert self.plan.get("format") == TEASER_FORMAT, f"{path}: format {self.plan.get('format')!r}"
        self.data_dir = data_dir
        self.video_size = tuple(video_size)
        self.warp_device = warp_device
        self.fixed_caption = fixed_caption or self.plan["prompt"]
        self.num_main_frames = int(self.plan["num_main_frames"])     # 77
        self.overlap_frames = int(self.plan["overlap_frames"])       # 5
        self.stride = self.num_main_frames - self.overlap_frames     # 72
        self.la_offset = int(self.plan["la_offset"])                 # 20
        self.num_ref_views = int(self.plan["num_ref_views"])         # 5
        self.ref_warp_distance = float(self.plan["ref_warp_distance"])
        self.lateral_offset = float(self.plan["lateral_offset"])
        self.fps = float(self.plan["fps"])
        self.route = list(self.plan["route"])
        assert len(self.route) == 12, "teaser.json: route must list 12 frames (6 per chunk)"
        self._loaded = False
        log.info(f"TeaserDataset: {data_dir}, lateral_offset={self.lateral_offset:.3f}m")

    # ----- file access ------------------------------------------------------
    def _rgb(self, rel):
        return np.array(Image.open(os.path.join(self.data_dir, rel)).convert("RGB"))

    def _resize_image(self, img):
        """Crop to the video aspect ratio, then resize to the video size."""
        target_h, target_w = self.video_size
        h0, w0 = img.shape[:2]
        if w0 / h0 > target_w / target_h:
            new_w = int(h0 * target_w / target_h)
            left = (w0 - new_w) // 2
            img = img[:, left:left + new_w]
        else:
            new_h = int(w0 / (target_w / target_h))
            top = (h0 - new_h) // 2
            img = img[top:top + new_h]
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _frame(self, name):
        """A posed frame resized as in training: image cropped, depth and K from the whole frame."""
        H, W = self.video_size
        img = self._resize_image(self._rgb(os.path.join("images", f"{name}.png")))
        depth = np.load(os.path.join(self.data_dir, "depth", f"{name}.npy")).astype(np.float32)
        K = np.load(os.path.join(self.data_dir, "intrinsics", f"{name}.npy")).astype(np.float32)
        w2c = extrinsic_3x4_to_4x4(np.load(os.path.join(self.data_dir, "extrinsics", f"{name}.npy"))[:3, :])
        sx, sy = W / depth.shape[1], H / depth.shape[0]
        K = K.copy()
        K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy
        return {
            "image": img,
            "depth": cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR),
            "K": K,
            "w2c": w2c,
            "position": w2c_to_c2w(w2c)[:3, 3],
        }

    def _token(self, rel):
        if rel is None:
            H, W = self.video_size
            return np.zeros((H, W, 3), np.uint8)
        img = self._rgb(rel)
        assert img.shape[:2] == self.video_size, f"{rel}: {img.shape[:2]} != video size {self.video_size}"
        return img

    # ----- camera path ------------------------------------------------------
    def _camera_path(self):
        T, total = self.num_main_frames, self.num_main_frames + self.stride
        rots, trans = [], []
        for name in self.route:
            c2w = w2c_to_c2w(extrinsic_3x4_to_4x4(
                np.load(os.path.join(self.data_dir, "extrinsics", f"{name}.npy"))[:3, :]))
            rots.append(Rotation.from_matrix(c2w[:3, :3]))
            trans.append(c2w[:3, 3])
        key_t = np.linspace(0, total - 1, 12)
        slerp = Slerp(key_t, Rotation.concatenate(rots))
        w2c_all = np.zeros((total, 3, 4), dtype=np.float64)
        for f in range(total):
            seg = max(0, min(np.searchsorted(key_t, f, side="right") - 1, 10))
            t0, t1 = key_t[seg], key_t[seg + 1]
            a = max(0.0, min(1.0, (f - t0) / (t1 - t0) if t1 > t0 else 0.0))
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = slerp(f).as_matrix()
            c2w[:3, 3] = (1 - a) * trans[seg] + a * trans[seg + 1]
            w2c_all[f] = np.linalg.inv(c2w)[:3, :]
        return [self._shift_sideways(w2c_all[:T]), self._shift_sideways(w2c_all[self.stride:self.stride + T])]

    def _shift_sideways(self, w2c_all):
        out = w2c_all.copy()
        for i in range(len(w2c_all)):
            m = np.eye(4, dtype=np.float64)
            m[:3, :] = w2c_all[i]
            c2w = np.linalg.inv(m)
            right = c2w[:3, 0] / (np.linalg.norm(c2w[:3, 0]) + 1e-8)   # RDF: column 0 = right
            c2w[:3, 3] += self.lateral_offset * right
            out[i] = np.linalg.inv(c2w)[:3, :]
        return out

    # ----- warp condition ---------------------------------------------------
    def _warp_chunk(self, route_frames, refs, w2c_chunk):
        H, W = self.video_size
        dev = self.warp_device
        to_t = lambda img: torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        warped, masks = [], []
        for w2c in w2c_chunk:
            target = np.eye(4, dtype=np.float64)
            target[:3, :] = w2c
            pos = np.linalg.inv(target)[:3, 3]
            src, best = None, float("inf")
            for r in refs:                       # a listed nearby view within ref_warp_distance first
                d = np.linalg.norm(pos - r["position"])
                if d <= self.ref_warp_distance and d < best:
                    best, src = d, r
            if src is None:                      # else the nearest route view of this chunk
                src = route_frames[int(np.argmin([np.linalg.norm(pos - g["position"]) for g in route_frames]))]
            pts = unproject_points_warp(
                torch.from_numpy(src["depth"]).float()[None, None].to(dev),
                torch.from_numpy(src["w2c"]).float()[None].to(dev),
                torch.from_numpy(src["K"]).float()[None].to(dev))
            w, m = forward_warp_frame(
                to_t(src["image"])[None].to(dev), pts,
                torch.from_numpy(target).float()[None].to(dev),
                torch.from_numpy(src["K"]).float()[None].to(dev))
            warped.append(w[0].cpu())
            masks.append(m[0].reshape(1, H // 8, 8, W // 8, 8).mean(dim=(2, 4)))
        video = (torch.stack(warped, dim=0).permute(1, 0, 2, 3) * 255).clamp(0, 255).to(torch.uint8)
        return video, torch.stack(masks, dim=0).permute(1, 0, 2, 3)

    def _load(self):
        if self._loaded:
            return
        self._w2c = self._camera_path()
        self._K_scaled, self._warped, self._warp_masks = [], [], []
        for c in range(2):
            route_frames = [self._frame(n) for n in self.route[6 * c:6 * c + 6]]
            refs = [self._frame(n) for n in self.plan["chunks"][c]["warp_references"]]
            self._K_scaled.append(np.mean([g["K"] for g in route_frames], axis=0))
            video, mask = self._warp_chunk(route_frames, refs, self._w2c[c])
            self._warped.append(video)
            self._warp_masks.append(mask)
        self._start = self._token(self.plan["start_frame"])
        self._loaded = True

    # ----- dataset ----------------------------------------------------------
    def __len__(self):
        return 2

    def __getitem__(self, chunk_idx):
        self._load()
        H, W = self.video_size
        T = self.num_main_frames
        chunk = self.plan["chunks"][chunk_idx]
        w2c, K = self._w2c[chunk_idx], self._K_scaled[chunk_idx]
        warped = self._warped[chunk_idx]
        video = warped.clone()
        if chunk_idx == 0:
            video[:, 0] = torch.from_numpy(self._start).permute(2, 0, 1)

        local = np.linspace(0, T - 1, self.num_ref_views + 1, dtype=int)[1:]
        ref_times = [min(int(f) // 4, 19) for f in local]
        ref_imgs = [self._token(rel) for rel in chunk["references"]]
        # reference / look-ahead poses come from the camera path, as in training (module docstring)
        ref_w2c = w2c[np.clip(np.array(ref_times) * 4, 0, T - 1)]
        la_img = self._token(chunk["lookahead"])
        return {
            "video": video,
            "warped_video": warped,
            "warp_mask": self._warp_masks[chunk_idx][:, :T][:, ::4],
            "ref_frames": torch.from_numpy(np.stack(ref_imgs)).permute(0, 3, 1, 2).contiguous().to(torch.uint8),
            "ref_mask": torch.tensor(chunk["reference_mask"], dtype=torch.float32),
            "ref_time_indices": torch.tensor(ref_times, dtype=torch.long),
            "la_frame": torch.from_numpy(la_img).permute(2, 0, 1).unsqueeze(0).contiguous().to(torch.uint8),
            "la_time_indices": torch.tensor([(T - 1 + self.la_offset) // 4], dtype=torch.long),
            "la_mask": torch.tensor([1.0]),
            "main_extrinsics": torch.from_numpy(make_relative_extrinsics(w2c[:T], w2c[0])).float(),
            "main_intrinsics": torch.from_numpy(np.tile(K, (T, 1, 1))).float(),
            "ref_extrinsics": torch.from_numpy(make_relative_extrinsics(ref_w2c, w2c[0])).float(),
            "ref_intrinsics": torch.from_numpy(np.tile(K, (len(ref_times), 1, 1))).float(),
            "la_extrinsics": torch.from_numpy(make_relative_extrinsics(w2c[T - 1:T], w2c[0])).float(),
            "la_intrinsics": torch.from_numpy(K[np.newaxis]).float(),
            "cam_mask": torch.tensor(1.0),
            "ai_caption": self.fixed_caption,
            "fps": torch.tensor(self.fps),
            "image_size": torch.tensor([H, W, H, W]),
            "num_frames": T,
            "padding_mask": torch.zeros(1, H, W),
            "num_conditional_frames": 1 if chunk_idx == 0 else self.overlap_frames,
            "chunk_idx": chunk_idx,
            "track_start": 0 if chunk_idx == 0 else self.stride,
        }

    def get_generation_metadata(self):
        return {
            "teaser": os.path.abspath(self.data_dir),
            "lateral_offset": self.lateral_offset,
            "num_main_frames": self.num_main_frames,
            "overlap_frames": self.overlap_frames,
            "stride": self.stride,
            "la_offset": self.la_offset,
            "video_size": list(self.video_size),
            "fps": self.fps,
            "chunks": self.plan["chunks"],
        }
