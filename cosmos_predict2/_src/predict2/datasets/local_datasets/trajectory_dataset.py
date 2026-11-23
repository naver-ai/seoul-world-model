# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
"""Trajectory generation dataset (retrieval + forward-warp).

Given a folder of posed reference frames — each an RGB image plus its depth,
intrinsics and extrinsics — you pick a start frame and a camera trajectory (see
``TRAJECTORIES``: straight-then-turn, S-curve, or continuous smooth turns, left or
right). The dataset generates a metric camera path, and for every pose along it
retrieves the nearest reference
(closest position, then most similar viewing direction) and emits the ingredients
for the model to forward-warp it on the GPU. References and a look-ahead frame are
retrieved the same way. No external services or pre-computed warps are needed.

Expected data layout (all four folders share the same per-frame *basename*; the
image extension is free — `.jpg`, `.png`, `.jpeg`, … — and `<name>` can be any
string, only the per-frame pairing across the four folders matters):

    data_dir/
        images/<name>.{jpg,jpeg,png,bmp,webp,tif,tiff}   RGB
        depth/<name>.npy        depth map (H, W)
        intrinsics/<name>.npy   camera intrinsics K (3, 3)
        extrinsics/<name>.npy   world-to-camera pose (3, 4)
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataloader.utils import (
    make_relative_extrinsics,
)

# Trajectory presets, mirroring the OOD reference set (minus plain `straight`).
# left = turn left, right = turn right; `*90` ramps a discrete turn, `smooth_*`
# turns continuously across the whole clip. Turn magnitude is `turn_deg`.
TRAJECTORIES = (
    "straight",           # drive straight ahead, no turn
    "straight_left90",    # straight, then a left turn, then straight
    "straight_right90",   # straight, then a right turn, then straight
    "left90_right90",     # left turn, then an opposite right turn (S-curve, net ~0)
    "right90_left90",     # right turn, then an opposite left turn (S-curve, net ~0)
    "smooth_left",        # continuous gentle left across the clip
    "smooth_right",       # continuous gentle right across the clip
)
# Supported reference-image extensions (matched case-insensitively).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        trajectory: str = "smooth_left",      # one of TRAJECTORIES
        start_frame: "int | str" = 0,         # image name, or index into the sorted frames
        prompt: str = (
            "A driving video captured from the front-facing camera of a vehicle moving "
            "through a city, with buildings, intersections, traffic and clear daytime lighting."
        ),
        speed: float = 0.3,                   # forward step per frame, in meters
        turn_deg: float = 90.0,               # magnitude of each turn (deg)
        turn_rate: float = 4.0,               # turn speed for `*90` turns (deg per frame)
        group_tol: float = 2.0,               # retrieval: positions within this many meters of the
                                              # nearest reference count as the same spot (then pick
                                              # the closest viewing direction among them)
        num_chunks: int = 2,
        video_size: tuple[int, int] = (704, 1280),  # (H, W)
        num_main_frames: int = 77,
        overlap_frames: int = 5,
        num_ref_views: int = 5,
        la_offset: int = 20,
        fps: float = 10.0,
        sample_idx: int = 0,                  # accepted+ignored (single clip)
        **kwargs,
    ):
        super().__init__()
        assert trajectory in TRAJECTORIES, f"trajectory must be one of {TRAJECTORIES}"
        self.H, self.W = video_size
        self.trajectory = trajectory
        self.prompt = prompt
        self.speed = speed
        self.turn_deg = turn_deg
        self.turn_rate = turn_rate
        self.group_tol = group_tol
        self.num_chunks = num_chunks
        self.num_main_frames = num_main_frames
        self.overlap_frames = overlap_frames
        self.stride = num_main_frames - overlap_frames
        self.num_ref_views = num_ref_views
        self.la_offset = la_offset
        self.fps = fps
        self.data_dir = data_dir

        # --- load every posed frame as a reference ---
        # Any image extension in IMAGE_EXTS works; frames are ordered by basename and
        # paired with depth/intrinsics/extrinsics of the same basename (`.npy`).
        img_paths = sorted(
            (p for p in glob.glob(os.path.join(data_dir, "images", "*"))
             if os.path.splitext(p)[1].lower() in IMAGE_EXTS),
            key=lambda p: os.path.splitext(os.path.basename(p))[0],
        )
        assert img_paths, f"no images found under {data_dir}/images (extensions: {IMAGE_EXTS})"
        self._depth_cache: dict[str, np.ndarray] = {}
        self.refs = []  # list of {name, image, extrinsic(3x4 w2c), intrinsic(3x3)}
        for p in img_paths:
            n = os.path.splitext(os.path.basename(p))[0]
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            assert img is not None, f"failed to read image {p}"
            self.refs.append({
                "name": n,
                "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                "extrinsic": np.load(os.path.join(data_dir, "extrinsics", f"{n}.npy")).astype(np.float64)[:3, :],
                "intrinsic": np.load(os.path.join(data_dir, "intrinsics", f"{n}.npy")).astype(np.float64),
            })
        self.ref_pos = np.stack([self._cam_pos(r["extrinsic"]) for r in self.refs])

        self._build_trajectory(start_frame)
        log.info(f"TrajectoryDataset: {len(self.refs)} refs from {data_dir}, "
                 f"trajectory={trajectory}, start={self.refs[self._start_i]['name']}, "
                 f"speed={self.speed}m, chunks={num_chunks}")

    # ----- geometry helpers -------------------------------------------------
    @staticmethod
    def _cam_pos(w2c):
        return -w2c[:3, :3].T @ w2c[:3, 3]

    @staticmethod
    def _geodesic(R1, R2):
        return float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)))

    def _scale_K(self, K, src_w, src_h):
        Ks = K.copy().astype(np.float64)
        s = self.W / src_w
        Ks[0, 0] *= s; Ks[1, 1] *= s; Ks[0, 2] *= s; Ks[1, 2] *= s
        Ks[1, 2] -= (round(src_h * s) - self.H) // 2  # center-crop offset
        return Ks

    def _resize(self, arr, interp):
        target_h, target_w = self.H, self.W
        h0, w0 = arr.shape[:2]
        if w0 != target_w:
            arr = cv2.resize(arr, (target_w, round(h0 * target_w / w0)), interpolation=interp)
            h0 = arr.shape[0]
        if h0 != target_h:
            top = (h0 - target_h) // 2
            arr = arr[top:top + target_h]
        return arr

    def _ref_image(self, r):
        return self._resize(r["image"], cv2.INTER_AREA)

    def _ref_depth(self, name):
        if name not in self._depth_cache:
            d = np.load(os.path.join(self.data_dir, "depth", f"{name}.npy")).astype(np.float32)
            self._depth_cache[name] = self._resize(d, cv2.INTER_LINEAR)
        return self._depth_cache[name]

    # ----- retrieval --------------------------------------------------------
    def _closest(self, w2c, forward_only=False):
        """Closest reference by position, then best rotation match.

        Treat every reference whose position is within ``group_tol`` metres of the
        nearest candidate as "the same spot", and among those pick the one whose
        viewing direction best matches ``w2c``. This replaces the old grid
        quantization (which split frames straddling a cell boundary into different
        spots); a distance window has no boundary artifacts. ``forward_only``
        restricts candidates to those ahead of the query (used for the look-ahead).
        """
        pos = self._cam_pos(w2c)
        R = w2c[:3, :3]
        cand = np.arange(len(self.refs))
        if forward_only:
            fwd = R.T @ np.array([0.0, 0.0, 1.0])
            ahead = cand[(self.ref_pos - pos) @ fwd > 0]
            cand = ahead if len(ahead) else cand
        dists = np.linalg.norm(self.ref_pos[cand] - pos, axis=1)
        near = cand[dists <= dists.min() + self.group_tol]
        i = min(near, key=lambda j: self._geodesic(R, self.refs[j]["extrinsic"][:3, :3]))
        return self.refs[i]

    # ----- trajectory -------------------------------------------------------
    # Turn-timing fractions of the *active* clip (everything before the look-ahead
    # padding). Tuned to reproduce the OOD reference set at the default 2-chunk /
    # 149-frame config: single turn at f~35; S-curve turns at f~20 and f~87.
    _SINGLE_TURN_START = 0.235   # straight_*90: where the single turn begins
    _S_TURN_START_1 = 0.134      # *90_*90: first turn begins
    _S_TURN_START_2 = 0.584      # *90_*90: second (opposite) turn begins

    def _heading_schedule(self, T, n_active):
        """Per-frame cumulative heading offset (radians) about world-up.

        Convention: left = +, right = -. ``*90`` turns ramp at ``turn_rate``
        deg/frame; ``smooth_*`` turns spread ``turn_deg`` continuously across the
        active clip. Beyond ``n_active`` the heading is held (the trailing frames
        exist only so the look-ahead can retrieve a forward reference).
        """
        A = np.deg2rad(self.turn_deg)
        r = max(np.deg2rad(self.turn_rate), 1e-6)
        d = np.zeros(T)

        def ramp(start, total):  # add a `total`-rad turn starting at frame `start`
            n_ramp = max(1, int(round(abs(total) / r)))
            for k in range(n_ramp):
                if 0 <= start + k < T:
                    d[start + k] += total / n_ramp

        traj = self.trajectory
        if traj == "straight":
            pass  # no turn; heading held constant (d stays zero)
        elif traj in ("smooth_left", "smooth_right"):
            s = 1.0 if traj == "smooth_left" else -1.0
            d[1:n_active] += s * A / max(n_active - 1, 1)
        elif traj in ("straight_left90", "straight_right90"):
            s = 1.0 if traj.endswith("left90") else -1.0
            ramp(int(round(self._SINGLE_TURN_START * n_active)), s * A)
        elif traj in ("left90_right90", "right90_left90"):
            s = 1.0 if traj.startswith("left90") else -1.0
            ramp(int(round(self._S_TURN_START_1 * n_active)), s * A)
            ramp(int(round(self._S_TURN_START_2 * n_active)), -s * A)
        else:
            raise ValueError(f"unknown trajectory {traj!r}")
        return np.cumsum(d)

    def _build_trajectory(self, start_frame):
        # start_frame: an image name (str) OR an index into the sorted frames
        if isinstance(start_frame, str):
            match = [i for i, r in enumerate(self.refs) if r["name"] == start_frame]
            assert match, f"start_frame '{start_frame}' not found in {self.data_dir}/images"
            self._start_i = match[0]
        else:
            self._start_i = int(start_frame) % len(self.refs)
        w2c0 = np.eye(4); w2c0[:3, :] = self.refs[self._start_i]["extrinsic"]
        c2w0 = np.linalg.inv(w2c0)
        R0, p0 = c2w0[:3, :3], c2w0[:3, 3].copy()
        up = -R0[:, 1]; up /= np.linalg.norm(up) + 1e-8   # RDF: -down = world up
        K = self.refs[self._start_i]["intrinsic"]

        # n_active = rendered frames; the +la_offset+30 tail is look-ahead padding
        n_active = self.stride * (self.num_chunks - 1) + self.num_main_frames
        T = n_active + self.la_offset + 30
        theta = self._heading_schedule(T, n_active)

        def axis_rot(a, ang):
            a = a / (np.linalg.norm(a) + 1e-8)
            X = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            return np.eye(3) + np.sin(ang) * X + (1 - np.cos(ang)) * (X @ X)

        ext, p = [], p0.copy()
        for t in range(T):
            Rt = axis_rot(up, theta[t]) @ R0
            if t > 0:
                p = p + self.speed * Rt[:, 2]
            c2w = np.eye(4); c2w[:3, :3] = Rt; c2w[:3, 3] = p
            ext.append(np.linalg.inv(c2w)[:3, :].astype(np.float32))
        self.track = np.stack(ext)          # (T, 3, 4) w2c
        self.track_K = K

    # ----- dataset ----------------------------------------------------------
    def __len__(self):
        return self.num_chunks

    def __getitem__(self, chunk_idx):
        H, W, T = self.H, self.W, self.num_main_frames
        track_start = chunk_idx * self.stride
        n = len(self.track)
        is_first = chunk_idx == 0
        num_cond = 1 if is_first else self.overlap_frames

        def w2c4(e):
            m = np.eye(4, dtype=np.float32); m[:3, :] = e; return m

        # forward-warp ingredients: per target pose, retrieve nearest ref as the source
        sf, sd, sw, si, tw = [], [], [], [], []
        for i in range(T):
            tw2c = self.track[min(track_start + i, n - 1)]
            r = self._closest(tw2c)
            img = self._ref_image(r).astype(np.float32) / 127.5 - 1.0   # [-1, 1]
            dep = self._ref_depth(r["name"])
            sf.append(torch.from_numpy(img).permute(2, 0, 1))
            sd.append(torch.from_numpy(dep)[None])
            sw.append(torch.from_numpy(w2c4(r["extrinsic"])))
            sh, swid = r["image"].shape[:2]
            si.append(torch.from_numpy(self._scale_K(r["intrinsic"], swid, sh).astype(np.float32)))
            tw.append(torch.from_numpy(w2c4(tw2c)))

        # video init: chunk-0 frame 0 = the start image (conditional frame)
        video = np.zeros((T, H, W, 3), dtype=np.uint8)
        if is_first:
            video[0] = self._ref_image(self.refs[self._start_i])
        video_t = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous().to(torch.uint8)

        # reference frames: retrieve along the trajectory at evenly-spaced anchors
        anchors = np.linspace(0, T - 1, self.num_ref_views + 1, dtype=int)[1:]
        ref_imgs, ref_ext, ref_K, ref_t, ref_mask = [], [], [], [], []
        seen = set()
        for a in anchors:
            r = self._closest(self.track[min(track_start + a, n - 1)])
            sh, swid = r["image"].shape[:2]
            ref_ext.append(r["extrinsic"]); ref_K.append(self._scale_K(r["intrinsic"], swid, sh))
            ref_t.append(min(int(a) // 4, 19))
            if r["name"] in seen:
                ref_imgs.append(np.zeros((H, W, 3), np.uint8)); ref_mask.append(0.0)
            else:
                seen.add(r["name"]); ref_imgs.append(self._ref_image(r)); ref_mask.append(1.0)

        # look-ahead: forward-closest ref from the chunk's last pose
        la_w2c = self.track[min(track_start + T - 1, n - 1)]
        la_r = self._closest(la_w2c, forward_only=True)
        la_sh, la_sw = la_r["image"].shape[:2]
        la_img = self._ref_image(la_r)

        # camera params relative to this chunk's first pose
        origin = self.track[min(track_start, n - 1)]
        main_w2c = np.stack([self.track[min(track_start + i, n - 1)] for i in range(T)])
        main_ext = make_relative_extrinsics(main_w2c, origin)
        ref_ext_rel = make_relative_extrinsics(np.stack(ref_ext), origin)
        la_ext_rel = make_relative_extrinsics(la_r["extrinsic"][None], origin)
        main_K = np.tile(self._scale_K(self.track_K, *self._track_src_wh()), (T, 1, 1))

        return {
            "video": video_t,
            "src_frames": torch.stack(sf), "src_depths": torch.stack(sd),
            "src_w2cs": torch.stack(sw), "src_intrinsics": torch.stack(si),
            "target_w2cs": torch.stack(tw), "mask_b1hw": torch.ones(1, H, W),
            "ref_frames": torch.from_numpy(np.stack(ref_imgs)).permute(0, 3, 1, 2).contiguous().to(torch.uint8),
            "ref_mask": torch.tensor(ref_mask, dtype=torch.float32),
            "ref_time_indices": torch.tensor(ref_t, dtype=torch.long),
            "la_frame": torch.from_numpy(la_img).permute(2, 0, 1)[None].contiguous().to(torch.uint8),
            "la_time_indices": torch.tensor([(T - 1 + self.la_offset) // 4], dtype=torch.long),
            "la_mask": torch.tensor([1.0]),
            "main_extrinsics": torch.from_numpy(main_ext).float(),
            "main_intrinsics": torch.from_numpy(main_K.astype(np.float32)).float(),
            "ref_extrinsics": torch.from_numpy(ref_ext_rel).float(),
            "ref_intrinsics": torch.from_numpy(np.stack(ref_K).astype(np.float32)).float(),
            "la_extrinsics": torch.from_numpy(la_ext_rel).float(),
            "la_intrinsics": torch.from_numpy(self._scale_K(la_r["intrinsic"], la_sw, la_sh)[None].astype(np.float32)).float(),
            "cam_mask": torch.tensor(1.0),
            "ai_caption": self.prompt,
            "fps": torch.tensor(self.fps),
            "image_size": torch.tensor([H, W, H, W]),
            "num_frames": T,
            "padding_mask": torch.zeros(1, H, W),
            "num_conditional_frames": num_cond,
            "chunk_idx": chunk_idx,
            "track_start": track_start,
        }

    def _track_src_wh(self):
        # the trajectory's intrinsic is the start view's K (image render res)
        sh, sw = self.refs[self._start_i]["image"].shape[:2]
        return sw, sh
