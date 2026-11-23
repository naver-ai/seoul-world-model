# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Camera transform utility functions for warping operations."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def extrinsic_3x4_to_4x4(ext_3x4: np.ndarray) -> np.ndarray:
    """Convert 3x4 extrinsic to 4x4 by adding [0, 0, 0, 1] row."""
    if ext_3x4.ndim == 2:
        ext_4x4 = np.eye(4, dtype=ext_3x4.dtype)
        ext_4x4[:3, :] = ext_3x4
        return ext_4x4
    else:
        N = ext_3x4.shape[0]
        ext_4x4 = np.tile(np.eye(4, dtype=ext_3x4.dtype), (N, 1, 1))
        ext_4x4[:, :3, :] = ext_3x4
        return ext_4x4


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert world-to-camera matrix to camera-to-world matrix."""
    return np.linalg.inv(w2c)


def interpolate_extrinsics(ext1: np.ndarray, ext2: np.ndarray, num_interp: int) -> np.ndarray:
    """Interpolate between two 4x4 extrinsic matrices using SLERP for rotation."""
    R1, t1 = ext1[:3, :3], ext1[:3, 3]
    R2, t2 = ext2[:3, :3], ext2[:3, 3]

    rot1 = Rotation.from_matrix(R1)
    rot2 = Rotation.from_matrix(R2)

    key_times = [0, 1]
    key_rots = Rotation.concatenate([rot1, rot2])
    slerp = Slerp(key_times, key_rots)

    times = np.linspace(0, 1, num_interp + 2)[1:-1]

    interp_rots = slerp(times)
    interp_R = interp_rots.as_matrix()
    interp_t = np.array([t1 + t * (t2 - t1) for t in times])

    interpolated = np.tile(np.eye(4, dtype=ext1.dtype), (num_interp, 1, 1))
    interpolated[:, :3, :3] = interp_R
    interpolated[:, :3, 3] = interp_t

    return interpolated
