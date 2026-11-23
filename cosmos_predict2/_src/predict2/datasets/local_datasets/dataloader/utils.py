# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

import numpy as np


def make_relative_extrinsics(w2c_all_3x4: np.ndarray, w2c_ref_3x4: np.ndarray) -> np.ndarray:
    """Make w2c matrices relative to a reference frame (reference becomes identity).

    Args:
        w2c_all_3x4: [N, 3, 4] w2c matrices to transform.
        w2c_ref_3x4: [3, 4] reference w2c (will become identity after transform).

    Returns:
        [N, 3, 4] relative w2c matrices (float32).
    """
    N = w2c_all_3x4.shape[0]
    w2c_all = np.zeros((N, 4, 4), dtype=np.float64)
    w2c_all[:, :3, :] = w2c_all_3x4.astype(np.float64)
    w2c_all[:, 3, 3] = 1.0
    ref = np.eye(4, dtype=np.float64)
    ref[:3, :] = w2c_ref_3x4.astype(np.float64)
    c2w_ref = np.linalg.inv(ref)
    w2c_rel = w2c_all @ c2w_ref
    return w2c_rel[:, :3, :].astype(np.float32)
