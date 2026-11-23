# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
"""Trajectory generation experiment.

Drive a world model along a user-defined camera trajectory, conditioned on a set
of posed reference frames (image + depth + intrinsics + extrinsics) via retrieval
and GPU forward-warping. Configure it with environment variables so a one-line
script can drive it:

    DATA_DIR      folder of posed frames (images/ depth/ intrinsics/ extrinsics/)
    TRAJECTORY    straight_left90 | straight_right90 | left90_right90 |
                  right90_left90 | smooth_left | smooth_right
    START_FRAME   index of the forward-facing frame to start from
    PROMPT        literal text prompt (wins if set)
    PROMPT_KEY    pick a named prompt from PROMPT_FILE (a {key: prompt} JSON)
    PROMPT_FILE   prompt library path (default sh/inference/teaser_prompts.json)
    SPEED         forward step per frame, in meters
    TURN_DEG      magnitude of each turn, in degrees
    TURN_RATE     turn speed for the `*90` turns, in degrees per frame
    GROUP_TOL     retrieval radius in meters (refs within this distance of the nearest
                  count as one spot; the closest viewing direction among them wins)

Register name: predict2_generate
"""
import json
import os
from datetime import datetime

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.trajectory_dataset import TrajectoryDataset
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelSize

DEFAULT_CHECKPOINT_2B = MODEL_CHECKPOINTS[ModelKey(post_trained=False, size=ModelSize._2B)]

_DEFAULT_PROMPT = (
    "A driving video captured from the front-facing camera of a vehicle moving "
    "through a city, with buildings, intersections, traffic and clear daytime lighting."
)
# Default location of the named-prompt library (overridable with PROMPT_FILE).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_PROMPT_FILE = os.path.join(_REPO_ROOT, "sh", "inference", "teaser_prompts.json")


def _resolve_prompt() -> str:
    """Resolve the text prompt, in priority order:

    1. PROMPT      — a literal prompt string (wins if set)
    2. PROMPT_KEY  — pick a named prompt from a JSON file of {key: prompt}
                     (file given by PROMPT_FILE, default sh/inference/teaser_prompts.json)
    3. the built-in default city-driving prompt
    """
    literal = os.environ.get("PROMPT")
    if literal:
        return literal
    key = os.environ.get("PROMPT_KEY")
    if key:
        path = os.environ.get("PROMPT_FILE", _DEFAULT_PROMPT_FILE)
        with open(path) as f:
            prompts = json.load(f)
        if key not in prompts:
            raise KeyError(
                f"PROMPT_KEY '{key}' not found in {path}. Available: {sorted(prompts)}"
            )
        return prompts[key]
    return _DEFAULT_PROMPT


_PROMPT = _resolve_prompt()

_start = os.environ.get("START_FRAME", "0")           # an image name, or an index
dataset = L(TrajectoryDataset)(
    data_dir=os.environ.get("DATA_DIR", ""),
    trajectory=os.environ.get("TRAJECTORY", "smooth_left"),
    start_frame=int(_start) if _start.lstrip("-").isdigit() else _start,
    prompt=_PROMPT,
    speed=float(os.environ.get("SPEED", "0.3")),
    turn_deg=float(os.environ.get("TURN_DEG", "90")),
    turn_rate=float(os.environ.get("TURN_RATE", "4")),
    group_tol=float(os.environ.get("GROUP_TOL", "2.0")),
    num_chunks=2,
    video_size=(704, 1280),
)

dataloader = L(get_generic_dataloader)(
    dataset=dataset,
    sampler=L(get_sampler)(dataset=dataset),
    # num_workers=0: load in the main process. The inference dataset is tiny (a few
    # chunks), so worker processes give no speedup but DO pass large frame tensors
    # through /dev/shm — which exhausts a container's (often small, e.g. 2 GB) shared
    # memory when several single-GPU runs execute concurrently, killing the dataloader
    # with a Bus error (SIGBUS). Loading in-process makes concurrent multi-GPU runs robust.
    batch_size=1, drop_last=False, num_workers=0, pin_memory=True,
)

_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
    {"override /model": "fsdp_rectified_flow_warp_ref_LA"},
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    {"override /net": "cosmos_v1_2B_warp_ref_LA_cam"},
    "_self_",
]
_checkpoint_config = dict(
    # Placeholder default; always overridden at run time via checkpoint.load_path=$CKPT.
    # Kept as the raw URI (not resolved) so building the config never triggers a
    # download of the gated base checkpoint.
    load_path=DEFAULT_CHECKPOINT_2B.s3.uri,
    strict_resume=False, load_ema_to_reg=False,
    load_from_object_store=dict(enabled=False), save_to_object_store=dict(enabled=False),
    save_iter=1000, keep_last_n_checkpoints=1, save_only_trainable=False, load_training_state=False,
)
_optimizer = dict(lr=2 ** (-14.5), weight_decay=0.001)
_scheduler = dict(f_max=[0.5], f_min=[0.2], warm_up_steps=[2_000], cycle_lengths=[100000])
_trainer = dict(
    run_validation=True, validation_iter=1, logging_iter=1, max_iter=0, max_val_iter=1,
    straggler_detection=dict(enabled=False),
    callbacks=dict(
        heart_beat=dict(save_s3=False), iter_speed=dict(hit_thres=200, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=20000, save_s3=False),
        every_n_sample_ema=dict(every_n=20000, save_s3=False),
        wandb=dict(save_s3=False), wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=1, guidance=[7.0], is_ema=False, save_s3=False,
            do_x0_prediction=False, use_negative_prompt=True, fps=10),
    ),
)
_model_config = dict(
    config=dict(
        min_num_conditional_frames=1, max_num_conditional_frames=2,
        conditional_frames_probs={1: 0.5, 2: 0.5}, conditional_frame_timestep=-1.0,
        conditioning_strategy="frame_replace", denoise_replace_gt_frames=True,
        state_t=20, la_dropout=0.0),
    _recursive_=False,
)

predict2_generate = dict(
    defaults=_defaults,
    job=dict(project="cosmos_predict_v2p5", group="generate",
             name=f"generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}", wandb_mode="offline"),
    dataloader_train=dataloader,
    dataloader_val=dataloader,
    checkpoint=_checkpoint_config,
    optimizer=_optimizer, scheduler=_scheduler, trainer=_trainer, model=_model_config,
    model_parallel=dict(context_parallel_size=1),
)

ConfigStore.instance().store(group="experiment", package="_global_",
                             name="predict2_generate", node=predict2_generate)
