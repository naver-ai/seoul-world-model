# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
"""Teaser experiment: regenerate the released teaser video from the sample folder.

Same model, checkpoint and sampler setup as ``predict2_generate``; only the dataset
differs (``TeaserDataset``: the camera path, warp condition and reference images of the
teaser, read from the sample folder). Configure it with environment variables:

    DATA_DIR   the sample folder (with teaser.json)
    PROMPT     literal text prompt (default: the teaser's prompt in teaser.json)

Register name: predict2_teaser   (wrapper: TEASER=1 bash sh/inference/generate.sh)
"""
import os
from datetime import datetime

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.teaser_dataset import TeaserDataset
from cosmos_predict2.experiments.base.generate import predict2_generate

dataset = L(TeaserDataset)(
    data_dir=os.environ.get("DATA_DIR", ""),
    fixed_caption=os.environ.get("PROMPT") or None,
    video_size=(704, 1280),
)

dataloader = L(get_generic_dataloader)(
    dataset=dataset,
    sampler=L(get_sampler)(dataset=dataset),
    batch_size=1, drop_last=False, num_workers=0, pin_memory=True,
)

predict2_teaser = dict(
    predict2_generate,
    job=dict(project="cosmos_predict_v2p5", group="teaser",
             name=f"teaser_{datetime.now().strftime('%Y%m%d_%H%M%S')}", wandb_mode="offline"),
    dataloader_train=dataloader,
    dataloader_val=dataloader,
)

ConfigStore.instance().store(group="experiment", package="_global_",
                             name="predict2_teaser", node=predict2_teaser)
