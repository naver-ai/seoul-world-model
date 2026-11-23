# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Callback to save training input videos in SMOKE mode for debugging."""

import os
import torch

from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video


class SaveTrainingInputs(Callback):
    """Save training input videos as MP4 files during SMOKE mode."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and SMOKE
        self.rank = distributed.get_rank()

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not self.enabled:
            return
        
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/TrainingInputs"
        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if not self.enabled:
            return

        # Only save from rank 0 to avoid duplicates
        if self.rank != 0:
            return

        # Get the raw video data before any processing
        raw_data, _, _ = model.get_data_and_condition(data_batch)

        # raw_data is in [-1, 1] range, convert to [0, 1] for save_img_or_video
        # Shape: [B, C, T, H, W]
        video_sample = raw_data[0].float().cpu()  # First sample: [C, T, H, W]
        video_sample = (video_sample + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
        video_sample = video_sample.clamp(0, 1)

        # Save to MP4
        save_path = os.path.join(self.local_dir, f"iter_{iteration:04d}_input")
        save_img_or_video(video_sample, save_path, fps=16)

        # Also save the caption if available
        if "ai_caption" in data_batch:
            caption = data_batch["ai_caption"][0] if isinstance(data_batch["ai_caption"], list) else str(data_batch["ai_caption"])
            caption_path = os.path.join(self.local_dir, f"iter_{iteration:04d}_caption.txt")
            with open(caption_path, "w") as f:
                f.write(caption)
