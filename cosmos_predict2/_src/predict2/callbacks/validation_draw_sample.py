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

import math
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange, repeat
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.callbacks.inference_visualization import save_inference_comparison

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    """
    Resize the image to the given size. This is done so that wandb can display the image correctly.
    """
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class ValidationDrawSample(Callback):
    """
    This callback sample condition inputs from validation data, run inference and save the results to wandb and s3.

    Args:
        n_samples (int): The number of samples to run inference on.
        n_viz_sample (int, optional): for each batch, min(n_viz_sample, batch_size) samples will be saved to wandb. Defaults to 3.
        n_sample_to_save (int, optional): number of samples to save. The actual number of samples to save is min(n_sample_to_save, data parallel instances). Defaults to 128.
        num_sampling_step (int, optional): number of sampling steps. Defaults to 35.
        guidance (List[float], optional): guidance scale. Defaults to [0.0, 3.0, 7.0].
        do_x0_prediction (bool, optional): whether to do x0 prediction. Defaults to True.
        n_sigmas_for_x0_prediction (int, optional): number of sigmas to use for x0 prediction. Defaults to 4.
        save_s3 (bool, optional): whether to save to s3. Defaults to False.
        is_ema (bool, optional): whether the callback is run for ema model. Defaults to False.
        use_negative_prompt (bool, optional): whether to use negative prompt. Defaults to False.
        fps (int, optional): frames per second when saving the video. Defaults to 16.
    """

    def __init__(
        self,
        n_samples: int,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [0.0, 3.0, 7.0],
        do_x0_prediction: bool = True,
        n_sigmas_for_x0_prediction: int = 4,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        prompt_type: str = "t5_xxl",
        fps: int = 16,
        run_at_start: bool = False,
        barrier_after_run: bool = True,
        seed: Optional[int] = None,
        inference_mode: bool = False,
        uncond_ref_dropout: bool = False,
        no_ref_images: bool = False,
        ref_guidance: float = 0.0,
    ):
        # s3: # files: min(n_sample_to_save, data instance)  # per file: min(batch_size, n_viz_sample)
        # wandb: 1 file, # per file: min(batch_size, n_viz_sample)
        self.barrier_after_run = barrier_after_run
        self.run_at_start = run_at_start

        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.do_x0_prediction = do_x0_prediction
        self.n_sigmas_for_x0_prediction = n_sigmas_for_x0_prediction
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.prompt_type = prompt_type
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self.n_samples = n_samples
        self.sample_counter = 0
        # If seed is None, use time-based random seed for different results each run
        self.seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        # Inference mode: save comparison visualizations with trajectory and GT frames
        self.inference_mode = inference_mode
        # CFG on reference images: use zero ref_latents for unconditional path
        self.uncond_ref_dropout = uncond_ref_dropout
        # No reference images: use zero ref_latents for BOTH cond and uncond
        self.no_ref_images = no_ref_images
        # Decomposed CFG: separate ref guidance scale
        self.ref_guidance = ref_guidance

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        self.wandb_online = self.config.job.wandb_mode == "online"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")
            log.info(f"Callback: seed: {self.seed}")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.use_negative_prompt:
            pass
            # if self.prompt_type == "t5_xxl":
            #     self.negative_prompt_data = easy_io.load(
            #         "s3://bucket/edify_video/v4/validation/item_dataset/negative_prompt/000000.pkl"
            #     )
            # elif self.prompt_type == "umt5_xxl":
            #     self.negative_prompt_data = easy_io.load(
            #         "s3://bucket/edify_video/v4/validation/item_dataset/negative_prompt/umt5_neg.pt"
            #     )
            # else:
            #     raise ValueError(f"Invalid prompt type: {self.prompt_type}")

    @misc.timer("ValidationDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, model, data_batch, iteration):
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)

        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        _, condition, x0, _ = model.broadcast_split_for_model_parallelsim(None, condition, x0, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(
                math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_sigmas_for_x0_prediction + 1
            )[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        # Use configurable seed + sample_counter for unique noise per sample
        sample_seed = self.seed + self.sample_counter + self.rank
        generator.manual_seed(sample_seed)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)

            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        local_path = self.run_save(to_show, batch_size, iteration, "x0")
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        self.sample_counter = 0

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        trainer = self.trainer

        past_samples = self.sample_counter * distributed.get_world_size()
        current_samples = past_samples + self.rank

        should_run = (iteration > 0 or self.run_at_start) and past_samples < self.n_samples
        should_save = current_samples < self.n_samples

        if should_run:
            save_log = "" if should_save else " (skipping save/log, running only for GPU synchronization)"
            log.debug(
                f"Callback {self.__class__.__name__} fired on validation_step_end step {iteration} [{current_samples + 1}/{self.n_samples}]{save_log}",
                rank0_only=False,
            )
            self.run_sample(trainer, model, data_batch, iteration, should_save)
            self.sample_counter += 1
            log.debug(
                f"Callback {self.__class__.__name__} finished on validation_step_end step {iteration} [{current_samples + 1}/{self.n_samples}]{save_log}",
                rank0_only=False,
            )

            if self.barrier_after_run:
                distributed.barrier()

    def run_sample(self, trainer, model, data_batch, iteration, should_save):
        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        log.debug("entering, every_n_impl", rank0_only=False)
        if self.do_x0_prediction:
            log.debug("entering, x0_pred", rank0_only=False)
            x0_save_dir, mse_loss, sigmas = self.x0_pred(
                model,
                data_batch,
                iteration,
            )
            log.debug("done, x0_pred", rank0_only=False)
            if self.save_s3 and self.rank == 0:
                easy_io.dump(
                    {
                        "mse_loss": mse_loss.tolist(),
                        "sigmas": sigmas.tolist(),
                        "iteration": iteration,
                    },
                    f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                )

        log.debug("entering, sample", rank0_only=False)
        sample_save_dir = self.sample(
            model,
            data_batch,
            iteration,
            should_save,
        )
        log.debug("done, sample", rank0_only=False)

        log.debug("waiting for all ranks to finish", rank0_only=False)
        dist.barrier()
        if self.rank == 0 and wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.do_x0_prediction:
                # convert mse_loss to a dict and log
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            # Log metrics only (videos saved to disk, not wandb)
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("ValidationDrawSample: sample")
    def sample(self, model, data_batch, iteration, should_save):
        tag = "ema" if self.is_ema else "reg"

        # Obtain text embeddings online
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        raw_data, x0, _ = model.get_data_and_condition(data_batch)

        # Debug: print x0 (input latent) range
        if self.rank == 0:
            log.info(f"[Input] x0 latent - min={x0.min().item():.4f}, max={x0.max().item():.4f}, mean={x0.mean().item():.4f}")

        if self.use_negative_prompt:
            batch_size = x0.shape[0]
            negative_prompt_emb = model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [_DEFAULT_NEGATIVE_PROMPT], "images": None},
                    input_caption_key="ai_caption",
                )
            if negative_prompt_emb.shape != data_batch["t5_text_embeddings"].shape:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    repeat(
                        negative_prompt_emb,
                        "... -> b ...",
                        b=batch_size,
                    ),
                    **model.tensor_kwargs,
                )
            else:
                data_batch["neg_t5_text_embeddings"] = misc.to(
                    negative_prompt_emb,
                    **model.tensor_kwargs,
                )

            assert data_batch["neg_t5_text_embeddings"].shape == data_batch["t5_text_embeddings"].shape, (
                f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            )
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]

        to_show = []
        # Use configurable seed + sample_counter for unique generation per sample
        sample_seed = self.seed + self.sample_counter + self.rank
        for guidance in self.guidance:
            sample_latent = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=self.num_sampling_step,
                is_negative_prompt=self.use_negative_prompt,
                seed=sample_seed,
                uncond_ref_dropout=self.uncond_ref_dropout,
                no_ref_images=self.no_ref_images,
                ref_guidance=self.ref_guidance,
            )

            # Debug: print latent value range BEFORE decoding
            if self.rank == 0:
                log.info(f"[Guidance {guidance}] Generated latent - min={sample_latent.min().item():.4f}, max={sample_latent.max().item():.4f}, mean={sample_latent.mean().item():.4f}")
                # Check if generated latent is the same as input x0
                if torch.allclose(sample_latent, x0, rtol=1e-3):
                    log.warning(f"[Guidance {guidance}] WARNING: Generated latent is identical to input x0!")

            if hasattr(model, "decode"):
                # Debug: check tokenizer decoder before decoding
                if self.rank == 0 and guidance == self.guidance[0]:
                    log.info(f"Tokenizer type: {type(model.tokenizer)}")
                    log.info(f"Tokenizer model type: {type(model.tokenizer.model)}")
                    # Check if decoder exists and is loaded
                    if hasattr(model.tokenizer.model, 'model'):
                        decoder_model = model.tokenizer.model.model
                        log.info(f"Decoder model device: {next(decoder_model.parameters()).device}")
                        log.info(f"Decoder model dtype: {next(decoder_model.parameters()).dtype}")
                        # Check a sample parameter value
                        first_param = next(decoder_model.parameters())
                        log.info(f"First decoder param - min={first_param.min().item():.4f}, max={first_param.max().item():.4f}, mean={first_param.mean().item():.4f}")

                sample = model.decode(sample_latent)

                # Debug: print value range AFTER decoding
                if self.rank == 0:
                    log.info(f"[Guidance {guidance}] Decoded sample - min={sample.min().item():.4f}, max={sample.max().item():.4f}, mean={sample.mean().item():.4f}")
            else:
                sample = sample_latent

            to_show.append(sample.float().cpu())

        # Debug: print raw_data range
        if self.rank == 0:
            log.info(f"[Ground truth] Raw data - min={raw_data.min().item():.4f}, max={raw_data.max().item():.4f}, mean={raw_data.mean().item():.4f}")

        to_show.append(raw_data.float().cpu())

        # Add warped video if available (for warp conditioning models)
        if "warped_video" in data_batch:
            warped_video = data_batch["warped_video"]  # [B, C, T, H, W] uint8 [0, 255]
            # Normalize to [-1, 1] (same as other data)
            warped_video_normalized = warped_video.float() / 127.5 - 1.0
            if self.rank == 0:
                log.info(f"[Warped video] - min={warped_video_normalized.min().item():.4f}, max={warped_video_normalized.max().item():.4f}, mean={warped_video_normalized.mean().item():.4f}")
            to_show.append(warped_video_normalized.cpu())

        # Add reference images if available (for warp+ref conditioning models)
        if "ref_frames" in data_batch and "ref_mask" in data_batch:
            ref_frames = data_batch["ref_frames"]  # [B, max_ref, C, H, W] uint8 [0, 255]
            ref_mask = data_batch["ref_mask"]  # [B, max_ref] float
            B, max_ref, C, H, W = ref_frames.shape
            T = raw_data.shape[2]  # Get temporal dimension from raw_data (decoded video, not latent)

            # Normalize to [-1, 1]
            ref_frames_normalized = ref_frames.float() / 127.5 - 1.0  # [B, max_ref, C, H, W]

            num_valid_refs = ref_mask.sum(dim=1).mean().item()

            if "ref_time_indices" in data_batch:
                # Temporal switching: show one ref at full resolution, switching at each ref's temporal index
                ref_time_indices_val = data_batch["ref_time_indices"]  # [B, max_ref] long
                num_latent = (T - 1) // 4 + 1

                # Move to CPU for indexing
                ref_norm_cpu = ref_frames_normalized.cpu()
                ref_mask_cpu = ref_mask.cpu()
                ref_ti_cpu = ref_time_indices_val.cpu()

                ref_video = torch.full((B, C, T, H, W), -0.5, dtype=torch.float32)

                for b in range(B):
                    valid_mask = ref_mask_cpu[b] > 0
                    if not valid_mask.any():
                        continue

                    valid_idx = torch.where(valid_mask)[0]
                    valid_times = ref_ti_cpu[b][valid_idx].float()

                    # Sort by temporal index
                    sorted_order = valid_times.argsort()
                    valid_idx = valid_idx[sorted_order]
                    valid_times = valid_times[sorted_order]

                    # Map each pixel frame to latent time
                    latent_times = torch.linspace(0, num_latent - 1, T)

                    # Find ref to show: largest ref_time <= current latent_time
                    switch_idx = torch.searchsorted(valid_times, latent_times, side='right') - 1
                    switch_idx = switch_idx.clamp(0, len(valid_idx) - 1)

                    orig_idx = valid_idx[switch_idx]  # [T]
                    ref_selected = ref_norm_cpu[b][orig_idx]  # [T, C, H, W]
                    ref_video[b] = ref_selected.permute(1, 0, 2, 3)  # [C, T, H, W]

                if self.rank == 0:
                    log.info(f"[Reference images] - {num_valid_refs:.1f} valid refs, temporal switching at full resolution")
            else:
                # Fallback: tile references horizontally as a static image
                W_each = W // max_ref
                orig_aspect = W / H
                H_scaled = min(int(W_each / orig_aspect), H)
                y_offset = (H - H_scaled) // 2

                ref_tiled = torch.zeros(B, C, H, W, device=ref_frames.device, dtype=torch.float32)
                ref_tiled.fill_(-0.5)

                for i in range(max_ref):
                    ref_i = ref_frames_normalized[:, i]  # [B, C, H, W]
                    mask_i = ref_mask[:, i:i+1, None, None]  # [B, 1, 1, 1]
                    ref_i_resized = F.interpolate(ref_i, size=(H_scaled, W_each), mode='bilinear', align_corners=False)
                    x_start = i * W_each
                    x_end = (i + 1) * W_each
                    ref_tiled[:, :, y_offset:y_offset+H_scaled, x_start:x_end] = (
                        ref_i_resized * mask_i.to(ref_i_resized.device) +
                        (-0.5) * (1 - mask_i.to(ref_i_resized.device))
                    )

                ref_video = ref_tiled.unsqueeze(2).expand(-1, -1, T, -1, -1)  # [B, C, T, H, W]

                if self.rank == 0:
                    log.info(f"[Reference images] - {num_valid_refs:.1f} valid refs, tiled to {max_ref} slots")

            if self.rank == 0:
                log.info(f"[Reference images] - min={ref_video.min().item():.4f}, max={ref_video.max().item():.4f}")
            to_show.append(ref_video.cpu())

        # Add LA frame if available (for Lookahead conditioning models)
        if "la_frame" in data_batch:
            la_frame = data_batch["la_frame"]  # [B, 1, C, H, W] uint8
            B_la, n_la, C_la, H_la, W_la = la_frame.shape
            T = raw_data.shape[2]

            # Normalize to [-1, 1]
            la_frame_normalized = la_frame.float() / 127.5 - 1.0  # [B, 1, C, H, W]

            # Repeat the LA frame across all temporal frames for visualization
            la_video = la_frame_normalized[:, 0].unsqueeze(2).expand(-1, -1, T, -1, -1)  # [B, C, T, H, W]

            la_time_idx = data_batch.get("la_time_indices", None)
            if self.rank == 0:
                la_t_str = f"t={la_time_idx[0].item()}" if la_time_idx is not None else "unknown"
                log.info(f"[LA frame] - latent_time={la_t_str}, min={la_video.min().item():.4f}, max={la_video.max().item():.4f}")
            to_show.append(la_video.cpu())

        # Build row names for logging
        row_names = [f"guidance {g}" for g in self.guidance]
        row_names.append("ground truth")
        if "warped_video" in data_batch:
            row_names.append("warped video")
        if "ref_frames" in data_batch and "ref_mask" in data_batch:
            row_names.append("reference images")
        if "la_frame" in data_batch:
            row_names.append("LA frame")

        batch_size = x0.shape[0]

        # Save inference comparison video (generated | trajectory | GT) when in inference mode
        if self.inference_mode and should_save:
            # Get the first guidance sample (typically the one we care about)
            # to_show[0] is the first guidance sample, we want the decoded sample
            for guid_idx, guidance in enumerate(self.guidance):
                generated_sample = to_show[guid_idx]  # [B, C, T, H, W] in [-1, 1]
                # Convert from [-1, 1] to [0, 255] uint8
                generated_uint8 = ((generated_sample.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)

                # Save comparison for each batch item
                for b_idx in range(batch_size):
                    comparison_output_dir = os.path.join(self.local_dir, str(iteration), str(self.sample_counter), "comparison")
                    # Include rank in sample_idx to avoid file overwrites across GPUs
                    world_size = distributed.get_world_size()
                    unique_sample_idx = (self.sample_counter * world_size + self.rank) * batch_size + b_idx
                    save_inference_comparison(
                        generated_video=generated_uint8[b_idx],  # [C, T, H, W]
                        data_batch={k: v[b_idx] if isinstance(v, torch.Tensor) and v.ndim > 0 else v for k, v in data_batch.items()},
                        output_dir=comparison_output_dir,
                        sample_idx=unique_sample_idx,
                        guidance=guidance,
                        fps=self.fps,
                    )

        if should_save:
            local_path = self.run_save(to_show, batch_size, iteration, "sample", row_names=row_names)
            return local_path
        return None

    def run_save(self, to_show, batch_size, iteration, save_name, row_names=None) -> Optional[str]:
        # Stack and debug
        to_show_stacked = torch.stack(to_show, dim=0)  # [n, b, c, t, h, w]
        if self.rank == 0:
            log.info(f"Stacked (before norm) - min={to_show_stacked.min().item():.4f}, max={to_show_stacked.max().item():.4f}")

        to_show = (1.0 + to_show_stacked.clamp(-1, 1)) / 2.0  # convert [-1,1] to [0,1]

        if self.rank == 0:
            log.info(f"After normalization - min={to_show.min().item():.4f}, max={to_show.max().item():.4f}")
            # Log each row separately
            for i in range(to_show.shape[0]):
                if row_names is not None and i < len(row_names):
                    row_name = row_names[i]
                elif i < len(self.guidance):
                    row_name = f"guidance {self.guidance[i]}"
                elif i == len(self.guidance):
                    row_name = "ground truth"
                elif i == len(self.guidance) + 1:
                    row_name = "warped video"
                else:
                    row_name = f"row {i}"
                log.info(f"  Row {i} ({row_name}): min={to_show[i].min().item():.4f}, max={to_show[i].max().item():.4f}, mean={to_show[i].mean().item():.4f}")
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        save_path = os.path.join(str(iteration), str(self.sample_counter), save_name)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{save_path}/{self.rank}",
                fps=self.fps,
            )

        local_save_dir = os.path.join(self.local_dir, save_path)
        os.makedirs(local_save_dir, exist_ok=True)

        # Always save full video/image to local filesystem as MP4/PNG
        # Note: save_img_or_video automatically adds .mp4 extension
        local_video_path = os.path.join(local_save_dir, f"{self.rank}")
        save_img_or_video(
            rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
            local_video_path,
            fps=self.fps,
        )

        # Also save wandb preview JPG if wandb is online
        local_path = os.path.join(local_save_dir, f"{self.rank}.jpg")
        if self.wandb_online:
            if is_single_frame:  # image case
                to_show = rearrange(
                    to_show[:, :n_viz_sample],
                    "n b c t h w -> t c (n h) (b w)",
                )
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                # resize so that wandb can handle it
                torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
            else:
                to_show = to_show[:, :n_viz_sample]  # [n, b, c, 3, h, w]

                # resize 3 frames frames so that we can display them on wandb
                _T = to_show.shape[3]
                three_frames_list = [0, _T // 2, _T - 1]
                to_show = to_show[:, :, :, three_frames_list]
                log_image_size = 1024
                to_show = rearrange(
                    to_show,
                    "n b c t h w -> 1 c (n h) (b t w)",
                )

                # resize so that wandb can handle it
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                torchvision.utils.save_image(
                    resize_image(image_grid, log_image_size), local_path, nrow=1, scale_each=True
                )

        return local_save_dir
