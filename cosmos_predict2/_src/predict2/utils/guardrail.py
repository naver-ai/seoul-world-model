# Seoul World Model
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""NVIDIA Cosmos-Guardrail1 on every generation. Always on; there is no switch to turn it off.

The same checks Cosmos-Predict2.5 runs in its own inference, built with
``imaginaire/auxiliary/guardrail/common/presets.py`` unchanged:

* text guardrail (Blocklist + Qwen3Guard) on every prompt before it conditions the model. An
  unsafe prompt stops the run before anything is generated.
* video guardrail on the whole generated video: the video content safety filter (a video whose
  frames are more than 10 % unsafe is not saved), then RetinaFace face pixelation of the frames
  that are saved.

The video guardrail runs once the autoregressive loop is done, on uint8 copies of the frames that
are written to disk, so the chunks are still generated and conditioned on the unfiltered frames.

The guardrail models are fetched from Hugging Face on first use: nvidia/Cosmos-Guardrail1 (gated:
accept its license and log in), Qwen/Qwen3Guard-Gen-0.6B and google/siglip-so400m-patch14-384.
"""

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import log


class GuardrailBlocked(RuntimeError):
    """A prompt or a generated video was blocked by Cosmos-Guardrail1."""


class Guardrails:
    """Text and video guardrail runners of Cosmos-Guardrail1, as in Cosmos-Predict2.5 inference."""

    def __init__(self, offload_model_to_cpu: bool = True) -> None:
        # Importing the presets resolves the Cosmos-Guardrail1 checkpoint, so a missing download fails here.
        from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets

        self._presets = presets
        self._text = presets.create_text_guardrail_runner(offload_model_to_cpu=offload_model_to_cpu)
        self._video = presets.create_video_guardrail_runner(offload_model_to_cpu=offload_model_to_cpu)
        self._passed: set[str] = set()

    def check_prompt(self, prompt) -> None:
        """Raise GuardrailBlocked if a prompt (or any of a list of prompts) is unsafe.

        A prompt that passed once is not checked again (every chunk of a clip has the same caption).
        """
        for p in prompt if isinstance(prompt, (list, tuple)) else [prompt]:
            p = "" if p is None else str(p)
            if p in self._passed:
                continue
            if not self._presets.run_text_guardrail(p, self._text):
                raise GuardrailBlocked(f"Guardrail blocked the prompt: {p}")
            log.success("Passed guardrail on prompt")
            self._passed.add(p)

    def check_video(self, video: torch.Tensor) -> torch.Tensor:
        """uint8 [C, T, H, W] RGB -> the same video with faces pixelated; GuardrailBlocked if it is unsafe."""
        frames = video.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C), as Cosmos passes it
        processed = self._presets.run_video_guardrail(frames, self._video)
        if processed is None:
            raise GuardrailBlocked("Guardrail blocked the generated video")
        log.success("Passed guardrail on generated video")
        return torch.from_numpy(np.ascontiguousarray(processed)).permute(3, 0, 1, 2).to(torch.uint8).contiguous()
