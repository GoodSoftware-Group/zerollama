"""Resident ASR helpers that stay out of the GGUF runtime process."""

from runtime.asr.qwen3_asr import Ear, model_dir_ready

__all__ = ["Ear", "model_dir_ready"]
