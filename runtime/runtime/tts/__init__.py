"""Resident TTS helpers that stay out of the GGUF runtime process."""

from runtime.tts.kokoro import Speaker, lang_for_voice, resolve_lang

__all__ = ["Speaker", "lang_for_voice", "resolve_lang"]
