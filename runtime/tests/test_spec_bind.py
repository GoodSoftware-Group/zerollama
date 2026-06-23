"""KVCacheSpec × page bind validation."""

from __future__ import annotations

import pytest

from runtime.kv.spec_bind import (
    assert_prefix_within_spec,
    prefix_within_spec,
    validate_decode_prefix,
)
from runtime.kv_cache_spec import KVCacheSpec
from runtime.worker.llama_server import LlamaServerError
from unittest.mock import MagicMock


def _swa_spec(window: int = 1024) -> KVCacheSpec:
    return KVCacheSpec(
        kind="sliding_window",
        effective_window=window,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )


def test_prefix_within_spec_allows_in_window():
    assert prefix_within_spec(_swa_spec(), seq_pos=0, prompt_tokens=512) is True


def test_prefix_within_spec_blocks_beyond_window():
    assert prefix_within_spec(_swa_spec(), seq_pos=1024, prompt_tokens=1) is False


def test_assert_prefix_within_spec_raises():
    with pytest.raises(LlamaServerError, match="effective_window"):
        assert_prefix_within_spec(_swa_spec(), seq_pos=900, prompt_tokens=200)


def test_validate_decode_prefix_skips_when_cache_disabled():
    req = MagicMock()
    req.prompt_cache_key = "k"
    validate_decode_prefix(
        _swa_spec(),
        req,
        decode_pos=2000,
        n_prompt=100,
        cache_prompt=False,
    )


def test_validate_decode_prefix_raises_on_swa_violation():
    req = MagicMock()
    req.prompt_cache_key = "k"
    with pytest.raises(LlamaServerError):
        validate_decode_prefix(
            _swa_spec(),
            req,
            decode_pos=900,
            n_prompt=200,
            cache_prompt=True,
        )


def test_resume_allowed_by_spec_blocks_swa_resume():
    from runtime.kv.spec_bind import resume_allowed_by_spec

    spec = _swa_spec(1024)
    assert resume_allowed_by_spec(
        spec,
        prompt_cache_key="k",
        seq_pos=0,
        n_prompt=512,
        cache_prompt=True,
    )
    assert not resume_allowed_by_spec(
        spec,
        prompt_cache_key="k",
        seq_pos=900,
        n_prompt=200,
        cache_prompt=True,
    )
