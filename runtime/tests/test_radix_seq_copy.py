"""Radix seq-copy HTTP client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from runtime.kv.radix_prefix_share import RadixSharePlan
from runtime.kv.radix_seq_copy import (
    copy_sequence_prefix_subprocess,
    execute_radix_share_plan,
)


def test_copy_sequence_prefix_subprocess_ok():
    plan = RadixSharePlan(
        source_slot=2,
        target_slot=5,
        copy_tokens=512,
        matched_blocks=1,
        tail_block_hash=None,
    )
    payload = json.dumps({"ok": True, "pos_end": 512}).encode()

    class FakeResp:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        assert execute_radix_share_plan(plan, subprocess_base_url="http://127.0.0.1:8082")


def test_copy_sequence_prefix_subprocess_404():
    def _raise(*args, **kwargs):
        raise HTTPError("http://x/kv/seq-copy", 404, "missing", None, None)

    with patch("urllib.request.urlopen", side_effect=_raise):
        assert copy_sequence_prefix_subprocess(
            "http://127.0.0.1:8082",
            source_slot=1,
            target_slot=2,
            pos_end=512,
        ) is False


def test_copy_sequence_prefix_inprocess():
    plan = RadixSharePlan(
        source_slot=1,
        target_slot=3,
        copy_tokens=256,
        matched_blocks=1,
        tail_block_hash=None,
    )
    lib = MagicMock()
    lib.llama_get_memory.return_value = MagicMock()
    ctx = MagicMock()
    with patch(
        "runtime.kv.radix_seq_copy.copy_sequence_prefix_inprocess",
        return_value=True,
    ) as mock_copy:
        assert execute_radix_share_plan(plan, inprocess_lib=lib, inprocess_ctx=ctx)
        mock_copy.assert_called_once()
