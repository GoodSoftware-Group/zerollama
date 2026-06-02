from unittest.mock import patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.gpu.admission import AdmissionRejected
from runtime.gpu.inference_policy import InferencePriority
from runtime.worker.llama_server import LlamaServerError


@pytest.fixture
def engine(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    return InferenceEngine(cfg)


def test_admit_one_leaves_no_orphan_on_tick_failure(engine):
    def reject(_req):
        raise AdmissionRejected("gpu full")

    with patch.object(engine, "_check_admit_policy", return_value=InferencePriority.NORMAL):
        with patch.object(engine.loop, "tick", return_value=[]):
            with patch.object(engine, "_loop_vram_check", reject):
                with pytest.raises(LlamaServerError, match="could not admit"):
                    engine._admit_one("hi", 8)
    assert len(engine.scheduler.waiting) == 0


def test_generate_batch_cancels_unadmitted(engine):
    with patch.object(
        engine, "_check_admit_policy", return_value=InferencePriority.LOW
    ):
        with patch.object(
            engine.loop,
            "tick",
            return_value=[],
        ):
            with pytest.raises(LlamaServerError, match="could not admit batch"):
                engine.generate_batch(["a", "b"], max_admit=2)
    assert len(engine.scheduler.waiting) == 0
