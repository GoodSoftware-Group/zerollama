from pathlib import Path

from runtime.config import RuntimeConfig
from runtime.speculative import SpeculativeConfig, llama_server_args_for, resolve_method


def test_resolve_method_aliases():
    assert resolve_method("ngram") == "ngram-simple"
    assert resolve_method("dflash") == "draft-eagle3"


def test_ngram_llama_args():
    spec = SpeculativeConfig(method="ngram", ngram_size_n=10, ngram_size_m=40)
    args = llama_server_args_for(spec)
    assert args[:2] == ["--spec-type", "ngram-simple"]
    assert "--spec-ngram-simple-size-n" in args


def test_draft_requires_model():
    spec = SpeculativeConfig(method="draft-simple", draft_model=None)
    try:
        llama_server_args_for(spec)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "draft_model" in str(e)


def test_load_ngram_yaml():
    path = Path(__file__).resolve().parents[1] / "configs" / "dual_4090_ngram.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.speculative.method == "ngram"
    args = cfg.llama_server_args()
    assert "ngram-simple" in args
