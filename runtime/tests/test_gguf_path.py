from pathlib import Path

from runtime.server.gguf_path import peek_gguf_path, pop_gguf_path


def test_pop_gguf_path_removes_key():
    opts = {"gguf": "/data/models/llama.gguf", "temperature": 0.7}
    p = pop_gguf_path(opts)
    assert p == Path("/data/models/llama.gguf")
    assert "gguf" not in opts
    assert opts["temperature"] == 0.7


def test_pop_gguf_path_model_path_alias():
    opts = {"model_path": "/tmp/x.gguf"}
    p = pop_gguf_path(opts)
    assert p == Path("/tmp/x.gguf")


def test_pop_gguf_path_empty():
    assert pop_gguf_path({}) is None


def test_peek_gguf_path_leaves_options():
    opts = {"gguf": "/tmp/m.gguf"}
    assert peek_gguf_path(opts) == Path("/tmp/m.gguf").resolve()
    assert "gguf" in opts
