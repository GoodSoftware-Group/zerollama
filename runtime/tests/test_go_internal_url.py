import os

from runtime.go_internal_url import connectable_go_base_url


def test_connectable_maps_unspecified_bind(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_GO_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://0.0.0.0:8080")
    assert connectable_go_base_url() == "http://127.0.0.1:8080"


def test_connectable_prefers_explicit_go_url(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GO_URL", "http://127.0.0.1:19180")
    monkeypatch.setenv("OLLAMA_HOST", "http://0.0.0.0:8080")
    assert connectable_go_base_url() == "http://127.0.0.1:19180"


def test_connectable_host_port_without_scheme(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_GO_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:9000")
    assert connectable_go_base_url() == "http://127.0.0.1:9000"
