from fastapi.testclient import TestClient

from runtime.config import RuntimeConfig
from runtime.server.app import create_app
from runtime.server.runtime_chat import (
    chat_needs_legacy,
    message_content_text,
    messages_to_prompt,
)


def test_chat_needs_legacy_tools():
    assert not chat_needs_legacy([], tools=[{"type": "function"}])


def test_chat_needs_legacy_ollama_images():
    assert chat_needs_legacy([{"role": "user", "content": "hi", "images": ["abc"]}])


def test_chat_needs_legacy_think():
    assert chat_needs_legacy([], think=True)


def test_chat_needs_legacy_tool_calls():
    assert not chat_needs_legacy(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "x"}}]}]
    )


def test_chat_needs_legacy_multipart_text_only():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert not chat_needs_legacy(msgs)


def test_message_content_text_multipart():
    assert message_content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"


def test_messages_to_prompt_multipart_text():
    prompt = messages_to_prompt(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    )
    assert prompt == "User: hello"


def test_chat_needs_legacy_openai_vision_parts():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:..."}}],
        }
    ]
    assert chat_needs_legacy(msgs)


def test_api_chat_accepts_tools_empty_engine(cfg_root):
    """Tools route is accepted; engine may fail without llama-server (not 501)."""
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=128,
        block_size=16,
        device_count=1,
    )
    client = TestClient(create_app(cfg))
    r = client.post(
        "/api/chat",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        },
    )
    assert r.status_code != 501
    assert "legacy runner" not in (r.text or "")


def test_v1_chat_accepts_tools_empty_engine(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=128,
        block_size=16,
        device_count=1,
    )
    client = TestClient(create_app(cfg))
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        },
    )
    assert r.status_code != 501
