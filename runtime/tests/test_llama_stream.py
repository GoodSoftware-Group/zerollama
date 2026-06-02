from runtime.worker.llama_stream import iter_llama_sse_lines


def test_iter_llama_sse_lines():
    raw = [
        b'data: {"content": "hi", "stop": false}\n\n',
        b'data: {"content": "!", "stop": true}\n\n',
    ]
    chunks = list(iter_llama_sse_lines(iter(raw)))
    assert len(chunks) == 2
    assert chunks[0]["content"] == "hi"
    assert chunks[1]["stop"] is True
