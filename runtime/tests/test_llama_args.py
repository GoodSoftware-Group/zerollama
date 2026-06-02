from runtime.llama_args import (
    inprocess_speculative_requested,
    parse_llama_server_args,
    with_llama_num_ctx,
)


def test_with_llama_num_ctx_replaces_flag():
    argv = ["-np", "2", "-c", "2048"]
    out = with_llama_num_ctx(argv, 8192)
    assert parse_llama_server_args(out).num_ctx == 8192


def test_with_llama_num_ctx_appends_when_missing():
    argv = ["-ngl", "99"]
    out = with_llama_num_ctx(argv, 4096)
    assert parse_llama_server_args(out).num_ctx == 4096


def test_with_llama_num_ctx_eq_form():
    out = with_llama_num_ctx(["--ctx-size=2048"], 1024)
    assert parse_llama_server_args(out).num_ctx == 1024


def test_parse_gpu_split_flags():
    hints = parse_llama_server_args(
        ["-mg", "1", "-sm", "tensor", "-ts", "0.5,0.5", "-ngl", "80"]
    )
    assert hints.main_gpu == 1
    assert hints.split_mode == "tensor"
    assert hints.tensor_split == (0.5, 0.5)
    assert hints.n_gpu_layers == 80


def test_inprocess_speculative_requested():
    assert not inprocess_speculative_requested(parse_llama_server_args([]))
    assert inprocess_speculative_requested(
        parse_llama_server_args(["--spec-type", "draft"])
    )
