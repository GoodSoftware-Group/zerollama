import pytest

from runtime.invariants import (
    UNLIMITED_N_PREDICT,
    admit_token_budget,
    apply_tool_choice,
    completion_cap,
    loaded_ctx_must_grow,
    runtime_port_conflict,
    shared_completion_cap,
    think_forces_legacy,
)


def test_omitted_cap_is_unlimited():
    assert completion_cap(None, None) == UNLIMITED_N_PREDICT
    assert completion_cap(64, None) == 64
    assert completion_cap(None, 32) == 32
    assert admit_token_budget(UNLIMITED_N_PREDICT) == 0
    assert admit_token_budget(None) == 0
    assert admit_token_budget(16) == 16


def test_think_false_stays_on_runtime():
    assert not think_forces_legacy(False)
    assert not think_forces_legacy(None)
    assert think_forces_legacy(True)
    assert think_forces_legacy("low")


def test_unknown_loaded_context_is_a_warm_start():
    assert not loaded_ctx_must_grow(True, True, 8192, None)
    assert loaded_ctx_must_grow(True, True, 8192, 4096)
    assert not loaded_ctx_must_grow(True, True, 4096, 8192)
    assert not loaded_ctx_must_grow(False, True, 8192, 4096)


def test_mixed_batch_keeps_an_unlimited_row():
    assert shared_completion_cap([-1, 50]) == -1
    assert shared_completion_cap([50, 8]) == 50
    assert shared_completion_cap([]) == -1


def test_runtime_ports_stay_off_reserved_listeners():
    assert runtime_port_conflict(8081) is None
    assert runtime_port_conflict(18081) is None
    assert runtime_port_conflict(11434) is not None
    assert runtime_port_conflict(11433) is not None
    assert runtime_port_conflict(18089) is not None


def test_tool_choice_filters():
    tools = [
        {"type": "function", "function": {"name": "a"}},
        {"type": "function", "function": {"name": "b"}},
    ]
    assert apply_tool_choice(tools, "none") == []
    assert apply_tool_choice(tools, "auto") == tools
    assert apply_tool_choice(tools, {"type": "function", "function": {"name": "b"}}) == [
        tools[1]
    ]
    with pytest.raises(ValueError):
        apply_tool_choice(tools, {"type": "function", "function": {"name": "missing"}})
