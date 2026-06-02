"""Phase 15: parallel slot count matches llama-server -np."""

from runtime.llama_args import resolve_parallel_slots


def test_resolve_parallel_slots_yaml_default():
    assert resolve_parallel_slots([], default=4) == 4


def test_resolve_parallel_slots_argv_overrides_default():
    assert resolve_parallel_slots(["-np", "2"], default=8) == 2
    assert resolve_parallel_slots(["--parallel", "3"], default=1) == 3


def test_resolve_parallel_slots_last_wins():
    assert resolve_parallel_slots(["-np", "2", "-np", "5"], default=1) == 5
