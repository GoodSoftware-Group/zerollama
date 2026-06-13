"""Map scheduler requests to llama-server parallel slots (Phase 15 v1 + L3 pin).

Phase 15 v1: dynamic ``acquire()`` for one-shot requests.
L3: ``try_acquire(slot)`` for session-pinned slots derived from cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlotAllocator:
    """Track which llama-server ``id_slot`` values (0 .. num_slots-1) are in use."""

    num_slots: int
    _in_use: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.num_slots < 1:
            raise ValueError("num_slots must be at least 1")

    def acquire(self) -> int | None:
        for slot in range(self.num_slots):
            if slot not in self._in_use:
                self._in_use.add(slot)
                return slot
        return None

    def try_acquire(self, slot: int) -> bool:
        """Reserve a specific slot (L3 pinned sessions). False if busy or out of range."""
        if slot < 0 or slot >= self.num_slots:
            return False
        if slot in self._in_use:
            return False
        self._in_use.add(slot)
        return True

    def release(self, slot: int) -> None:
        if 0 <= slot < self.num_slots:
            self._in_use.discard(slot)

    def in_use_count(self) -> int:
        return len(self._in_use)

    def snapshot(self) -> dict[str, int | list[int]]:
        return {
            "num_slots": self.num_slots,
            "in_use": self.in_use_count(),
            "slots_busy": sorted(self._in_use),
        }
