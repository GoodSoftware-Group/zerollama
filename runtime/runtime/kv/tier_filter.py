"""Per-request secondary-tier load filter (vLLM #48123 pattern).

WHY: multi-tenant / fleet clients may allow only host DRAM, only local storage,
or deny all cold-tier restores while still using same-slot ``cache_prompt``
(primary). Empty ``kv_load_tiers`` denies every secondary; omit → allow all.

Wire: ``options.zerollama.kv_load_tiers`` or top-level ``kv_load_tiers`` —
list of ``{"medium": "CPU"|"STORAGE", "locality": "LOCAL"|"REMOTE"}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, NamedTuple


class Medium(str, Enum):
    CPU = "CPU"
    STORAGE = "STORAGE"


class Locality(str, Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class TierMatcher(NamedTuple):
    medium: Medium | None = None
    locality: Locality | None = None

    def matches(self, medium: Medium | None, locality: Locality | None) -> bool:
        medium_ok = self.medium is None or medium is None or self.medium == medium
        locality_ok = (
            self.locality is None or locality is None or self.locality == locality
        )
        return medium_ok and locality_ok


@dataclass(frozen=True)
class TierFilter:
    """Per-request filter for secondary KV load tiers (not same-slot primary)."""

    matchers: tuple[TierMatcher, ...] = ()

    ALL: ClassVar[TierFilter]

    def allows(self, medium: Medium | None, locality: Locality | None) -> bool:
        if self is TierFilter.ALL:
            return True
        if not self.matchers:
            return False
        return any(m.matches(medium, locality) for m in self.matchers)

    def allows_lmcache(self, *, remote: bool = False) -> bool:
        loc = Locality.REMOTE if remote else Locality.LOCAL
        return self.allows(Medium.STORAGE, loc)

    def allows_host_disk(self) -> bool:
        return self.allows(Medium.CPU, Locality.LOCAL)


TierFilter.ALL = TierFilter(matchers=(TierMatcher(),))


def _parse_medium(raw: Any) -> Medium | None:
    if raw is None:
        return None
    if isinstance(raw, Medium):
        return raw
    s = str(raw).strip().upper()
    if not s:
        return None
    aliases = {
        "CPU": Medium.CPU,
        "HOST": Medium.CPU,
        "DRAM": Medium.CPU,
        "RAM": Medium.CPU,
        "STORAGE": Medium.STORAGE,
        "DISK": Medium.STORAGE,
        "LMCACHE": Medium.STORAGE,
        "BLOB": Medium.STORAGE,
    }
    return aliases.get(s)


def _parse_locality(raw: Any) -> Locality | None:
    if raw is None:
        return None
    if isinstance(raw, Locality):
        return raw
    s = str(raw).strip().upper()
    if not s:
        return None
    aliases = {
        "LOCAL": Locality.LOCAL,
        "REMOTE": Locality.REMOTE,
        "PEER": Locality.REMOTE,
        "FLEET": Locality.REMOTE,
    }
    return aliases.get(s)


def parse_tier_filter(raw: Any) -> TierFilter:
    """Parse ``kv_load_tiers`` JSON list → TierFilter.

    ``None`` / missing → ALL. ``[]`` → deny all secondaries.
    Invalid entries are skipped; if every entry is invalid, treat as ALL.
    """
    if raw is None:
        return TierFilter.ALL
    if not isinstance(raw, (list, tuple)):
        return TierFilter.ALL
    if len(raw) == 0:
        return TierFilter(matchers=())
    matchers: list[TierMatcher] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        medium = _parse_medium(item.get("medium") or item.get("tier"))
        locality = _parse_locality(item.get("locality"))
        # Require at least one recognized field so junk objects don't become wildcards.
        if medium is None and locality is None and not item:
            continue
        if (
            medium is None
            and locality is None
            and "medium" not in item
            and "tier" not in item
            and "locality" not in item
        ):
            continue
        matchers.append(TierMatcher(medium=medium, locality=locality))
    if not matchers:
        return TierFilter.ALL
    return TierFilter(matchers=tuple(matchers))


def lmcache_is_remote() -> bool:
    """True when LMCache URI is redis (cross-node metadata) vs local file."""
    try:
        from runtime.env import lmcache_uri

        uri = (lmcache_uri() or "").strip().lower()
    except Exception:
        return False
    return uri.startswith("redis://") or uri.startswith("rediss://")
