#!/usr/bin/env bash
# Go scheduler watchdog defaults (LocalAI borrowings). Source before zerollama serve.
# Override any variable in the shell before sourcing.
#
#   source ./scripts/sched_watchdog_env.sh
#   ./zerollama serve
#
# Disable memory reclaimer: ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD=0
# Stuck-runner recovery (opt-in): ZEROLLAMA_RUNNER_BUSY_TIMEOUT=30m

export ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD="${ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD:-0.95}"
export ZEROLLAMA_SCHED_WATCHDOG_INTERVAL="${ZEROLLAMA_SCHED_WATCHDOG_INTERVAL:-30s}"
