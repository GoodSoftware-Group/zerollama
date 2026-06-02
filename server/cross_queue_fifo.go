package server

import "sync/atomic"

// crossQueueSeq is a process-wide monotonic ticket for T6 cross-queue FIFO ordering
// (ggml pending, training defer, Python runtime waiting).
var crossQueueSeq atomic.Uint64

// AllocCrossQueueSeq returns the next global FIFO ticket (never 0).
func AllocCrossQueueSeq() uint64 {
	for {
		n := crossQueueSeq.Add(1)
		if n != 0 {
			return n
		}
	}
}

// minNonZeroUint64 returns the smallest positive value, or 0 when both are 0.
func minNonZeroUint64(a, b uint64) uint64 {
	if a == 0 {
		return b
	}
	if b == 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}
