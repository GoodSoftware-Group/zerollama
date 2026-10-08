package storaged

import (
	"sync/atomic"
)

// metrics holds process-lifetime counters for /v1/metrics (Prometheus text).
type metrics struct {
	gets          atomic.Int64
	puts          atomic.Int64
	putBytes      atomic.Int64
	getBytes      atomic.Int64
	errors        atomic.Int64
	inFlight      atomic.Int64
	replicates    atomic.Int64
	replicateErrs atomic.Int64
}

func (m *metrics) incGet(n int64) {
	m.gets.Add(1)
	if n > 0 {
		m.getBytes.Add(n)
	}
}

func (m *metrics) incPut(n int64) {
	m.puts.Add(1)
	if n > 0 {
		m.putBytes.Add(n)
	}
}
