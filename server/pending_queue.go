package server

import "sync"

// pendingQueue holds scheduler load requests in FIFO order with peek/drain
// so the scheduler can run same-model pending work before evicting a runner.
type pendingQueue struct {
	mu    sync.Mutex
	items []*LlmRequest
	cap   int
	wake  chan struct{}
}

func newPendingQueue(capacity int) *pendingQueue {
	return &pendingQueue{
		cap:  capacity,
		wake: make(chan struct{}, 1),
	}
}

func (q *pendingQueue) notify() {
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

// Push appends a request. Returns false when at capacity.
func (q *pendingQueue) Push(req *LlmRequest) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) >= q.cap {
		return false
	}
	q.items = append(q.items, req)
	q.notify()
	return true
}

// Pop removes and returns the oldest request, or nil if empty.
func (q *pendingQueue) Pop() *LlmRequest {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil
	}
	req := q.items[0]
	q.items = q.items[1:]
	return req
}

// PopPreferringKeys dequeues the oldest request whose model key is in prefer,
// otherwise the FIFO head. Reduces evictions when a loaded model still has
// queued work behind a different model at the head.
func (q *pendingQueue) PopPreferringKeys(prefer map[string]struct{}) *LlmRequest {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil
	}
	if len(prefer) > 0 {
		for i, req := range q.items {
			if _, ok := prefer[schedulerModelKey(req.model)]; ok {
				q.items = append(q.items[:i], q.items[i+1:]...)
				return req
			}
		}
	}
	req := q.items[0]
	q.items = q.items[1:]
	return req
}

// OldestFifoSeq returns the smallest positive fifoSeq in the queue, or 0 when empty.
func (q *pendingQueue) OldestFifoSeq() uint64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	var min uint64
	for _, req := range q.items {
		if req == nil || req.fifoSeq == 0 {
			continue
		}
		if min == 0 || req.fifoSeq < min {
			min = req.fifoSeq
		}
	}
	return min
}

// FifoPosition returns the 1-based index of ticket in the pending queue and total depth.
// Returns position=0 when ticket is not waiting (loading or unknown).
func (q *pendingQueue) FifoPosition(ticket uint64) (position int, depth int) {
	if ticket == 0 {
		return 0, q.Len()
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	depth = len(q.items)
	for i, req := range q.items {
		if req != nil && req.fifoSeq == ticket {
			return i + 1, depth
		}
	}
	return 0, depth
}

// Len returns the number of queued requests.
func (q *pendingQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// Remove deletes req from the queue if present. Returns true when removed.
func (q *pendingQueue) Remove(req *LlmRequest) bool {
	if req == nil {
		return false
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	for i, item := range q.items {
		if item == req {
			q.items = append(q.items[:i], q.items[i+1:]...)
			return true
		}
	}
	return false
}

// WakeCh signals when a request may be available to pop.
func (q *pendingQueue) WakeCh() <-chan struct{} {
	return q.wake
}

// DrainMatching removes all queued requests whose model key equals key.
func (q *pendingQueue) DrainMatching(key string) []*LlmRequest {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil
	}
	var matched, rest []*LlmRequest
	for _, req := range q.items {
		if schedulerModelKey(req.model) == key {
			matched = append(matched, req)
		} else {
			rest = append(rest, req)
		}
	}
	q.items = rest
	return matched
}

// RequeueFront prepends requests in order (first req will be next after any future pop).
func (q *pendingQueue) RequeueFront(reqs []*LlmRequest) {
	if len(reqs) == 0 {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = append(reqs, q.items...)
	q.notify()
}
