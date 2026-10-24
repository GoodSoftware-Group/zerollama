package autoroute

import (
	"container/list"
	"fmt"
	"sync"
	"time"
)

// Memory is a small TTL + LRU map from a routing key to the model last chosen for it.
// Two instances back AR6: the sticky table (per conversation, minutes) and the decision
// cache (per identical request, seconds). Safe for concurrent use.
//
// WHY remember models, not decisions: a remembered model is only a hint — callers
// re-validate it against the current request (Validate) before reusing it, so stale
// availability or new capability needs never leak through the cache.
type Memory struct {
	mu    sync.Mutex
	ttl   time.Duration
	max   int
	ll    *list.List // front = most recent
	items map[string]*list.Element
	now   func() time.Time
}

type memEntry struct {
	key     string
	model   string
	expires time.Time
}

// NewMemory returns a memory holding at most max keys for ttl after their last use.
func NewMemory(ttl time.Duration, max int) *Memory {
	return &Memory{ttl: ttl, max: max, ll: list.New(), items: map[string]*list.Element{}, now: time.Now}
}

// Get returns the remembered model and refreshes its TTL (sliding expiry).
func (m *Memory) Get(key string) (string, bool) {
	if m == nil || key == "" {
		return "", false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	el, ok := m.items[key]
	if !ok {
		return "", false
	}
	e := el.Value.(*memEntry)
	now := m.now()
	if now.After(e.expires) {
		m.ll.Remove(el)
		delete(m.items, key)
		return "", false
	}
	e.expires = now.Add(m.ttl)
	m.ll.MoveToFront(el)
	return e.model, true
}

// Put remembers model for key, evicting the least recently used key past max.
func (m *Memory) Put(key, model string) {
	if m == nil || key == "" || model == "" || m.ttl <= 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	expires := m.now().Add(m.ttl)
	if el, ok := m.items[key]; ok {
		e := el.Value.(*memEntry)
		e.model, e.expires = model, expires
		m.ll.MoveToFront(el)
		return
	}
	m.items[key] = m.ll.PushFront(&memEntry{key: key, model: model, expires: expires})
	for m.max > 0 && m.ll.Len() > m.max {
		old := m.ll.Back()
		m.ll.Remove(old)
		delete(m.items, old.Value.(*memEntry).key)
	}
}

// Forget drops key (e.g. after the remembered model failed validation).
func (m *Memory) Forget(key string) {
	if m == nil || key == "" {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if el, ok := m.items[key]; ok {
		m.ll.Remove(el)
		delete(m.items, key)
	}
}

// Len reports the number of live (possibly expired, not yet swept) keys.
func (m *Memory) Len() int {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.ll.Len()
}

// Validate checks one remembered candidate against the current request with every hard
// filter applied strictly — surface, capabilities, Check, context fit and availability.
//
// WHY strict context: Decide's context filter is soft (it never rejects every candidate),
// so validating a lone candidate through Decide would keep a model the conversation has
// outgrown. A remembered model must actually fit, or the request re-routes.
func Validate(c Candidate, sig Signals, avail AvailabilityFunc) (string, bool) {
	if reason, ok := sig.Admits(c); !ok {
		return reason, false
	}
	if need := sig.contextNeed(); need > 0 && c.ContextLength > 0 && c.ContextLength < need {
		return fmt.Sprintf("context %d < needed %d", c.ContextLength, need), false
	}
	if reject := voiceFilter(sig); reject != nil {
		if reason := reject(c); reason != "" {
			return reason, false
		}
	}
	if avail == nil {
		return "", true
	}
	a := avail(c)
	switch {
	case a.Excluded != "":
		return a.Excluded, false
	case !a.Warm && !a.CanLoad:
		reason := "cannot load now"
		if a.Detail != "" {
			reason += ": " + a.Detail
		}
		return reason, false
	}
	return "", true
}
