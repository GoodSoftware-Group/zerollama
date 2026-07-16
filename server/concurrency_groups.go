// Concurrency groups enforce mutual exclusion between loaded models on tight GPUs.
// Why: OLLAMA_MAX_LOADED_MODELS limits count but cannot express "never keep chat
// and imagegen resident together" — LocalAI uses the same pattern for vram-heavy pairs.
package server

import (
	"strings"
	"time"
)

// concurrencyGroupsForModel returns declared mutual-exclusion groups (LocalAI pattern).
// Modelfile: PARAMETER concurrency_groups ["vram-heavy","vision"]
func concurrencyGroupsForModel(m *Model) []string {
	if m == nil {
		return nil
	}
	if g := normalizeConcurrencyGroups(m.Options["concurrency_groups"]); len(g) > 0 {
		return g
	}
	return normalizeConcurrencyGroups(m.Config.ConcurrencyGroups)
}

func normalizeConcurrencyGroups(v any) []string {
	switch t := v.(type) {
	case []string:
		return dedupeConcurrencyGroups(t)
	case []any:
		out := make([]string, 0, len(t))
		for _, item := range t {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return dedupeConcurrencyGroups(out)
	case string:
		s := strings.TrimSpace(t)
		if s == "" {
			return nil
		}
		parts := strings.FieldsFunc(s, func(r rune) bool {
			return r == ',' || r == ';'
		})
		return dedupeConcurrencyGroups(parts)
	default:
		return nil
	}
}

func dedupeConcurrencyGroups(groups []string) []string {
	seen := make(map[string]struct{}, len(groups))
	out := make([]string, 0, len(groups))
	for _, g := range groups {
		g = strings.TrimSpace(g)
		if g == "" {
			continue
		}
		key := strings.ToLower(g)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, g)
	}
	return out
}

func concurrencyGroupsOverlap(a, b []string) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	for _, ga := range a {
		la := strings.ToLower(ga)
		for _, gb := range b {
			if la == strings.ToLower(gb) {
				return true
			}
		}
	}
	return false
}

// findConcurrencyGroupConflict returns a resident runner that shares a concurrency
// group with pending. Callers evict it before loading pending (group exclusivity).
func (s *Scheduler) findConcurrencyGroupConflict(pending *Model) *runnerRef {
	if s == nil || pending == nil {
		return nil
	}
	pendingGroups := concurrencyGroupsForModel(pending)
	if len(pendingGroups) == 0 {
		return nil
	}
	pendingKey := schedulerModelKey(pending)

	s.loadedMu.Lock()
	candidates := make([]*runnerRef, 0)
	for key, runner := range s.loaded {
		if key == pendingKey {
			continue
		}
		if runner == nil || runner.model == nil {
			continue
		}
		if !concurrencyGroupsOverlap(pendingGroups, concurrencyGroupsForModel(runner.model)) {
			continue
		}
		candidates = append(candidates, runner)
	}
	s.loadedMu.Unlock()

	var victim *runnerRef
	var victimRefs uint
	var victimLastUsed time.Time
	for _, runner := range candidates {
		runner.refMu.Lock()
		rc := runner.refCount
		lastUsed := runner.lastUsedAt
		runner.refMu.Unlock()

		if victim == nil {
			victim = runner
			victimRefs = rc
			victimLastUsed = lastUsed
			continue
		}
		if rc == 0 && victimRefs > 0 {
			victim = runner
			victimRefs = rc
			victimLastUsed = lastUsed
			continue
		}
		if !lastUsed.IsZero() && (victimLastUsed.IsZero() || lastUsed.Before(victimLastUsed)) {
			victim = runner
			victimRefs = rc
			victimLastUsed = lastUsed
		}
	}
	return victim
}

// evictConcurrencyGroupConflicts unloads every loaded runner that shares a group with m.
func (s *Scheduler) evictConcurrencyGroupConflicts(m *Model) {
	if s == nil || m == nil {
		return
	}
	pendingGroups := concurrencyGroupsForModel(m)
	if len(pendingGroups) == 0 {
		return
	}
	pendingKey := schedulerModelKey(m)

	s.loadedMu.Lock()
	conflicts := make([]*runnerRef, 0)
	for key, runner := range s.loaded {
		if key == pendingKey || runner == nil || runner.model == nil {
			continue
		}
		if concurrencyGroupsOverlap(pendingGroups, concurrencyGroupsForModel(runner.model)) {
			conflicts = append(conflicts, runner)
		}
	}
	s.loadedMu.Unlock()

	for _, runner := range conflicts {
		s.scheduleExpiredRunner(runner)
	}
}
