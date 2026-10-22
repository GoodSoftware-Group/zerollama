// Package trainprio — T6 training submit priority / SLO labels (no CGO).
//
// WHY a tiny package: server/ needs CGO+llama on this host; operators and CI
// still need a host-unit for "unknown priority → error" without building ggml.
package trainprio

import (
	"fmt"
	"strings"
)

// Priority controls idle-wait and defer-queue behavior on train submit.
type Priority int

const (
	Normal Priority = iota
	Low
	High
)

func (p Priority) String() string {
	switch p {
	case Low:
		return "low"
	case High:
		return "high"
	default:
		return "normal"
	}
}

func (p Priority) BypassesIdleWait() bool { return p == High }

func (p Priority) PrefersDeferQueue() bool { return p == Low }

// Parse maps a single label to a tier.
//
// WHY reject unknown (not coerce to normal): silent coerce hid typos ("higgh")
// the same way unknown train kinds used to hide "grpoo" → SFT.
// Empty → normal keeps legacy clients that omit the field.
func Parse(raw string) (Priority, error) {
	s := strings.ToLower(strings.TrimSpace(raw))
	switch s {
	case "", "normal", "default", "standard", "medium":
		return Normal, nil
	case "low", "batch", "background", "overnight", "night":
		return Low, nil
	case "high", "interactive", "urgent", "realtime", "operator":
		return High, nil
	default:
		return Normal, fmt.Errorf(
			"unknown priority %q (want normal|low|high; aliases batch/overnight, interactive/urgent/realtime)",
			raw,
		)
	}
}

// Resolve merges priority + optional slo / slo_class.
//
// WHY accept slo_class: operators think in latency SLOs; we keep three tiers
// (not a fourth scheduler) but allow the synonym on the wire.
// WHY conflict → error: silent preference for one field would surprise automation.
func Resolve(priority, slo string) (Priority, error) {
	priority = strings.TrimSpace(priority)
	slo = strings.TrimSpace(slo)
	if priority == "" && slo == "" {
		return Normal, nil
	}
	if priority != "" && slo != "" {
		p1, err1 := Parse(priority)
		p2, err2 := Parse(slo)
		if err1 != nil {
			return Normal, err1
		}
		if err2 != nil {
			return Normal, fmt.Errorf("slo_class: %w", err2)
		}
		if p1 != p2 {
			return Normal, fmt.Errorf(
				"priority %q and slo_class %q disagree (resolved %s vs %s)",
				priority, slo, p1, p2,
			)
		}
		return p1, nil
	}
	if priority != "" {
		return Parse(priority)
	}
	p, err := Parse(slo)
	if err != nil {
		return Normal, fmt.Errorf("slo_class: %w", err)
	}
	return p, nil
}
