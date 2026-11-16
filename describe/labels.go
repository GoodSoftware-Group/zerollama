package describe

import (
	"strings"
)

// ResolveLabels turns a caller label list into (json labels, embed phrases).
// An empty caller list uses BuiltinLabels. A non-empty list uses each word as
// both the JSON label and the embedded text.
func ResolveLabels(caller []string) (labels []string, phrases []string) {
	if len(caller) == 0 {
		labels = make([]string, len(BuiltinLabels))
		phrases = make([]string, len(BuiltinLabels))
		for i, lp := range BuiltinLabels {
			labels[i] = lp.Label
			phrases[i] = lp.Phrase
		}
		return labels, phrases
	}
	seen := map[string]struct{}{}
	for _, raw := range caller {
		label := strings.TrimSpace(raw)
		if label == "" {
			continue
		}
		key := strings.ToLower(label)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		labels = append(labels, label)
		phrases = append(phrases, label)
	}
	return labels, phrases
}

// ParseCommaLabels splits a --labels flag.
func ParseCommaLabels(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// LabelNames returns just the short labels from BuiltinLabels.
func LabelNames() []string {
	out := make([]string, len(BuiltinLabels))
	for i, lp := range BuiltinLabels {
		out[i] = lp.Label
	}
	return out
}
