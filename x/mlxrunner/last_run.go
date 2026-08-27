package mlxrunner

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const lastRunKeep = 8

// LastRun is one completed MLX decode, persisted so doctor can tune without
// scraping serve logs.
type LastRun struct {
	Model        string    `json:"model"`
	At           time.Time `json:"at"`
	Iterations   int       `json:"iterations"`
	Drafted      int       `json:"drafted"`
	Accepted     int       `json:"accepted"`
	MaxDraft     int       `json:"max_draft"`
	Scheduled    int       `json:"scheduled"`
	Acceptance   float64   `json:"acceptance"`
	Enabled      bool      `json:"enabled"`
	PLD          bool      `json:"pld"`
	PromptTokens int       `json:"prompt_tokens,omitempty"`
	CtxBucket    int       `json:"ctx_bucket,omitempty"`
	Hint         string    `json:"hint,omitempty"`
}

type lastRunFile struct {
	Runs []LastRun `json:"runs"`
}

func lastRunPath(modelName string) string {
	p := roundCostPath(modelName)
	if p == "" {
		return ""
	}
	return strings.TrimSuffix(p, ".json") + ".last.json"
}

func readLastRuns(path string) []LastRun {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var wrap lastRunFile
	if json.Unmarshal(data, &wrap) == nil && len(wrap.Runs) > 0 {
		return wrap.Runs
	}
	var one LastRun
	if json.Unmarshal(data, &one) == nil && (one.Model != "" || one.Drafted > 0 || one.At != (time.Time{})) {
		return []LastRun{one}
	}
	return nil
}

func (s *speculationSession) saveLastRun(acceptance float64) {
	if s == nil || s.spec == nil || s.spec.r == nil {
		return
	}
	path := lastRunPath(s.spec.r.modelName)
	if path == "" {
		return
	}
	scheduled := 0
	if s.spec.depth != nil {
		scheduled = s.spec.depth.scheduled
	}
	f := LastRun{
		Model:        s.spec.r.modelName,
		At:           time.Now().UTC(),
		Iterations:   s.stats.iterations,
		Drafted:      s.stats.drafted,
		Accepted:     s.stats.accepted,
		MaxDraft:     s.stats.maxDraft,
		Scheduled:    scheduled,
		Acceptance:   acceptance,
		Enabled:      s.enabled,
		PLD:          s.pld,
		PromptTokens: s.promptTokens,
		CtxBucket:    ctxBucket(s.promptTokens),
		Hint:         s.tuneHint(acceptance),
	}
	runs := append(readLastRuns(path), f)
	if len(runs) > lastRunKeep {
		runs = runs[len(runs)-lastRunKeep:]
	}
	data, err := json.Marshal(lastRunFile{Runs: runs})
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return
	}
	_ = os.Rename(tmp, path)
}

func summarizeRuns(runs []LastRun) (detail string, warn bool) {
	if len(runs) == 0 {
		return "", false
	}
	best := runs[len(runs)-1]
	parked := 0
	for _, r := range runs {
		if !r.Enabled {
			parked++
		}
	}
	kind := "mtp"
	if best.PLD {
		kind = "pld"
	}
	state := "on"
	if !best.Enabled {
		state = "parked"
	}
	detail = fmt.Sprintf("%s last %d: %s accept=%.2f drafted=%d/%d width=%d scheduled=%d %s parked=%d/%d ctx=%d",
		best.Model, len(runs), kind, best.Acceptance, best.Accepted, best.Drafted, best.MaxDraft, best.Scheduled, state, parked, len(runs), best.CtxBucket)
	if best.Hint != "" {
		detail += " — " + best.Hint
	}
	// One novel chat parking PLD is expected. Warn when it is the usual path.
	if parked*2 >= len(runs) && parked >= 3 {
		warn = true
	}
	if best.Hint != "" && strings.Contains(best.Hint, "draft width 0") {
		warn = true
	}
	return detail, warn
}

// LastRunTuneReport summarizes recent persisted decodes for doctor.
func LastRunTuneReport() (detail string, warn bool) {
	dir := roundCostDir()
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "no last MLX decode yet", false
	}
	var bestRuns []LastRun
	var bestAt time.Time
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".last.json") {
			continue
		}
		runs := readLastRuns(filepath.Join(dir, e.Name()))
		if len(runs) == 0 {
			continue
		}
		at := runs[len(runs)-1].At
		if bestRuns == nil || at.After(bestAt) {
			bestRuns, bestAt = runs, at
		}
	}
	if len(bestRuns) == 0 {
		return "no last MLX decode yet", false
	}
	return summarizeRuns(bestRuns)
}
