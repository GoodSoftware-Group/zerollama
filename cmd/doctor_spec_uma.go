package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// Upstream trap 98: speculative depth K and sequence capacity coincide with
// OOMs on unified memory. Contributor observations (vLLM/GB10, not isolated):
// K=15×seqs=32 crashed; K=15×seqs=4 stable; a different soak at K=7×32 was
// stable. Treat K×slots as a pressure product, not a proven threshold.
const doctorSpecUMAWarnProduct = 224

func doctorCheckSpeculativeUMA() doctorCheck {
	const name = "serving trap-98 (spec × slots on UMA)"
	slots := doctorEnvInt("ZEROLLAMA_LLAMA_PARALLEL_SLOTS", 0)
	draft := doctorEnvInt("ZEROLLAMA_DRAFT_MAX", 0)
	specOn := envconfigTruthy(os.Getenv("ZEROLLAMA_ANE_DRAFT")) ||
		strings.TrimSpace(os.Getenv("LLAMA_DRAFT_MODEL")) != "" ||
		strings.TrimSpace(os.Getenv("ZEROLLAMA_SPEC_TYPE")) != ""

	profileSlots, profileDraft, profileID := doctorAppleProfileSpecDefaults()
	if slots <= 0 {
		slots = profileSlots
	}
	if draft <= 0 {
		draft = profileDraft
	}
	if slots <= 0 {
		slots = 1
	}
	if draft <= 0 {
		draft = 1
	}
	product := slots * draft

	parts := []string{
		fmt.Sprintf("slots=%d draft_k≈%d product=%d", slots, draft, product),
	}
	if profileID != "" {
		parts = append(parts, "profile="+profileID)
	}
	if specOn {
		parts = append(parts, "speculative env on")
	} else {
		parts = append(parts, "speculative env off (product still matters if llama-server --spec-type is set)")
	}

	uma := runtime.GOOS == "darwin"
	status := "ok"
	fix := ""
	if uma && specOn && product >= doctorSpecUMAWarnProduct {
		status = "warn"
		parts = append(parts, "trap 98 SIGNAL: K×slots in the contributor-crash region on UMA")
		fix = "lower n_parallel or draft_max together; a seqs value validated at one K is not safe at another"
	} else if uma && product >= 128 {
		parts = append(parts, "trap 98 note: Apple 128 GiB profile is draft_max=16 × n_parallel=8 (product 128); raise neither without a soak")
	} else {
		parts = append(parts, "trap 98: report K and n_parallel with any UMA OOM; do not copy vLLM max-num-seqs defaults")
	}

	return doctorCheck{
		Name:    name,
		Status:  status,
		Detail:  strings.Join(parts, "; "),
		FixHint: fix,
	}
}

func doctorEnvInt(key string, fallback int) int {
	s := strings.TrimSpace(os.Getenv(key))
	if s == "" {
		return fallback
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return fallback
	}
	return n
}

func envconfigTruthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func doctorAppleProfileSpecDefaults() (slots, draft int, id string) {
	root := doctorRepoRoot()
	path := filepath.Join(root, "runtime", "configs", "gpu", "apple_silicon_128g.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, ""
	}
	var doc struct {
		ID    string `json:"id"`
		Flags struct {
			NParallel int `json:"n_parallel"`
			DraftMax  int `json:"draft_max"`
		} `json:"llama_server_flags"`
	}
	if json.Unmarshal(raw, &doc) != nil {
		return 0, 0, ""
	}
	return doc.Flags.NParallel, doc.Flags.DraftMax, doc.ID
}
