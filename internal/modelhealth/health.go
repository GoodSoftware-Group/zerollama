// Package modelhealth inspects local model manifests and blob integrity.
package modelhealth

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/lmstudio"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// Status summarizes model readiness on disk.
type Status string

const (
	StatusOK         Status = "ok"
	StatusRepairable Status = "repairable"
	StatusOrphaned   Status = "orphaned"
	StatusBroken     Status = "broken"
)

// Report is one model tag health result.
type Report struct {
	Name    string
	Status  Status
	Detail  string
	FixHint string
}

// MissingBlobPaths returns blob paths referenced by mf that are absent or broken symlinks.
func MissingBlobPaths(mf *manifest.Manifest) ([]string, error) {
	return missingBlobPathsIn(envconfig.Models(), mf)
}

func missingBlobPathsIn(modelsDir string, mf *manifest.Manifest) ([]string, error) {
	if mf == nil {
		return nil, fmt.Errorf("nil manifest")
	}

	var missing []string
	check := func(digest string) error {
		if digest == "" {
			return nil
		}
		path, err := manifest.BlobsPathIn(modelsDir, digest)
		if err != nil {
			return err
		}
		if !blobAccessible(path) {
			missing = append(missing, path)
		}
		return nil
	}

	if err := check(mf.Config.Digest); err != nil {
		return nil, err
	}
	for _, layer := range mf.Layers {
		if err := check(layer.Digest); err != nil {
			return nil, err
		}
	}
	return missing, nil
}

func blobAccessible(path string) bool {
	fi, err := os.Lstat(path)
	if err != nil {
		return false
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		if _, err := os.Stat(path); err != nil {
			return false
		}
	}
	return true
}

// HasMissingBlobs reports whether any manifest layer blob is missing or dangling.
func HasMissingBlobs(mf *manifest.Manifest) bool {
	missing, err := MissingBlobPaths(mf)
	return err != nil || len(missing) > 0
}

// CheckName inspects one fully-qualified model tag.
func CheckName(name string) (Report, error) {
	n := model.ParseName(name)
	if !n.IsValid() {
		return Report{}, fmt.Errorf("invalid model name %q", name)
	}
	return checkParsedName(n)
}

func checkParsedName(n model.Name) (Report, error) {
	display := n.DisplayShortest()
	for _, modelsDir := range envconfig.ModelsSearchDirs() {
		mf, err := manifest.ParseNamedManifestIn(modelsDir, n)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return Report{}, err
		}

		missing, err := missingBlobPathsIn(modelsDir, mf)
		if err != nil {
			return Report{}, err
		}
		if len(missing) == 0 {
			return Report{Name: display, Status: StatusOK}, nil
		}

		detail := fmt.Sprintf("%d missing blob(s), e.g. %s", len(missing), filepath.Base(missing[0]))
		if envconfig.LMStudioImport(true) {
			if _, _, ok := lmstudio.MatchSelection(n); ok {
				return Report{
					Name:    display,
					Status:  StatusRepairable,
					Detail:  detail + " (LM Studio cache match)",
					FixHint: "zerollama list or zerollama pull " + display + " to re-import from LM Studio",
				}, nil
			}
		}
		return Report{
			Name:    display,
			Status:  StatusOrphaned,
			Detail:  detail + " (no LM Studio source)",
			FixHint: "zerollama rm " + display + " then re-pull or re-download in LM Studio",
		}, nil
	}

	return Report{
		Name:    display,
		Status:  StatusBroken,
		Detail:  "manifest not found",
		FixHint: "zerollama pull " + display,
	}, nil
}

// CheckAll walks every local manifest tag under OLLAMA_MODELS search paths.
func CheckAll() ([]Report, error) {
	mfs, err := manifest.ManifestsSearch(envconfig.ModelsSearchDirs(), true)
	if err != nil {
		return nil, err
	}

	names := make([]model.Name, 0, len(mfs))
	for n := range mfs {
		names = append(names, n)
	}
	slices.SortFunc(names, func(a, b model.Name) int {
		return strings.Compare(a.DisplayShortest(), b.DisplayShortest())
	})

	out := make([]Report, 0, len(names))
	for _, n := range names {
		r, err := checkParsedName(n)
		if err != nil {
			return out, fmt.Errorf("%s: %w", n, err)
		}
		out = append(out, r)
	}
	return out, nil
}

// IsBenchable returns false when the model should be skipped for generate benchmarks.
func IsBenchable(r Report) bool {
	return r.Status == StatusOK || r.Status == StatusRepairable
}

// FormatSummary returns a one-line human summary for doctor output.
func FormatSummary(r Report) string {
	switch r.Status {
	case StatusOK:
		return "ok"
	default:
		s := string(r.Status)
		if r.Detail != "" {
			s += ": " + r.Detail
		}
		return s
	}
}

// RemoveManifest deletes the manifest file for name without pruning blobs.
func RemoveManifest(name string) error {
	n := model.ParseName(name)
	if !n.IsValid() {
		return fmt.Errorf("invalid model name %q", name)
	}
	for _, modelsDir := range envconfig.ModelsSearchDirs() {
		mf, err := manifest.ParseNamedManifestIn(modelsDir, n)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return err
		}
		return mf.Remove()
	}
	return os.ErrNotExist
}

// FilterHealthy returns reports that are not OK for display/fix.
func FilterUnhealthy(reports []Report) []Report {
	var out []Report
	for _, r := range reports {
		if r.Status != StatusOK {
			out = append(out, r)
		}
	}
	return out
}

// MatchByPrefix filters reports whose name starts with any prefix (case-insensitive).
func MatchByPrefix(reports []Report, prefixes []string) []Report {
	if len(prefixes) == 0 {
		return reports
	}
	var out []Report
	for _, r := range reports {
		lower := strings.ToLower(r.Name)
		for _, p := range prefixes {
			if strings.HasPrefix(lower, strings.ToLower(p)) {
				out = append(out, r)
				break
			}
		}
	}
	return out
}
