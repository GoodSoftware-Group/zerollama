package describe

import (
	"slices"
	"strings"
)

// Want is the categories requested for this call.
type Want struct {
	Tags    bool
	Caption bool
	Objects bool
}

// ParseFields turns a --fields / request fields list into Want.
// Empty means tags on (file facts are always recorded).
func ParseFields(fields []string) Want {
	if len(fields) == 0 {
		return Want{Tags: true}
	}
	var w Want
	for _, f := range fields {
		switch strings.ToLower(strings.TrimSpace(f)) {
		case "tags", "tag":
			w.Tags = true
		case "caption", "captions":
			w.Caption = true
		case "objects", "object", "detect":
			w.Objects = true
		case "file", "files":
			// always on; ignore
		}
	}
	return w
}

// Settings are the model knobs that invalidate stored fields when they change.
type Settings struct {
	TagModel     string
	CaptionModel string
	Labels       []string
	MinScore     float64
}

func labelsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	return slices.Equal(a, b)
}

func minScoreOrDefault(v float64) float64 {
	if v <= 0 {
		return DefaultMinScore
	}
	return v
}

// Decision is what to do for one image id.
type Decision string

const (
	DecisionNew      Decision = "new"
	DecisionKept     Decision = "kept"
	DecisionChanged  Decision = "changed"
	DecisionStale    Decision = "stale"
	DecisionNeedsImg Decision = "needs_image"
)

// Decide returns what to do for an uploaded or hash-only image against a catalog entry.
// existing may be nil. sha is the content hash (from data or the caller). hasData
// is true when image bytes are present in this request.
func Decide(existing *Entry, sha string, hasData bool, want Want, settings Settings) Decision {
	if existing == nil {
		if !hasData {
			return DecisionNeedsImg
		}
		return DecisionNew
	}
	if sha == "" {
		if hasData {
			return DecisionChanged
		}
		return DecisionNeedsImg
	}
	if existing.SHA256 != "" && existing.SHA256 != sha {
		if !hasData {
			return DecisionNeedsImg
		}
		return DecisionChanged
	}
	if existing.SHA256 == "" && hasData {
		return DecisionChanged
	}
	if existing.SHA256 == "" && !hasData {
		return DecisionNeedsImg
	}

	// Same hash. Check whether requested fields are already present for these settings.
	if want.Tags {
		sameSettings := existing.TagModel == settings.TagModel &&
			labelsEqual(existing.Labels, settings.Labels) &&
			minScoreOrDefault(existing.MinScore) == minScoreOrDefault(settings.MinScore)
		if !existing.HasTags() || !sameSettings {
			if !hasData {
				return DecisionNeedsImg
			}
			return DecisionStale
		}
	}
	if want.Caption {
		sameCap := existing.CaptionModel == settings.CaptionModel
		if !existing.HasCaption() || !sameCap {
			if !hasData {
				return DecisionNeedsImg
			}
			return DecisionStale
		}
	}
	if want.Objects {
		// wire_only / empty detect is not a finished result; label list is the query set.
		okObjects := existing.Objects != nil &&
			existing.Objects.Status == "ok" &&
			labelsEqual(existing.Labels, settings.Labels)
		if !okObjects {
			if !hasData {
				return DecisionNeedsImg
			}
			return DecisionStale
		}
	}
	return DecisionKept
}

// LocalSkip is true when the CLI can skip hashing and posting: size and mtime
// match the catalog and Decide would return kept for the stored sha.
func LocalSkip(existing *Entry, bytes int64, mtimeEqual bool, want Want, settings Settings) bool {
	if existing == nil || existing.File == nil {
		return false
	}
	if existing.File.Bytes != bytes || !mtimeEqual {
		return false
	}
	// Unsupported formats are checkpointed without a content hash; do not retry forever.
	if existing.Error != "" && strings.Contains(existing.Error, "unsupported image format") {
		return true
	}
	if existing.SHA256 == "" {
		return false
	}
	return Decide(existing, existing.SHA256, true, want, settings) == DecisionKept
}
