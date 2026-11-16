package describe

import (
	"testing"
	"time"
)

func TestDecideNeedsImage(t *testing.T) {
	want := Want{Tags: true}
	settings := Settings{TagModel: "siglip", Labels: []string{"dog"}, MinScore: 0.2}

	if d := Decide(nil, "", false, want, settings); d != DecisionNeedsImg {
		t.Fatalf("new without data: %s", d)
	}
	existing := &Entry{ID: "a", SHA256: "abc", Tags: &[]TagScore{}}
	if d := Decide(existing, "def", false, want, settings); d != DecisionNeedsImg {
		t.Fatalf("changed hash without data: %s", d)
	}
}

func TestDecideKept(t *testing.T) {
	tags := []TagScore{}
	existing := &Entry{
		ID:       "a",
		SHA256:   "abc",
		Tags:     &tags,
		TagModel: "siglip",
		Labels:   []string{"dog"},
		MinScore: 0.2,
	}
	want := Want{Tags: true}
	settings := Settings{TagModel: "siglip", Labels: []string{"dog"}, MinScore: 0.2}
	if d := Decide(existing, "abc", false, want, settings); d != DecisionKept {
		t.Fatalf("kept: %s", d)
	}
	if d := Decide(existing, "abc", true, want, settings); d != DecisionKept {
		t.Fatalf("kept with data: %s", d)
	}
}

func TestDecideStaleSettings(t *testing.T) {
	tags := []TagScore{{Label: "dog", Score: 0.3}}
	existing := &Entry{
		ID:       "a",
		SHA256:   "abc",
		Tags:     &tags,
		TagModel: "siglip",
		Labels:   []string{"dog"},
		MinScore: 0.2,
	}
	want := Want{Tags: true}
	settings := Settings{TagModel: "siglip", Labels: []string{"cat"}, MinScore: 0.2}
	if d := Decide(existing, "abc", true, want, settings); d != DecisionStale {
		t.Fatalf("stale labels: %s", d)
	}
	if d := Decide(existing, "abc", false, want, settings); d != DecisionNeedsImg {
		t.Fatalf("stale without data: %s", d)
	}
}

func TestDecideCaptionFillIn(t *testing.T) {
	tags := []TagScore{}
	existing := &Entry{
		ID:       "a",
		SHA256:   "abc",
		Tags:     &tags,
		TagModel: "siglip",
		Labels:   []string{"dog"},
		MinScore: 0.2,
	}
	want := Want{Tags: true, Caption: true}
	settings := Settings{TagModel: "siglip", CaptionModel: "vl", Labels: []string{"dog"}, MinScore: 0.2}
	if d := Decide(existing, "abc", true, want, settings); d != DecisionStale {
		t.Fatalf("need caption: %s", d)
	}
}

func TestLocalSkip(t *testing.T) {
	tags := []TagScore{}
	mtime := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	existing := &Entry{
		ID:       "a.jpg",
		SHA256:   "abc",
		Tags:     &tags,
		TagModel: "siglip",
		Labels:   []string{"dog"},
		MinScore: 0.2,
		File:     &FileFacts{Bytes: 100, Mtime: mtime},
	}
	want := Want{Tags: true}
	settings := Settings{TagModel: "siglip", Labels: []string{"dog"}, MinScore: 0.2}
	if !LocalSkip(existing, 100, true, want, settings) {
		t.Fatal("expected skip")
	}
	if LocalSkip(existing, 101, true, want, settings) {
		t.Fatal("size change must not skip")
	}
	if LocalSkip(existing, 100, false, want, settings) {
		t.Fatal("mtime change must not skip without rehash path")
	}

	heic := &Entry{
		ID:    "shot.heic",
		Error: "unsupported image format",
		File:  &FileFacts{Bytes: 50, Mtime: mtime},
	}
	if !LocalSkip(heic, 50, true, want, settings) {
		t.Fatal("unsupported HEIC should skip on same size/mtime")
	}
	if LocalSkip(heic, 51, true, want, settings) {
		t.Fatal("replaced HEIC must not skip")
	}
}

func TestDecideObjectsWireOnlyStale(t *testing.T) {
	existing := &Entry{
		ID:     "a",
		SHA256: "abc",
		Labels: []string{"dog"},
		Objects: &ObjectsResult{Status: "wire_only", Detections: []Detection{}},
	}
	want := Want{Objects: true}
	settings := Settings{Labels: []string{"dog"}, MinScore: 0.2}
	if d := Decide(existing, "abc", true, want, settings); d != DecisionStale {
		t.Fatalf("wire_only must be stale, got %s", d)
	}
	existing.Objects.Status = "ok"
	if d := Decide(existing, "abc", true, want, settings); d != DecisionKept {
		t.Fatalf("ok objects kept: %s", d)
	}
	settings.Labels = []string{"cat"}
	if d := Decide(existing, "abc", true, want, settings); d != DecisionStale {
		t.Fatalf("label change must redo objects: %s", d)
	}
}

func TestKeepTagsMinScore(t *testing.T) {
	labels := []string{"dog", "cat"}
	// unit vectors at known angles is overkill; use identical / orthogonal
	img := []float32{1, 0}
	vecs := [][]float32{{1, 0}, {0, 1}}
	got := KeepTags(labels, vecs, img, 0.5)
	if len(got) != 1 || got[0].Label != "dog" {
		t.Fatalf("got %#v", got)
	}
	got = KeepTags(labels, vecs, img, 1.1)
	if len(got) != 0 {
		t.Fatalf("expected empty, got %#v", got)
	}
}

func TestResolveLabels(t *testing.T) {
	labels, phrases := ResolveLabels(nil)
	if len(labels) != len(BuiltinLabels) || phrases[0] != "a photograph" {
		t.Fatalf("builtin: %v %v", labels, phrases)
	}
	labels, phrases = ResolveLabels([]string{" dog ", "dog", "beach"})
	if len(labels) != 2 || labels[0] != "dog" || phrases[1] != "beach" {
		t.Fatalf("custom: %v %v", labels, phrases)
	}
}
