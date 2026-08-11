package media

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPutDedupeAndList(t *testing.T) {
	s := New(t.TempDir(), time.Hour, 1<<30)
	png := tinyPNG()

	r1, err := s.Put("sess1", "kf0", "image/png", bytes.NewReader(png))
	if err != nil {
		t.Fatal(err)
	}
	r2, err := s.Put("sess1", "kf1", "image/png", bytes.NewReader(png))
	if err != nil {
		t.Fatal(err)
	}
	if r1.Digest != r2.Digest {
		t.Fatalf("expected same digest, got %s vs %s", r1.Digest, r2.Digest)
	}
	if r1.Kind != KindImage {
		t.Fatalf("kind: %s", r1.Kind)
	}

	casEntries, err := os.ReadDir(s.casDir())
	if err != nil {
		t.Fatal(err)
	}
	nCAS := 0
	for _, e := range casEntries {
		name := e.Name()
		if !e.IsDir() && !strings.HasPrefix(name, "upload-") {
			nCAS++
		}
	}
	if nCAS != 1 {
		t.Fatalf("expected 1 CAS object, got %d", nCAS)
	}

	list, err := s.List("sess1")
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("list len=%d", len(list))
	}
}

func TestResolveMissing(t *testing.T) {
	s := New(t.TempDir(), time.Hour, 1<<30)
	_, err := s.Put("sess1", "kf0", "image/png", bytes.NewReader(tinyPNG()))
	if err != nil {
		t.Fatal(err)
	}
	_, _, missing, err := s.ResolveMany("sess1", []string{"kf0", "final"})
	if err != nil {
		t.Fatal(err)
	}
	if len(missing) != 1 || missing[0] != "final" {
		t.Fatalf("missing=%v", missing)
	}
}

func TestCASEvictedReportsMissing(t *testing.T) {
	s := New(t.TempDir(), time.Hour, 1<<30)
	r, err := s.Put("sess1", "kf0", "image/png", bytes.NewReader(tinyPNG()))
	if err != nil {
		t.Fatal(err)
	}
	casPath, err := s.casPath(r.Digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(casPath); err != nil {
		t.Fatal(err)
	}
	_, err = s.Head("sess1", "kf0")
	if err != ErrNotFound {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
}

func TestParseKeyframeRefs(t *testing.T) {
	session, labels, err := ParseKeyframeRefs(map[string]any{
		"media_session": "anim-1",
		"keyframes":     []any{"kf0", "final"},
	})
	if err != nil || session != "anim-1" || len(labels) != 2 {
		t.Fatalf("got %q %v err=%v", session, labels, err)
	}

	session, labels, err = ParseKeyframeRefs(map[string]any{
		"keyframes": []any{"anim-1/kf0", "anim-1/final"},
	})
	if err != nil || session != "anim-1" || labels[0] != "kf0" {
		t.Fatalf("composite: %q %v err=%v", session, labels, err)
	}
}

func TestMaterialize(t *testing.T) {
	s := New(t.TempDir(), time.Hour, 1<<30)
	if _, err := s.Put("s", "a", "image/png", bytes.NewReader(tinyPNG())); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Put("s", "b", "image/png", bytes.NewReader(tinyPNG())); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "out")
	missing, err := s.Materialize("s", []string{"a", "b"}, dest)
	if err != nil || len(missing) != 0 {
		t.Fatalf("missing=%v err=%v", missing, err)
	}
	ents, _ := os.ReadDir(dest)
	if len(ents) != 2 {
		t.Fatalf("files=%d", len(ents))
	}
}

// 1x1 PNG
func tinyPNG() []byte {
	return []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
		0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
		0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
		0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xfe, 0xd4, 0xef, 0x00, 0x00,
		0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}
}
