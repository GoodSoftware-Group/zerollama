package cmd

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/describe"
)

// 1x1 PNG
var tinyPNG, _ = base64.StdEncoding.DecodeString(
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
)

func TestWalkImages(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.jpg"), tinyPNG, 0o644)
	_ = os.WriteFile(filepath.Join(dir, "skip.txt"), []byte("x"), 0o644)
	_ = os.Mkdir(filepath.Join(dir, ".hidden"), 0o755)
	_ = os.WriteFile(filepath.Join(dir, ".hidden", "b.png"), tinyPNG, 0o644)
	_ = os.WriteFile(filepath.Join(dir, "shot.heic"), []byte("heic"), 0o644)

	out := filepath.Join(dir, "photos.json")
	files, err := walkImages(dir, out)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, f := range files {
		got[f.rel] = true
	}
	if !got["a.jpg"] || !got["shot.heic"] || got["skip.txt"] || got[".hidden/b.png"] {
		t.Fatalf("%v", got)
	}
}

func TestDescribeResumeAndNewAndRemoved(t *testing.T) {
	dir := t.TempDir()
	aPath := filepath.Join(dir, "a.jpg")
	if err := os.WriteFile(aPath, tinyPNG, 0o644); err != nil {
		t.Fatal(err)
	}
	outPath := dir + ".json"

	var describeCalls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/tags":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"models": []map[string]any{{
					"name": "siglip2",
					"details": map[string]any{
						"family": "siglip2",
					},
				}},
			})
		case r.URL.Path == "/api/unload":
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "siglip2", "message": "unloaded"})
		case r.URL.Path == "/api/describe":
			describeCalls.Add(1)
			var raw map[string]any
			if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
				http.Error(w, err.Error(), 400)
				return
			}
			// Empty labels means builtin phrases on the server.
			if labels, ok := raw["labels"].([]any); ok && len(labels) > 0 {
				t.Fatalf("default run must not send custom labels, got %v", labels)
			}
			imgs, _ := raw["images"].([]any)
			out := describe.Response{Images: make([]describe.ResultEntry, 0, len(imgs))}
			for _, it := range imgs {
				m := it.(map[string]any)
				id, _ := m["id"].(string)
				tags := []describe.TagScore{{Label: "photograph", Score: 0.5}}
				sha := describe.HashBytes(tinyPNG)
				if dataStr, ok := m["data"].(string); ok && dataStr != "" {
					rawBytes, _ := base64.StdEncoding.DecodeString(dataStr)
					sha = describe.HashBytes(rawBytes)
				}
				e := describe.Entry{
					ID:       id,
					SHA256:   sha,
					TagModel: "siglip2",
					Labels:   describe.LabelNames(),
					MinScore: 0.2,
					File:     &describe.FileFacts{Bytes: int64(len(tinyPNG))},
				}
				e.SetTags(tags)
				out.Images = append(out.Images, describe.ResultEntry{Entry: e, Action: describe.ActionProcessed})
			}
			_ = json.NewEncoder(w).Encode(out)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	if err := runDescribe(cmd, dir, describeFlags{
		OutPath:  outPath,
		MinScore: 0.2,
		TagModel: "siglip2",
	}); err != nil {
		t.Fatal(err)
	}
	if describeCalls.Load() != 1 {
		t.Fatalf("first run calls=%d", describeCalls.Load())
	}
	cat, err := describe.LoadCatalog(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Images) != 1 || cat.Images[0].ID != "a.jpg" {
		t.Fatalf("%#v", cat.Images)
	}

	if err := runDescribe(cmd, dir, describeFlags{
		OutPath:  outPath,
		MinScore: 0.2,
		TagModel: "siglip2",
	}); err != nil {
		t.Fatal(err)
	}
	if describeCalls.Load() != 1 {
		t.Fatalf("resume should not re-describe, calls=%d", describeCalls.Load())
	}

	// new image
	if err := os.WriteFile(filepath.Join(dir, "b.jpg"), tinyPNG, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runDescribe(cmd, dir, describeFlags{
		OutPath:  outPath,
		MinScore: 0.2,
		TagModel: "siglip2",
	}); err != nil {
		t.Fatal(err)
	}
	if describeCalls.Load() != 2 {
		t.Fatalf("new file should describe once, calls=%d", describeCalls.Load())
	}
	cat, _ = describe.LoadCatalog(outPath)
	if len(cat.Images) != 2 {
		t.Fatalf("want 2 images, got %d", len(cat.Images))
	}

	// remove b
	_ = os.Remove(filepath.Join(dir, "b.jpg"))
	if err := runDescribe(cmd, dir, describeFlags{
		OutPath:  outPath,
		MinScore: 0.2,
		TagModel: "siglip2",
	}); err != nil {
		t.Fatal(err)
	}
	cat, _ = describe.LoadCatalog(outPath)
	if len(cat.Images) != 1 || cat.Images[0].ID != "a.jpg" {
		t.Fatalf("removed path should drop: %#v", cat.Images)
	}
}

func TestDescribeHEICErrorCheckpoint(t *testing.T) {
	dir := t.TempDir()
	heicPath := filepath.Join(dir, "shot.heic")
	if err := os.WriteFile(heicPath, []byte("not-a-real-heic"), 0o644); err != nil {
		t.Fatal(err)
	}
	outPath := dir + ".json"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"models": []map[string]any{{"name": "siglip2", "details": map[string]any{"family": "siglip2"}}},
			})
			return
		}
		if r.URL.Path == "/api/describe" {
			t.Fatal("HEIC failure should not call describe")
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)
	// Hide sips so HEIC cannot convert.
	t.Setenv("PATH", t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	if err := runDescribe(cmd, dir, describeFlags{OutPath: outPath, MinScore: 0.2, TagModel: "siglip2"}); err != nil {
		t.Fatal(err)
	}
	cat, err := describe.LoadCatalog(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Images) != 1 || !strings.Contains(cat.Images[0].Error, "unsupported") {
		t.Fatalf("expected unsupported error: %#v", cat.Images[0])
	}
	// Second run must skip the same unsupported file (no describe call).
	if err := runDescribe(cmd, dir, describeFlags{OutPath: outPath, MinScore: 0.2, TagModel: "siglip2"}); err != nil {
		t.Fatal(err)
	}
}

func TestLoadImageBytesUnsupported(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.heic")
	_ = os.WriteFile(path, []byte("x"), 0o644)
	t.Setenv("PATH", dir)
	_, err := loadImageBytes(path, ".heic")
	if err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("got %v", err)
	}
	_ = time.Now
}

func TestNewDescribeCommand(t *testing.T) {
	c := NewDescribeCommand()
	if c.Use != "describe DIRECTORY" {
		t.Fatalf("%s", c.Use)
	}
}
