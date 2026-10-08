package server

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/openai"
)

// videoArtifactMeta is a durable sidecar next to {id}.mp4 under $OLLAMA_MODELS/generated/.
//
// WHY: the training JobQueue is in-process memory. After progress=100 Wan2GP teardown
// (or an OOM restart) can wipe the job record while the mp4 remains — clients then see
// GET /v1/videos/{id} → 404 "video job not found" with no way to recover /content.
// Persist meta at completion so status+content survive queue eviction and serve restarts
// until ZEROLLAMA_VIDEO_ARTIFACT_TTL (default 24h).
type videoArtifactMeta struct {
	ID        string  `json:"id"`
	Object    string  `json:"object"`
	CreatedAt int64   `json:"created_at"`
	Status    string  `json:"status"`
	Model     string  `json:"model,omitempty"`
	Progress  float64 `json:"progress,omitempty"`
	Size      string  `json:"size,omitempty"`
	Output    string  `json:"output_path,omitempty"`
	ExpiresAt int64   `json:"expires_at,omitempty"`
}

func videoArtifactMetaPath(jobID string) string {
	return filepath.Join(videoArtifactRoot(), jobID+".json")
}

func videoArtifactTTL() time.Duration {
	raw := strings.TrimSpace(envconfig.Var("ZEROLLAMA_VIDEO_ARTIFACT_TTL"))
	if raw == "" {
		return 24 * time.Hour
	}
	if sec, err := strconv.Atoi(raw); err == nil && sec > 0 {
		return time.Duration(sec) * time.Second
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return 24 * time.Hour
}

func writeVideoArtifactMeta(v openai.Video, outputPath string) error {
	id := strings.TrimSpace(v.ID)
	if id == "" {
		return errors.New("empty video id")
	}
	if err := os.MkdirAll(videoArtifactRoot(), 0o755); err != nil {
		return err
	}
	ttl := videoArtifactTTL()
	meta := videoArtifactMeta{
		ID:        id,
		Object:    "video",
		CreatedAt: v.CreatedAt,
		Status:    "completed",
		Model:     v.Model,
		Progress:  100,
		Size:      v.Size,
		Output:    outputPath,
		ExpiresAt: time.Now().Add(ttl).Unix(),
	}
	if meta.CreatedAt <= 0 {
		meta.CreatedAt = time.Now().Unix()
	}
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	tmp := videoArtifactMetaPath(id) + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, videoArtifactMetaPath(id))
}

// persistCompletedVideoArtifact writes/refreshes the sidecar when a live job hits completed.
func persistCompletedVideoArtifact(v openai.Video, outputPath string) {
	if strings.TrimSpace(v.ID) == "" || strings.TrimSpace(outputPath) == "" {
		return
	}
	if st, err := os.Stat(outputPath); err != nil || st.Size() < 1 {
		return
	}
	_ = writeVideoArtifactMeta(v, outputPath)
}

func loadVideoArtifactMeta(jobID string) (videoArtifactMeta, string, error) {
	id := strings.TrimSpace(jobID)
	if id == "" || strings.Contains(id, "..") || strings.ContainsAny(id, `/\`) {
		return videoArtifactMeta{}, "", errors.New("invalid job id")
	}
	mp4 := videoArtifactPath(id)
	st, err := os.Stat(mp4)
	if err != nil || st.Size() < 1 {
		return videoArtifactMeta{}, "", errors.New("artifact missing")
	}
	metaPath := videoArtifactMetaPath(id)
	b, err := os.ReadFile(metaPath)
	var meta videoArtifactMeta
	if err == nil {
		if err := json.Unmarshal(b, &meta); err != nil {
			meta = videoArtifactMeta{}
		}
	}
	if meta.ID == "" {
		meta.ID = id
	}
	meta.Object = "video"
	meta.Status = "completed"
	meta.Progress = 100
	if meta.CreatedAt <= 0 {
		meta.CreatedAt = st.ModTime().Unix()
	}
	if meta.Output == "" {
		meta.Output = mp4
	}
	if videoArtifactExpired(meta, st.ModTime()) {
		return videoArtifactMeta{}, "", errors.New("artifact expired")
	}
	return meta, mp4, nil
}

func videoArtifactExpired(meta videoArtifactMeta, mtime time.Time) bool {
	if meta.ExpiresAt > 0 {
		return time.Now().Unix() > meta.ExpiresAt
	}
	return time.Since(mtime) > videoArtifactTTL()
}

func videoFromDiskArtifact(jobID string) (openai.Video, string, bool) {
	meta, path, err := loadVideoArtifactMeta(jobID)
	if err != nil {
		return openai.Video{}, "", false
	}
	return openai.Video{
		ID:        meta.ID,
		Object:    "video",
		CreatedAt: meta.CreatedAt,
		Status:    "completed",
		Model:     meta.Model,
		Progress:  100,
		Size:      meta.Size,
	}, path, true
}

// deleteVideoArtifact removes {id}.mp4 and {id}.json under generated/.
// Idempotent: missing files are success (client already cleaned up / TTL prune raced).
func deleteVideoArtifact(jobID string) error {
	id := strings.TrimSpace(jobID)
	if id == "" || strings.Contains(id, "..") || strings.ContainsAny(id, `/\`) {
		return errors.New("invalid job id")
	}
	mp4 := videoArtifactPath(id)
	meta := videoArtifactMetaPath(id)
	var first error
	for _, p := range []string{mp4, meta} {
		if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
			if first == nil {
				first = err
			}
		}
	}
	return first
}

func videoArtifactPruneInterval() time.Duration {
	raw := strings.TrimSpace(envconfig.Var("ZEROLLAMA_VIDEO_ARTIFACT_PRUNE_INTERVAL"))
	if raw == "" || raw == "0" || strings.EqualFold(raw, "off") {
		return 15 * time.Minute
	}
	if sec, err := strconv.Atoi(raw); err == nil {
		if sec <= 0 {
			return 0 // disabled
		}
		return time.Duration(sec) * time.Second
	}
	if d, err := time.ParseDuration(raw); err == nil {
		return d
	}
	return 15 * time.Minute
}

// pruneExpiredVideoArtifacts deletes completed mp4+json past TTL (unclaimed leftovers).
// Returns count of jobs removed.
func pruneExpiredVideoArtifacts() int {
	root := videoArtifactRoot()
	ents, err := os.ReadDir(root)
	if err != nil {
		return 0
	}
	now := time.Now()
	removed := 0
	seen := map[string]struct{}{}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		ext := filepath.Ext(name)
		if ext != ".mp4" && ext != ".json" {
			continue
		}
		id := strings.TrimSuffix(name, ext)
		if id == "" || strings.Contains(id, "..") {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}

		mp4 := videoArtifactPath(id)
		st, err := os.Stat(mp4)
		metaPath := videoArtifactMetaPath(id)
		var meta videoArtifactMeta
		if b, rerr := os.ReadFile(metaPath); rerr == nil {
			_ = json.Unmarshal(b, &meta)
		}
		expired := false
		switch {
		case err == nil:
			expired = videoArtifactExpired(meta, st.ModTime())
		case errors.Is(err, os.ErrNotExist):
			// Orphan sidecar: expire by meta or file mtime.
			if meta.ExpiresAt > 0 {
				expired = now.Unix() > meta.ExpiresAt
			} else if mst, merr := os.Stat(metaPath); merr == nil {
				expired = now.Sub(mst.ModTime()) > videoArtifactTTL()
			} else {
				expired = true
			}
		default:
			continue
		}
		if !expired {
			continue
		}
		if err := deleteVideoArtifact(id); err != nil {
			slog.Warn("video artifact prune failed", "id", id, "error", err)
			continue
		}
		removed++
		slog.Info("video artifact pruned", "id", id, "ttl", videoArtifactTTL().String())
	}
	return removed
}

// runVideoArtifactPruner periodically deletes unclaimed/expired generated videos.
func (s *Server) runVideoArtifactPruner(ctx context.Context) {
	interval := videoArtifactPruneInterval()
	if interval <= 0 {
		slog.Info("video artifact prune disabled", "env", "ZEROLLAMA_VIDEO_ARTIFACT_PRUNE_INTERVAL")
		return
	}
	// First pass shortly after boot so restarts reclaim disk without waiting a full interval.
	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	slog.Info("video artifact prune started", "interval", interval.String(), "ttl", videoArtifactTTL().String())
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			if n := pruneExpiredVideoArtifacts(); n > 0 {
				slog.Info("video artifact prune pass", "removed", n)
			}
		case <-ticker.C:
			if n := pruneExpiredVideoArtifacts(); n > 0 {
				slog.Info("video artifact prune pass", "removed", n)
			}
		}
	}
}
