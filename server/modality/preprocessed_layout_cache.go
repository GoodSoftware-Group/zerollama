package modality

import (
	"crypto/sha256"
	"fmt"
	"log/slog"
	"time"

	"github.com/ollama/ollama/api"
)

const preprocessedLayoutKeyPrefix = "preprocessed:"

// preprocessedMessageFingerprint hashes single-clip pre-expanded content for session layout cache.
// Still images precede video frames in msg.Images (see docs/video-understanding.md).
func preprocessedMessageFingerprint(msg api.Message) string {
	if len(msg.VideoSpans) != 1 {
		return ""
	}
	sp := msg.VideoSpans[0]
	if sp.FrameCount <= 0 || sp.FrameCount > len(msg.Images) {
		return ""
	}
	stillImages := len(msg.Images) - sp.FrameCount
	var meta string
	if len(sp.GridTHW) == 3 {
		meta = fmt.Sprintf("fc:%d;grid:%d,%d,%d", sp.FrameCount, sp.GridTHW[0], sp.GridTHW[1], sp.GridTHW[2])
	} else {
		meta = fmt.Sprintf("fc:%d", sp.FrameCount)
	}
	h := sha256.New()
	h.Write([]byte(meta))
	for i := stillImages; i < len(msg.Images); i++ {
		sum := sha256.Sum256(msg.Images[i])
		h.Write(sum[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func sessionPreprocessedLayoutKey(msg api.Message) string {
	fp := preprocessedMessageFingerprint(msg)
	if fp == "" {
		return ""
	}
	return preprocessedLayoutKeyPrefix + fp
}

func rememberSessionPreprocessedLayout(sessionKey, layoutKey string, paddedInputIDs []int) {
	if sessionKey == "" || layoutKey == "" || len(paddedInputIDs) == 0 {
		return
	}
	now := time.Now().UTC()
	globalSessionVideoExpandCache.mu.Lock()
	defer globalSessionVideoExpandCache.mu.Unlock()
	st, ok := globalSessionVideoExpandCache.sessions[sessionKey]
	if !ok {
		if len(globalSessionVideoExpandCache.sessions) >= sessionVideoExpandMaxSessions {
			evictOldestSessionLocked(now)
		}
		st = sessionVideoExpandState{videos: make(map[string]videoExpandEntry)}
	}
	ensureSessionLayouts(&st)
	st.layouts[layoutKey] = append([]int(nil), paddedInputIDs...)
	st.updatedAt = now
	globalSessionVideoExpandCache.sessions[sessionKey] = st
}

func lookupSessionPreprocessedLayout(sessionKey, layoutKey string) ([]int, bool) {
	if sessionKey == "" || layoutKey == "" {
		return nil, false
	}
	now := time.Now().UTC()
	globalSessionVideoExpandCache.mu.Lock()
	defer globalSessionVideoExpandCache.mu.Unlock()
	st, ok := globalSessionVideoExpandCache.sessions[sessionKey]
	if !ok {
		return nil, false
	}
	if sessionVideoExpandTTL > 0 && now.Sub(st.updatedAt) > sessionVideoExpandTTL {
		delete(globalSessionVideoExpandCache.sessions, sessionKey)
		return nil, false
	}
	ids, ok := sessionVideoLayoutLocked(st, layoutKey)
	if !ok {
		return nil, false
	}
	st.updatedAt = now
	globalSessionVideoExpandCache.sessions[sessionKey] = st
	return ids, true
}

// maybeRestorePreprocessedLayout caches or restores padded_input_ids for SGLang pre-expanded
// messages (images + video_spans, no raw videos). Single-clip only; restore applies to latest user.
func maybeRestorePreprocessedLayout(sessionKey string, msgIdx, lastUser int, msg *api.Message) {
	if sessionKey == "" || len(msg.VideoSpans) == 0 {
		return
	}
	layoutKey := sessionPreprocessedLayoutKey(*msg)
	if layoutKey == "" {
		return
	}
	if len(msg.PaddedInputIDs) > 0 {
		rememberSessionPreprocessedLayout(sessionKey, layoutKey, msg.PaddedInputIDs)
		return
	}
	if msgIdx != lastUser {
		return
	}
	if ids, ok := lookupSessionPreprocessedLayout(sessionKey, layoutKey); ok {
		msg.PaddedInputIDs = ids
		slog.Info("preprocessed layout session cache hit",
			"session_key", sessionKey,
			"padded_input_ids_len", len(ids),
		)
	}
}
