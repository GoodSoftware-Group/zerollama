// Package media implements a session/label media index with content-addressed
// storage for agent uploads (keyframes, future clips).
//
// WHY not $OLLAMA_MODELS/blobs: those are permanent model layers without session
// namespaces or TTL. Animation frames are soft state — agents re-PUT on miss.
//
// WHY no client digests: the server hashes PUT bodies so clients cannot claim a
// digest they did not upload; recovery is re-upload by label, not "fix hash".
//
// WHY no refcounts: pin/unpin across the training queue is easy to get wrong for
// disposable frames. TTL + CAS byte-cap LRU + media_missing on create is enough.
// See docs/media-uploads.md.
package media

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
)

const (
	// MaxImageBytes caps a single image PUT (~25 MiB).
	// WHY: keyframes/stills; keeps abuse and peak RAM bounded without blocking HD PNGs.
	MaxImageBytes = 25 << 20
	// MaxVideoBytes caps a single video PUT (future clips / morph inputs).
	// WHY larger than images: short morph clips are bigger; still far below model blobs.
	MaxVideoBytes = 256 << 20
	// MaxObjectBytes is the absolute stream cap (video-sized).
	MaxObjectBytes = MaxVideoBytes
	// MaxVideoCreateBody caps POST /v1/videos JSON (label refs only).
	// WHY: force large frames onto PUT /v1/media instead of base64-in-JSON.
	MaxVideoCreateBody = 8 << 20

	defaultTTL    = 24 * time.Hour
	defaultMaxCAS = 10 << 30 // 10 GiB
	labelMaxLen   = 128
	sessionMaxLen = 128
	digestPrefix  = "sha256-"
)

var (
	ErrInvalidSession = errors.New("invalid media session")
	ErrInvalidLabel   = errors.New("invalid media label")
	ErrNotFound       = errors.New("media not found")
	ErrTooLarge       = errors.New("media object exceeds size limit")
	ErrEmptyBody      = errors.New("empty media body")
)

var (
	sessionRe = regexp.MustCompile(`^[A-Za-z0-9._-]{1,128}$`)
	labelRe   = regexp.MustCompile(`^[A-Za-z0-9._-]{1,128}$`)
)

// Kind classifies stored media for runner validation.
type Kind string

const (
	KindImage Kind = "image"
	KindVideo Kind = "video"
	KindOther Kind = "other"
)

// Meta is stored per session/label pointer.
type Meta struct {
	Digest      string    `json:"digest"`
	Size        int64     `json:"size"`
	ContentType string    `json:"content_type"`
	Kind        Kind      `json:"kind"`
	CreatedAt   time.Time `json:"created_at"`
	AccessedAt  time.Time `json:"accessed_at"`
}

// LabelInfo is returned by session list.
type LabelInfo struct {
	Label       string `json:"label"`
	Digest      string `json:"digest"`
	Bytes       int64  `json:"bytes"`
	ContentType string `json:"content_type"`
	Kind        Kind   `json:"kind"`
}

// PutResult is the PUT response body.
type PutResult struct {
	Session string `json:"session"`
	Label   string `json:"label"`
	Digest  string `json:"digest"`
	Bytes   int64  `json:"bytes"`
	Kind    Kind   `json:"kind"`
}

// Store is a process-local media index + CAS under $OLLAMA_MODELS/media.
type Store struct {
	mu  sync.Mutex
	root string
	ttl  time.Duration
	maxCAS int64
}

// Default returns a store rooted at $OLLAMA_MODELS/media.
func Default() *Store {
	return New(filepath.Join(envconfig.Models(), "media"), defaultTTL, defaultMaxCAS)
}

// New constructs a Store. Tests pass a temp root.
func New(root string, ttl time.Duration, maxCAS int64) *Store {
	if ttl <= 0 {
		ttl = defaultTTL
	}
	if maxCAS <= 0 {
		maxCAS = defaultMaxCAS
	}
	return &Store{root: root, ttl: ttl, maxCAS: maxCAS}
}

func (s *Store) casDir() string      { return filepath.Join(s.root, "cas") }
func (s *Store) sessionsDir() string { return filepath.Join(s.root, "sessions") }

func ValidateSession(session string) error {
	if !sessionRe.MatchString(session) {
		return ErrInvalidSession
	}
	return nil
}

func ValidateLabel(label string) error {
	if !labelRe.MatchString(label) {
		return ErrInvalidLabel
	}
	return nil
}

func (s *Store) casPath(digest string) (string, error) {
	if !strings.HasPrefix(digest, digestPrefix) || len(digest) != len(digestPrefix)+64 {
		return "", fmt.Errorf("invalid digest %q", digest)
	}
	hexPart := digest[len(digestPrefix):]
	if _, err := hex.DecodeString(hexPart); err != nil {
		return "", fmt.Errorf("invalid digest %q", digest)
	}
	return filepath.Join(s.casDir(), digest), nil
}

func (s *Store) metaPath(session, label string) string {
	return filepath.Join(s.sessionsDir(), session, label+".json")
}

// Put streams r into CAS (deduped by SHA-256) and writes the session/label pointer.
// WHY CAS: identical frames across labels/retries share one blob. WHY pointer JSON:
// agents address by session/label; digests are an implementation detail in responses.
func (s *Store) Put(session, label, contentType string, r io.Reader) (PutResult, error) {
	if err := ValidateSession(session); err != nil {
		return PutResult{}, err
	}
	if err := ValidateLabel(label); err != nil {
		return PutResult{}, err
	}

	if err := os.MkdirAll(s.casDir(), 0o755); err != nil {
		return PutResult{}, err
	}
	if err := os.MkdirAll(filepath.Join(s.sessionsDir(), session), 0o755); err != nil {
		return PutResult{}, err
	}

	tmp, err := os.CreateTemp(s.casDir(), "upload-*.tmp")
	if err != nil {
		return PutResult{}, err
	}
	tmpName := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}()

	h := sha256.New()
	limited := io.LimitReader(r, MaxObjectBytes+1)
	n, err := io.Copy(io.MultiWriter(tmp, h), limited)
	if err != nil {
		return PutResult{}, err
	}
	if n == 0 {
		return PutResult{}, ErrEmptyBody
	}
	if n > MaxObjectBytes {
		return PutResult{}, ErrTooLarge
	}
	if err := tmp.Sync(); err != nil {
		return PutResult{}, err
	}
	if err := tmp.Close(); err != nil {
		return PutResult{}, err
	}

	digest := digestPrefix + hex.EncodeToString(h.Sum(nil))
	casPath, err := s.casPath(digest)
	if err != nil {
		return PutResult{}, err
	}

	// Sniff kind from head of file (and Content-Type header).
	kind, ct := classifyMedia(tmpName, contentType)
	if kind == KindImage && n > MaxImageBytes {
		return PutResult{}, fmt.Errorf("%w: images limited to %d bytes (got %d); use /v1/media for video clips up to %d", ErrTooLarge, MaxImageBytes, n, MaxVideoBytes)
	}
	if kind != KindImage && kind != KindVideo && n > MaxImageBytes {
		// Unknown/other: keep the tighter image budget.
		return PutResult{}, fmt.Errorf("%w: object limited to %d bytes for kind %q", ErrTooLarge, MaxImageBytes, kind)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, err := os.Stat(casPath); errors.Is(err, os.ErrNotExist) {
		if err := os.Rename(tmpName, casPath); err != nil {
			// Another writer may have won; fall through if dest exists.
			if _, err2 := os.Stat(casPath); err2 != nil {
				return PutResult{}, err
			}
			_ = os.Remove(tmpName)
		}
	} else if err != nil {
		return PutResult{}, err
	} else {
		_ = os.Remove(tmpName)
	}

	now := time.Now().UTC()
	meta := Meta{
		Digest:      digest,
		Size:        n,
		ContentType: ct,
		Kind:        kind,
		CreatedAt:   now,
		AccessedAt:  now,
	}
	if err := writeMeta(s.metaPath(session, label), meta); err != nil {
		return PutResult{}, err
	}

	s.evictLocked()

	return PutResult{
		Session: session,
		Label:   label,
		Digest:  digest,
		Bytes:   n,
		Kind:    kind,
	}, nil
}

func classifyMedia(path, headerCT string) (Kind, string) {
	ct := strings.TrimSpace(strings.Split(headerCT, ";")[0])
	f, err := os.Open(path)
	if err == nil {
		defer f.Close()
		buf := make([]byte, 512)
		n, _ := f.Read(buf)
		if n > 0 {
			detected := http.DetectContentType(buf[:n])
			if ct == "" || ct == "application/octet-stream" {
				ct = detected
			}
			// Prefer sniff for kind when header is generic.
			if strings.HasPrefix(detected, "image/") {
				return KindImage, preferCT(ct, detected)
			}
			if strings.HasPrefix(detected, "video/") {
				return KindVideo, preferCT(ct, detected)
			}
		}
	}
	switch {
	case strings.HasPrefix(ct, "image/"):
		return KindImage, ct
	case strings.HasPrefix(ct, "video/"):
		return KindVideo, ct
	case ct == "":
		return KindOther, "application/octet-stream"
	default:
		return KindOther, ct
	}
}

func preferCT(header, detected string) string {
	if header != "" && header != "application/octet-stream" {
		return header
	}
	return detected
}

func writeMeta(path string, meta Meta) error {
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readMeta(path string) (Meta, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Meta{}, err
	}
	var meta Meta
	if err := json.Unmarshal(b, &meta); err != nil {
		return Meta{}, err
	}
	return meta, nil
}

// Head returns meta if the label exists and CAS is present (and not TTL-expired).
func (s *Store) Head(session, label string) (Meta, error) {
	if err := ValidateSession(session); err != nil {
		return Meta{}, err
	}
	if err := ValidateLabel(label); err != nil {
		return Meta{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.resolveLocked(session, label, true)
}

// GetPath returns an absolute CAS path for reading bytes.
func (s *Store) GetPath(session, label string) (string, Meta, error) {
	if err := ValidateSession(session); err != nil {
		return "", Meta{}, err
	}
	if err := ValidateLabel(label); err != nil {
		return "", Meta{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	meta, err := s.resolveLocked(session, label, true)
	if err != nil {
		return "", Meta{}, err
	}
	p, err := s.casPath(meta.Digest)
	if err != nil {
		return "", Meta{}, err
	}
	return p, meta, nil
}

// List returns labels in a session (skips expired / dangling).
func (s *Store) List(session string) ([]LabelInfo, error) {
	if err := ValidateSession(session); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	dir := filepath.Join(s.sessionsDir(), session)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []LabelInfo{}, nil
		}
		return nil, err
	}
	out := make([]LabelInfo, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		label := strings.TrimSuffix(e.Name(), ".json")
		meta, err := s.resolveLocked(session, label, true)
		if err != nil {
			continue
		}
		out = append(out, LabelInfo{
			Label:       label,
			Digest:      meta.Digest,
			Bytes:       meta.Size,
			ContentType: meta.ContentType,
			Kind:        meta.Kind,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Label < out[j].Label })
	return out, nil
}

// Delete removes the session/label pointer (CAS may linger until LRU).
// WHY not delete CAS immediately: another session label may still point at the
// same digest; without refcounts we rely on byte-cap LRU instead of eager unlink.
func (s *Store) Delete(session, label string) error {
	if err := ValidateSession(session); err != nil {
		return err
	}
	if err := ValidateLabel(label); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	path := s.metaPath(session, label)
	if err := os.Remove(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return ErrNotFound
		}
		return err
	}
	return nil
}

// ResolveMany looks up labels under session; returns missing label names.
func (s *Store) ResolveMany(session string, labels []string) (paths []string, metas []Meta, missing []string, err error) {
	if err := ValidateSession(session); err != nil {
		return nil, nil, nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	paths = make([]string, 0, len(labels))
	metas = make([]Meta, 0, len(labels))
	for _, label := range labels {
		if err := ValidateLabel(label); err != nil {
			missing = append(missing, label)
			continue
		}
		meta, err := s.resolveLocked(session, label, true)
		if err != nil {
			missing = append(missing, label)
			continue
		}
		p, err := s.casPath(meta.Digest)
		if err != nil {
			missing = append(missing, label)
			continue
		}
		paths = append(paths, p)
		metas = append(metas, meta)
	}
	return paths, metas, missing, nil
}

func (s *Store) resolveLocked(session, label string, touch bool) (Meta, error) {
	path := s.metaPath(session, label)
	meta, err := readMeta(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Meta{}, ErrNotFound
		}
		return Meta{}, err
	}
	if time.Since(meta.AccessedAt) > s.ttl && time.Since(meta.CreatedAt) > s.ttl {
		_ = os.Remove(path)
		return Meta{}, ErrNotFound
	}
	casPath, err := s.casPath(meta.Digest)
	if err != nil {
		return Meta{}, ErrNotFound
	}
	if _, err := os.Stat(casPath); err != nil {
		_ = os.Remove(path)
		return Meta{}, ErrNotFound
	}
	if touch {
		meta.AccessedAt = time.Now().UTC()
		_ = writeMeta(path, meta)
	}
	return meta, nil
}

// Materialize copies/hardlinks resolved CAS objects into destDir as 000.ext, 001.ext, …
// WHY: Wan jobs run for tens of minutes; CAS LRU may delete the original blob.
// Staging under generated/keyframes/ freezes stable paths for the subprocess.
func (s *Store) Materialize(session string, labels []string, destDir string) (missing []string, err error) {
	paths, metas, missing, err := s.ResolveMany(session, labels)
	if err != nil {
		return nil, err
	}
	if len(missing) > 0 {
		return missing, nil
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, err
	}
	for i, src := range paths {
		ext := extFor(metas[i])
		dst := filepath.Join(destDir, fmt.Sprintf("%03d%s", i, ext))
		if err := linkOrCopy(src, dst); err != nil {
			return nil, err
		}
	}
	return nil, nil
}

func extFor(m Meta) string {
	switch {
	case strings.Contains(m.ContentType, "png"):
		return ".png"
	case strings.Contains(m.ContentType, "jpeg"), strings.Contains(m.ContentType, "jpg"):
		return ".jpg"
	case strings.Contains(m.ContentType, "webp"):
		return ".webp"
	case strings.Contains(m.ContentType, "mp4"):
		return ".mp4"
	case strings.Contains(m.ContentType, "webm"):
		return ".webm"
	case m.Kind == KindVideo:
		return ".mp4"
	default:
		return ".bin"
	}
}

func linkOrCopy(src, dst string) error {
	// Prefer hardlink: zero extra bytes when CAS and staging share a filesystem.
	if err := os.Link(src, dst); err == nil {
		return nil
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

type casEntry struct {
	path    string
	size    int64
	modTime time.Time
}

func (s *Store) evictLocked() {
	entries, err := os.ReadDir(s.casDir())
	if err != nil {
		return
	}
	var list []casEntry
	var total int64
	for _, e := range entries {
		if e.IsDir() || strings.HasPrefix(e.Name(), "upload-") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		list = append(list, casEntry{
			path:    filepath.Join(s.casDir(), e.Name()),
			size:    info.Size(),
			modTime: info.ModTime(),
		})
		total += info.Size()
	}
	if total <= s.maxCAS {
		return
	}
	sort.Slice(list, func(i, j int) bool { return list[i].modTime.Before(list[j].modTime) })
	for _, e := range list {
		if total <= s.maxCAS {
			break
		}
		if err := os.Remove(e.path); err == nil {
			total -= e.size
		}
	}
}

func optionString(opts map[string]any, key string) string {
	if opts == nil {
		return ""
	}
	v, ok := opts[key]
	if !ok || v == nil {
		return ""
	}
	return strings.TrimSpace(fmt.Sprint(v))
}

// OptionString is the exported form of optionString (video last-frame checks).
func OptionString(opts map[string]any, key string) string {
	return optionString(opts, key)
}

// ParseKeyframeRefs resolves options.media_session + options.keyframes into session + labels.
// Also accepts "session/label" entries when media_session is empty.
// mlx-serve aliases: first_frame_image / last_frame_image as ordered stills when
// keyframes is omitted.
// WHY two shapes: agents that already namespace labels as "sid/kf0" need not
// duplicate the session field; media_session + bare labels is the common case.
func ParseKeyframeRefs(opts map[string]any) (session string, labels []string, err error) {
	if opts == nil {
		return "", nil, nil
	}
	if v, ok := opts["media_session"]; ok {
		session = strings.TrimSpace(fmt.Sprint(v))
	}
	raw, ok := opts["keyframes"]
	if !ok || raw == nil {
		if first := optionString(opts, "first_frame_image"); first != "" {
			labels = append(labels, first)
		}
		if last := optionString(opts, "last_frame_image"); last != "" {
			labels = append(labels, last)
		}
		if len(labels) == 0 {
			return session, nil, nil
		}
	} else {
		switch t := raw.(type) {
		case []any:
			for _, item := range t {
				labels = append(labels, strings.TrimSpace(fmt.Sprint(item)))
			}
		case []string:
			labels = append(labels, t...)
		default:
			return "", nil, fmt.Errorf("options.keyframes must be an array of strings")
		}
		if last := optionString(opts, "last_frame_image"); last != "" && len(labels) == 1 && labels[0] != last {
			labels = append(labels, last)
		}
		if first := optionString(opts, "first_frame_image"); first != "" && len(labels) == 0 {
			labels = append(labels, first)
		}
		if len(labels) == 0 {
			return session, nil, nil
		}
	}

	// Normalize session/label composite refs.
	out := make([]string, 0, len(labels))
	for _, ref := range labels {
		if ref == "" {
			return "", nil, fmt.Errorf("empty keyframe label")
		}
		if strings.Contains(ref, "/") {
			parts := strings.SplitN(ref, "/", 2)
			if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
				return "", nil, fmt.Errorf("invalid keyframe ref %q", ref)
			}
			if session == "" {
				session = parts[0]
			} else if session != parts[0] {
				return "", nil, fmt.Errorf("keyframes span multiple media sessions")
			}
			out = append(out, parts[1])
			continue
		}
		out = append(out, ref)
	}
	if session == "" {
		return "", nil, fmt.Errorf("options.media_session is required when keyframes are bare labels")
	}
	if err := ValidateSession(session); err != nil {
		return "", nil, err
	}
	for _, l := range out {
		if err := ValidateLabel(l); err != nil {
			return "", nil, fmt.Errorf("invalid keyframe label %q", l)
		}
	}
	return session, out, nil
}
