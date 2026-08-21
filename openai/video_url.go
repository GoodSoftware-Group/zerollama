package openai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

const videoFetchMaxRedirects = 5

// videoFetchTransport is shared across remote video_url GETs; per-request deadlines use context.
// Why pooled: agent threads repeat the same HTTPS clip; cloning DefaultTransport preserves
// proxy/TLS behavior while MaxIdleConnsPerHost reuses connections.
var videoFetchTransport = func() *http.Transport {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = 60 * time.Second
	tr.MaxIdleConnsPerHost = 8
	return tr
}()

// Remote video fetch policy (HTTPS default, SSRF checks, transport cloning) lives in this file
// so it stays co-located and testable instead of spreading across the OpenAI mapper.

// ChatCompletionRequestHasVideoURL reports whether any message uses OpenAI video_url content parts.
func ChatCompletionRequestHasVideoURL(r *ChatCompletionRequest) bool {
	if r == nil {
		return false
	}
	for _, msg := range r.Messages {
		parts, ok := msg.Content.([]any)
		if !ok {
			continue
		}
		for _, p := range parts {
			m, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if t, _ := m["type"].(string); strings.EqualFold(t, "video_url") {
				return true
			}
		}
	}
	return false
}

// decodeVideoURL loads raw video bytes from a data URI or HTTP(S) URL.
func decodeVideoURL(ctx context.Context, rawURL string) (api.VideoData, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil, errors.New("empty video url")
	}

	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		return fetchVideoURL(ctx, rawURL)
	}

	if strings.HasPrefix(rawURL, "data:;base64,") {
		s := strings.TrimPrefix(rawURL, "data:;base64,")
		return decodeVideoBase64(s)
	}

	lower := strings.ToLower(rawURL)
	videoTypes := []string{"mp4", "webm", "quicktime", "x-matroska", "mpeg", "ogg"}
	for _, t := range videoTypes {
		prefix := "data:video/" + t + ";base64,"
		if strings.HasPrefix(lower, prefix) {
			s := rawURL[len(prefix):]
			return decodeVideoBase64(s)
		}
	}

	return nil, errors.New("invalid video input: use data:video/...;base64 or http(s) URL")
}

func decodeVideoBase64(s string) (api.VideoData, error) {
	s = strings.TrimSpace(s)
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		b, err = base64.RawStdEncoding.DecodeString(s)
	}
	if err != nil {
		return nil, fmt.Errorf("invalid video base64: %w", err)
	}
	if int64(len(b)) > envconfig.VideoMaxBytes() {
		return nil, fmt.Errorf("video exceeds max size (%d bytes)", envconfig.VideoMaxBytes())
	}
	return b, nil
}

func fetchVideoURL(ctx context.Context, rawURL string) (api.VideoData, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}
	if err := checkRemoteMediaURL(u); err != nil {
		return nil, err
	}

	if body, ok := lookupVideoURLFetchCache(rawURL); ok {
		slog.Info("video url fetch cache hit", "url_host", u.Hostname())
		return body, nil
	}

	if deadline := envconfig.VideoFetchTimeout(); deadline > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, deadline)
		defer cancel()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "video/*,*/*")

	client := &http.Client{
		Transport: videoFetchTransport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			// SGLang #34892: re-validate scheme/host/allowlist/SSRF on every hop so a
			// public CDN cannot bounce into loopback or off-allowlist hosts.
			if len(via) >= videoFetchMaxRedirects {
				return errors.New("video_url: too many redirects")
			}
			if req.URL == nil {
				return errors.New("video_url: redirect missing URL")
			}
			return checkRemoteMediaURL(req.URL)
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch video: status %d", resp.StatusCode)
	}

	maxBytes := envconfig.VideoMaxBytes()
	if cl := resp.ContentLength; cl > 0 && cl > maxBytes {
		return nil, fmt.Errorf("video exceeds max size (%d bytes)", maxBytes)
	}

	limited := io.LimitReader(resp.Body, maxBytes+1)
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > maxBytes {
		return nil, fmt.Errorf("video exceeds max size (%d bytes)", maxBytes)
	}
	rememberVideoURLFetchCache(rawURL, body)
	return body, nil
}

// checkRemoteMediaURL enforces scheme, optional allowlist, and SSRF host checks
// for the initial URL and every redirect target (SGLang #34892).
func checkRemoteMediaURL(u *url.URL) error {
	if u == nil {
		return errors.New("invalid video URL")
	}
	if u.Scheme != "https" && u.Scheme != "http" {
		return errors.New("unsupported video URL scheme")
	}
	if u.Scheme == "http" && !envconfig.VideoAllowInsecureHTTP() {
		return errors.New("video_url: use https for remote URLs, or set OLLAMA_VIDEO_ALLOW_INSECURE_HTTP=1")
	}
	host := u.Hostname()
	if host == "" {
		return errors.New("invalid video URL host")
	}
	if err := checkMediaHostAllowed(host); err != nil {
		return err
	}
	return verifyURLHostSafe(host)
}

func checkMediaHostAllowed(host string) error {
	allowed := envconfig.MediaAllowedHosts()
	if len(allowed) == 0 {
		return nil
	}
	h := strings.ToLower(strings.Trim(host, "[]"))
	for _, a := range allowed {
		if h == a || strings.HasSuffix(h, "."+a) {
			return nil
		}
	}
	return fmt.Errorf("video URL host %q not in OLLAMA_MEDIA_ALLOWED_HOSTS", host)
}

// verifyURLHostSafe rejects loopback and private addresses after DNS resolution.
// The subsequent TCP dial is not pinned to these IPs (DNS rebinding is not fully mitigated here).
func verifyURLHostSafe(host string) error {
	if host == "" {
		return errors.New("empty host")
	}
	if ip := net.ParseIP(strings.Trim(host, "[]")); ip != nil {
		if isBlockedIP(ip) {
			return errors.New("video URL resolves to a non-public address")
		}
		return nil
	}
	ips, err := net.LookupIP(host)
	if err != nil {
		return fmt.Errorf("video URL host lookup: %w", err)
	}
	if len(ips) == 0 {
		return errors.New("video URL host has no addresses")
	}
	for _, ip := range ips {
		if isBlockedIP(ip) {
			return errors.New("video URL resolves to a non-public address")
		}
	}
	return nil
}

func isBlockedIP(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast()
}
