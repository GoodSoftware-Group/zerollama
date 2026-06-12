package fleet

import (
	"fmt"
	"net/url"
	"strings"
)

// ParsePeers splits a comma-separated peer list into normalized base URLs.
// Why normalize: operators mix host:port and http:// forms; node_id stability depends on consistent URLs.
func ParsePeers(raw string) ([]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, fmt.Errorf("fleet peers list is empty")
	}

	var out []string
	seen := make(map[string]struct{})
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		u, err := normalizePeerURL(part)
		if err != nil {
			return nil, err
		}
		if _, ok := seen[u]; ok {
			continue
		}
		seen[u] = struct{}{}
		out = append(out, u)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("fleet peers list is empty")
	}
	return out, nil
}

func normalizePeerURL(raw string) (string, error) {
	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("invalid peer URL %q: %w", raw, err)
	}
	if u.Host == "" {
		return "", fmt.Errorf("invalid peer URL %q: missing host", raw)
	}
	if u.Scheme == "" {
		u.Scheme = "http"
	}
	u.Path = ""
	u.RawPath = ""
	u.RawQuery = ""
	u.Fragment = ""
	return strings.TrimSuffix(u.String(), "/"), nil
}

// NodeIDFromURL derives a stable node id from a peer base URL.
// Why host:port not UUID: v0 static config has no registration service; URL host is enough for exclude/retry.
func NodeIDFromURL(base string) string {
	u, err := url.Parse(base)
	if err != nil || u.Host == "" {
		return base
	}
	return u.Host
}
