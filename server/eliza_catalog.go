package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"golang.org/x/sync/singleflight"
)

// elizaModelsPath is Eliza's model list URL. It lives under /api/v1 (not bare /v1) so it aligns
// with the same prefix used after elizaUpstreamPath rewrites client /v1 routes.
const elizaModelsPath = "/api/v1/models"

// elizaOpenAIModelsResponse is the subset of Eliza GET /api/v1/models JSON we need.
type elizaOpenAIModelsResponse struct {
	Object string             `json:"object"`
	Data   []elizaOpenAIModel `json:"data"`
}

type elizaOpenAIModel struct {
	ID string `json:"id"`
}

var (
	elizaCatalogMu         sync.Mutex
	elizaCatalogCached     []api.ListModelResponse
	elizaCatalogFetched    time.Time
	elizaCatalogTTL        = time.Hour // default until a successful response supplies Cache-Control
	elizaCatalogClient     = &http.Client{Timeout: 30 * time.Second}
	elizaCatalogFetchGroup singleflight.Group // coalesces concurrent GET /api/v1/models fetches
)

func elizaCloudBaseURLParsed() (*url.URL, error) {
	return url.Parse(cloudProxyBaseURL)
}

// mergeElizaCloudModels appends remote Eliza models to local listings when cloud is enabled.
// Local rows win on duplicate model names (case-insensitive) so a user-defined :cloud tag does not
// appear twice if Eliza also returns the same id.
func mergeElizaCloudModels(ctx context.Context, local []api.ListModelResponse) []api.ListModelResponse {
	if envconfig.NoCloud() {
		return local
	}
	remote, err := fetchElizaModelList(ctx)
	if err != nil {
		elizaCatalogMu.Lock()
		stale := elizaCatalogCached
		elizaCatalogMu.Unlock()
		if len(stale) > 0 {
			slog.Warn("eliza catalog fetch failed, using stale cache", "error", err)
			remote = stale
		} else {
			slog.Debug("eliza catalog unavailable", "error", err)
			return local
		}
	}

	seen := make(map[string]struct{}, len(local)+len(remote))
	out := make([]api.ListModelResponse, 0, len(local)+len(remote))
	for _, m := range local {
		k := strings.ToLower(m.Model)
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, m)
	}
	for _, m := range remote {
		k := strings.ToLower(m.Model)
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, m)
	}
	return out
}

func fetchElizaModelList(ctx context.Context) ([]api.ListModelResponse, error) {
	elizaCatalogMu.Lock()
	if len(elizaCatalogCached) > 0 && time.Since(elizaCatalogFetched) < elizaCatalogTTL {
		out := elizaCatalogCached
		elizaCatalogMu.Unlock()
		return out, nil
	}
	elizaCatalogMu.Unlock()

	v, err, _ := elizaCatalogFetchGroup.Do("eliza-models", func() (any, error) {
		return fetchElizaModelListFromNetwork(ctx)
	})
	if err != nil {
		return nil, err
	}
	return v.([]api.ListModelResponse), nil
}

func fetchElizaModelListFromNetwork(ctx context.Context) ([]api.ListModelResponse, error) {
	elizaCatalogMu.Lock()
	if len(elizaCatalogCached) > 0 && time.Since(elizaCatalogFetched) < elizaCatalogTTL {
		out := elizaCatalogCached
		elizaCatalogMu.Unlock()
		return out, nil
	}
	elizaCatalogMu.Unlock()

	base, err := elizaCloudBaseURLParsed()
	if err != nil {
		return nil, err
	}
	target := base.ResolveReference(&url.URL{Path: elizaModelsPath})
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return nil, err
	}
	applyElizaOutboundAuth(req)

	resp, err := elizaCatalogClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		snippet := strings.TrimSpace(string(body))
		if len(snippet) > 200 {
			snippet = snippet[:200] + "…"
		}
		if snippet == "" {
			return nil, fmt.Errorf("eliza catalog: %s", resp.Status)
		}
		return nil, fmt.Errorf("eliza catalog: %s: %s", resp.Status, snippet)
	}

	var parsed elizaOpenAIModelsResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, err
	}

	ttl := cacheTTLFromCacheControl(resp.Header)
	now := time.Now().UTC()
	out := make([]api.ListModelResponse, 0, len(parsed.Data))
	for _, m := range parsed.Data {
		id := strings.TrimSpace(m.ID)
		if id == "" {
			continue
		}
		display := id + ":cloud"
		out = append(out, api.ListModelResponse{
			Name:        display,
			Model:       display,
			RemoteModel: id,
			RemoteHost:  strings.TrimSuffix(cloudProxyBaseURL, "/"),
			ModifiedAt:  now,
			Size:        0,
			Digest:      "",
			Details:     api.ModelDetails{Family: "cloud"},
		})
	}

	elizaCatalogMu.Lock()
	elizaCatalogCached = out
	elizaCatalogFetched = time.Now()
	elizaCatalogTTL = ttl
	elizaCatalogMu.Unlock()

	return out, nil
}

// cacheTTLFromCacheControl picks a sensible refresh interval from Eliza Cache-Control (defaults to 1h).
func cacheTTLFromCacheControl(h http.Header) time.Duration {
	const defaultTTL = time.Hour
	cc := h.Get("Cache-Control")
	if cc == "" {
		return defaultTTL
	}
	var best time.Duration
	for _, part := range strings.Split(cc, ",") {
		part = strings.TrimSpace(part)
		key, val, ok := strings.Cut(part, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(strings.ToLower(key))
		val = strings.TrimSpace(val)
		if key != "s-maxage" && key != "max-age" {
			continue
		}
		sec, err := strconv.Atoi(val)
		if err != nil || sec <= 0 {
			continue
		}
		d := time.Duration(sec) * time.Second
		if d > best {
			best = d
		}
	}
	if best == 0 {
		return defaultTTL
	}
	if best < time.Minute {
		return time.Minute
	}
	if best > 24*time.Hour {
		return 24 * time.Hour
	}
	return best
}
