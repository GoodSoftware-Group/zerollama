package fleet

import (
	"time"

	"github.com/ollama/ollama/api"
)

// AssignRequest is POST /api/fleet/assign.
type AssignRequest struct {
	Model             string   `json:"model"`
	PreferWarm        *bool    `json:"prefer_warm,omitempty"` // nil = true; cold load on busy fleet is the expensive path.
	WarmOnly          bool     `json:"warm_only,omitempty"`   // SLA: reject when no loaded peer (404).
	Exclude           []string `json:"exclude,omitempty"`     // Retry after F1 cancel-while-queued without re-picking same node.
	SessionKey        string   `json:"session_key,omitempty"` // L3 / agent thread id (eliza.conversationId, prompt_cache_key, …).
	PromptCacheKey    string   `json:"prompt_cache_key,omitempty"` // alias for SessionKey when agents send OpenAI-style options only to the node
	PrefixBlockHashes []string `json:"prefix_block_hashes,omitempty"` // L3-R9 / LA13: ordered hashes from token 0
}

// AssignResponse tells an agent which node to call directly.
// Why url + node_id: agents need the base URL for ollama client; node_id for exclude on retry.
type AssignResponse struct {
	URL              string     `json:"url"`
	NodeID           string     `json:"node_id"`
	Warm             bool       `json:"warm"`
	QueueDepth       int        `json:"queue_depth"`
	Loading          bool       `json:"loading,omitempty"`
	Score            float64    `json:"score,omitempty"`
	GeneratedAt      time.Time  `json:"generated_at"`
	AssignmentToken  string     `json:"assignment_token,omitempty"` // F5: present when ZEROLLAMA_FLEET_ASSIGN_SECRET set
	ExpiresAt        *time.Time `json:"expires_at,omitempty"`
	ExpiresIn        int        `json:"expires_in,omitempty"` // seconds remaining at mint
}

// WarmNodeEntry is one node carrying a loaded model.
type WarmNodeEntry struct {
	NodeID     string `json:"node_id"`
	URL        string `json:"url"`
	QueueDepth int    `json:"queue_depth"`
	Loading    bool   `json:"loading,omitempty"`
}

// WarmModelEntry maps a model to nodes that currently have it loaded.
type WarmModelEntry struct {
	Model string          `json:"model"`
	Nodes []WarmNodeEntry `json:"nodes"`
}

// NodeSnapshot is the cached view of one zerollama peer.
type NodeSnapshot struct {
	ID           string              `json:"node_id"`
	URL          string              `json:"url"`
	Available    bool                `json:"available"`
	LastPoll     time.Time           `json:"last_poll"`
	LastError    string              `json:"last_error,omitempty"`
	LoadedModels []string            `json:"loaded_models,omitempty"`
	QueueDepth   int                 `json:"queue_depth"`
	Loading      bool                `json:"loading,omitempty"`
	Inference    api.InferenceStatus `json:"inference,omitempty"`
}

// FleetStatusResponse is GET /api/fleet/status.
type FleetStatusResponse struct {
	Nodes       []NodeSnapshot   `json:"nodes"`
	WarmModels  []WarmModelEntry `json:"warm_models"`
	GeneratedAt time.Time        `json:"generated_at"`
}
