package llm

import "context"

// MultiDecodeRequest is a forest decode against llama-server POST /v1/multidecode.
// WHY not CompletionRequest: custom RoPE pos + ancestor parent (or parent_node_ids)
// for shared-prefix fan-out — see docs/multidecode-llama-cpp.md.
type MultiDecodeRequest struct {
	Model          string  `json:"model,omitempty"`
	Tokens         []int32 `json:"tokens"`
	Pos            []int32 `json:"pos"`
	Parent         []int32 `json:"parent,omitempty"`          // batch indices; omit when ParentNodeIDs set
	NodeIDs        []int32 `json:"node_ids,omitempty"`        // optional client-assigned
	ParentNodeIDs  []int32 `json:"parent_node_ids,omitempty"` // multi-step sticky KV
	Leaves         []int32 `json:"leaves"`
	ReturnLogits   bool    `json:"return_logits,omitempty"`
	Clear          *bool   `json:"clear,omitempty"` // default: true when ParentNodeIDs empty
	NPredict       int     `json:"n_predict,omitempty"`
}

// MultiDecodeLeafResult is one leaf sample (and optional full logit row).
type MultiDecodeLeafResult struct {
	Leaf   int32     `json:"leaf"`
	Token  int32     `json:"token"`
	NodeID int32     `json:"node_id"`
	Logits []float32 `json:"logits,omitempty"`
}

// MultiDecodeResponse is POST /v1/multidecode.
type MultiDecodeResponse struct {
	Results []MultiDecodeLeafResult `json:"results"`
	NodeIDs []int32                 `json:"node_ids"`
}

// MultiDecoder runs a MultiDecode forest on a llama-server with --multidecode.
type MultiDecoder interface {
	MultiDecode(ctx context.Context, req MultiDecodeRequest) (MultiDecodeResponse, error)
}
