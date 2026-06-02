package server

import (
	"encoding/json"
	"strings"
)

// runtimeGGUFPath resolves a local Ollama model name to its on-disk GGUF blob path.
func runtimeGGUFPath(modelName string) (string, bool) {
	modelRef, err := parseAndValidateModelRef(modelName)
	if err != nil {
		return "", false
	}
	if modelRef.Source == modelSourceCloud {
		return "", false
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		return "", false
	}
	m, err := GetModel(name.String())
	if err != nil || m.ModelPath == "" {
		return "", false
	}
	return m.ModelPath, true
}

// runtimeProxyOptions builds runtime request options: client opts (e.g. num_ctx),
// optional num_predict cap, and manifest GGUF path for the Python sidecar.
func runtimeProxyOptions(
	modelName string,
	nPredict int,
	limited bool,
	clientOpts map[string]any,
) map[string]any {
	opts := map[string]any{}
	for k, v := range clientOpts {
		opts[k] = v
	}
	if limited {
		opts["num_predict"] = nPredict
	}
	clientGGUF := ""
	if g, ok := opts["gguf"].(string); ok {
		clientGGUF = strings.TrimSpace(g)
	}
	if clientGGUF != "" {
		opts["gguf"] = clientGGUF
	} else if p, ok := runtimeGGUFPath(modelName); ok {
		opts["gguf"] = p
	}
	return opts
}

// v1NumPredictFromBody reads OpenAI max_tokens when set (maps to runtime num_predict cap).
func v1NumPredictFromBody(body map[string]any) (int, bool) {
	if body == nil {
		return 0, false
	}
	v, ok := body["max_tokens"]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		if n > 0 {
			return int(n), true
		}
	case int:
		if n > 0 {
			return n, true
		}
	case int64:
		if n > 0 {
			return int(n), true
		}
	}
	return 0, false
}

// runtimeV1ProxyOptions builds options for Python /v1/chat/completions (manifest GGUF + client opts).
// Why: Phase 13 resolve_num_ctx_for_request on the runtime reads options.gguf / options.num_ctx;
// OpenAI bodies omit Ollama options unless the Go proxy injects them (same as /api/chat).
func runtimeV1ProxyOptions(modelName string, body map[string]any) map[string]any {
	clientOpts := map[string]any{}
	if raw, ok := body["options"].(map[string]any); ok {
		for k, v := range raw {
			clientOpts[k] = v
		}
	}
	nPredict, limited := v1NumPredictFromBody(body)
	if !limited {
		if np, ok := numPredictFromOptions(clientOpts); ok {
			nPredict = np
			limited = true
		}
	}
	if body != nil {
		if nc, ok := body["num_ctx"]; ok {
			if _, has := clientOpts["num_ctx"]; !has {
				clientOpts["num_ctx"] = nc
			}
		}
	}
	return runtimeProxyOptions(modelName, nPredict, limited, clientOpts)
}

// runtimeV1ChatBodyWithOptions rewrites the v1 JSON body with merged runtime options.
func runtimeV1ChatBodyWithOptions(modelName string, body []byte) ([]byte, error) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	m["options"] = runtimeV1ProxyOptions(modelName, m)
	return json.Marshal(m)
}
