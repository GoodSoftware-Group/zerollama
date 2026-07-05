package openai

import (
	"encoding/json"
)

// BindImageGenerationRequest unmarshals an OpenAI image body and merges extra_body zerollama/options.
func BindImageGenerationRequest(body []byte) (ImageGenerationRequest, error) {
	var req ImageGenerationRequest
	if len(body) == 0 {
		return req, nil
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return req, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return req, err
	}
	if eb, ok := raw["extra_body"]; ok {
		mergeImageExtraBody(&req, eb)
	}
	return req, nil
}

// BindImageEditRequest unmarshals an OpenAI image edit body and merges extra_body zerollama/options.
func BindImageEditRequest(body []byte) (ImageEditRequest, error) {
	var req ImageEditRequest
	if len(body) == 0 {
		return req, nil
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return req, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return req, err
	}
	if eb, ok := raw["extra_body"]; ok {
		mergeImageEditExtraBody(&req, eb)
	}
	return req, nil
}

func mergeImageExtraBody(req *ImageGenerationRequest, extra json.RawMessage) {
	var flat map[string]json.RawMessage
	if err := json.Unmarshal(extra, &flat); err != nil {
		return
	}
	if optsRaw, ok := flat["options"]; ok {
		var opts map[string]any
		if json.Unmarshal(optsRaw, &opts) == nil {
			req.Options = mergeOptionsMaps(req.Options, opts)
		}
	}
	if zRaw, ok := flat["zerollama"]; ok {
		var z map[string]any
		if json.Unmarshal(zRaw, &z) == nil {
			req.Options = mergeZerollamaOptions(req.Options, z)
		}
	}
}

func mergeImageEditExtraBody(req *ImageEditRequest, extra json.RawMessage) {
	var flat map[string]json.RawMessage
	if err := json.Unmarshal(extra, &flat); err != nil {
		return
	}
	if optsRaw, ok := flat["options"]; ok {
		var opts map[string]any
		if json.Unmarshal(optsRaw, &opts) == nil {
			req.Options = mergeOptionsMaps(req.Options, opts)
		}
	}
	if zRaw, ok := flat["zerollama"]; ok {
		var z map[string]any
		if json.Unmarshal(zRaw, &z) == nil {
			req.Options = mergeZerollamaOptions(req.Options, z)
		}
	}
}
