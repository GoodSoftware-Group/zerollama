package mlxrunner

import (
	"encoding/json"
	"testing"
)

func TestCompletionRequestTokensJSON(t *testing.T) {
	raw := `{"prompt":"hi","tokens":[1,2,3],"options":{}}`
	var req CompletionRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Tokens) != 3 || req.Tokens[0] != 1 {
		t.Fatalf("tokens = %v", req.Tokens)
	}
}
