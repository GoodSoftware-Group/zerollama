package fleet

import (
	"testing"
	"time"
)

func TestMintParseAssignTokenRoundTrip(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_SECRET", "test-secret")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TOKEN", "1")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TTL", "5s")
	now := time.Unix(1_700_000_000, 0).UTC()
	tok, exp, jti, err := MintAssignToken("node-a", "llama3", now)
	if err != nil {
		t.Fatal(err)
	}
	if jti == "" || tok == "" {
		t.Fatal("empty token")
	}
	if !exp.Equal(now.Add(5 * time.Second)) {
		t.Fatalf("exp=%v", exp)
	}
	claims, err := ParseAssignToken(tok, now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if claims.NodeID != "node-a" || claims.Model != "llama3" || claims.JTI != jti {
		t.Fatalf("claims=%+v", claims)
	}
	_, err = ParseAssignToken(tok, now.Add(6*time.Second))
	if err != ErrAssignTokenExpired {
		t.Fatalf("want expired, got %v", err)
	}
	_, err = ParseAssignToken(tok+"x", now.Add(time.Second))
	if err != ErrAssignTokenInvalid {
		t.Fatalf("want invalid, got %v", err)
	}
}

func TestAssignIncludesTokenWhenSecretSet(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_SECRET", "sec")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TOKEN", "1")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_PUSH", "0")
	nodes := []NodeSnapshot{
		{ID: "b", URL: "http://b:11434", Available: true, LoadedModels: []string{"llama3"}, QueueDepth: 0},
	}
	resp, err := Assign(nodes, AssignRequest{Model: "llama3"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.AssignmentToken == "" || resp.ExpiresAt == nil || resp.ExpiresIn <= 0 {
		t.Fatalf("resp=%+v", resp)
	}
}

func TestAssignOmitsTokenWithoutSecret(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_SECRET", "")
	t.Setenv("ZEROLLAMA_FLEET_ASSIGN_TOKEN", "")
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, LoadedModels: []string{"llama3"}, QueueDepth: 0},
	}
	resp, err := Assign(nodes, AssignRequest{Model: "llama3"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.AssignmentToken != "" {
		t.Fatalf("unexpected token %q", resp.AssignmentToken)
	}
}
