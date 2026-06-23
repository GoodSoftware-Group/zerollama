package fleet

import (
	"testing"
	"time"
)

func TestParsePeers(t *testing.T) {
	got, err := ParsePeers("http://127.0.0.1:11434, 192.168.1.2:11435")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("len=%d", len(got))
	}
	if got[0] != "http://127.0.0.1:11434" {
		t.Fatalf("peer0=%q", got[0])
	}
	if got[1] != "http://192.168.1.2:11435" {
		t.Fatalf("peer1=%q", got[1])
	}
}

func TestAssignWarmLowestQueue(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, LoadedModels: []string{"llama3:latest"}, QueueDepth: 2},
		{ID: "b", URL: "http://b:11434", Available: true, LoadedModels: []string{"llama3:latest"}, QueueDepth: 0},
		{ID: "c", URL: "http://c:11434", Available: true, QueueDepth: 0},
	}
	resp, err := Assign(nodes, AssignRequest{Model: "llama3:latest"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.NodeID != "b" || !resp.Warm {
		t.Fatalf("got %+v", resp)
	}
}

func TestAssignColdWhenNoWarm(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, QueueDepth: 3},
		{ID: "b", URL: "http://b:11434", Available: true, QueueDepth: 1},
	}
	resp, err := Assign(nodes, AssignRequest{Model: "qwen2.5:14b"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.NodeID != "b" || resp.Warm {
		t.Fatalf("got %+v", resp)
	}
}

func TestAssignWarmOnlyRejectsCold(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, QueueDepth: 0},
	}
	_, err := Assign(nodes, AssignRequest{Model: "llama3", WarmOnly: true}, nil)
	if err != ErrNoWarmNode {
		t.Fatalf("err=%v", err)
	}
}

func TestAssignExcludeNode(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, LoadedModels: []string{"llama3"}, QueueDepth: 0},
		{ID: "b", URL: "http://b:11434", Available: true, LoadedModels: []string{"llama3"}, QueueDepth: 1},
	}
	resp, err := Assign(nodes, AssignRequest{Model: "llama3", Exclude: []string{"a"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.NodeID != "b" {
		t.Fatalf("got %+v", resp)
	}
}

func TestModelMatchesBaseName(t *testing.T) {
	if !modelMatches("llama3:latest", "llama3") {
		t.Fatal("expected base match")
	}
	if modelMatches("llama3:latest", "llama4") {
		t.Fatal("unexpected match")
	}
}

func TestAssignEmptyNodes(t *testing.T) {
	_, err := Assign(nil, AssignRequest{Model: "llama3"}, nil)
	if err != ErrNoNodes {
		t.Fatalf("err=%v", err)
	}
}

func TestAssignPreferWarmFalseStillReportsWarm(t *testing.T) {
	preferWarm := false
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, LoadedModels: []string{"llama3:latest"}, QueueDepth: 0},
	}
	resp, err := Assign(nodes, AssignRequest{Model: "llama3", PreferWarm: &preferWarm}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Warm {
		t.Fatalf("expected warm=true when assigned node has model loaded, got %+v", resp)
	}
}

func TestBuildWarmMap(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", Available: true, URL: "http://a:11434", LoadedModels: []string{"llama3:latest"}, QueueDepth: 1},
		{ID: "b", Available: true, URL: "http://b:11434", LoadedModels: []string{"llama3:latest"}, QueueDepth: 0},
		{ID: "c", Available: false, URL: "http://c:11434", LoadedModels: []string{"llama3:latest"}},
	}
	warm := BuildWarmMap(nodes)
	if len(warm) != 1 || len(warm[0].Nodes) != 2 {
		t.Fatalf("warm=%+v", warm)
	}
	if warm[0].Nodes[0].NodeID != "b" {
		t.Fatalf("expected lowest queue first, got %+v", warm[0].Nodes)
	}
}

func TestBuildWarmMapPreservesModelName(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", Available: true, URL: "http://a:11434", LoadedModels: []string{"Llama3:Latest"}},
	}
	warm := BuildWarmMap(nodes)
	if len(warm) != 1 || warm[0].Model != "Llama3:Latest" {
		t.Fatalf("warm=%+v", warm)
	}
}

func TestAssignPrefixAffinity(t *testing.T) {
	cache := NewPrefixCache(time.Hour)
	cache.Remember("llama3", "agent-thread-1", "b")

	nodes := []NodeSnapshot{
		{ID: "a", Available: true, URL: "http://a:11434", LoadedModels: []string{"llama3"}, QueueDepth: 0},
		{ID: "b", Available: true, URL: "http://b:11434", LoadedModels: []string{"llama3"}, QueueDepth: 3},
	}
	resp, err := Assign(nodes, AssignRequest{Model: "llama3", SessionKey: "agent-thread-1"}, cache)
	if err != nil {
		t.Fatal(err)
	}
	if resp.NodeID != "b" {
		t.Fatalf("expected affinity node b, got %+v", resp)
	}
	if !resp.Warm {
		t.Fatalf("expected warm assign, got %+v", resp)
	}
}
