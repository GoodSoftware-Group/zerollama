package server

import (
	"os"
	"testing"
)

func TestResolveAliasOneHop(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/aliases.yaml"
	if err := os.WriteFile(path, []byte("aliases:\n  gpt-4: llama3.2:3b\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_ALIASES_CONFIG", path)
	aliasFileMu.Lock()
	aliasFileCache = nil
	aliasFileMu.Unlock()
	aliasExtraMu.Lock()
	aliasExtra = map[string]string{}
	aliasExtraMu.Unlock()

	served, aliased, err := resolveAlias("gpt-4")
	if err != nil || !aliased || served != "llama3.2:3b" {
		t.Fatalf("got %q aliased=%v err=%v", served, aliased, err)
	}
	served, aliased, err = resolveAlias("llama3.2:3b")
	if err != nil || aliased || served != "llama3.2:3b" {
		t.Fatalf("passthrough %q %v %v", served, aliased, err)
	}
}

func TestResolveAliasRejectsChain(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/aliases.yaml"
	if err := os.WriteFile(path, []byte("aliases:\n  a: b\n  b: c\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_ALIASES_CONFIG", path)
	aliasFileMu.Lock()
	aliasFileCache = nil
	aliasFileMu.Unlock()
	if _, _, err := resolveAlias("a"); err == nil {
		t.Fatal("expected chain error")
	}
}

func TestAliasOverlay(t *testing.T) {
	t.Setenv("ZEROLLAMA_ALIASES_CONFIG", "0")
	aliasFileMu.Lock()
	aliasFileCache = nil
	aliasFileMu.Unlock()
	aliasExtraMu.Lock()
	aliasExtra = map[string]string{}
	aliasExtraMu.Unlock()
	addAliasOverlay("gpt-4o", "llama3.2:1b")
	served, aliased, err := resolveAlias("gpt-4o")
	if err != nil || !aliased || served != "llama3.2:1b" {
		t.Fatalf("%q %v %v", served, aliased, err)
	}
}
