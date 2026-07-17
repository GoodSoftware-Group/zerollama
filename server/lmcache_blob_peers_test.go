package server

import "testing"

func TestLmcacheBlobPeersForCoordination(t *testing.T) {
	t.Setenv("ZEROLLAMA_LMCACHE_BLOB_PEERS", "")
	t.Setenv("ZEROLLAMA_FLEET_PEERS", "http://a:11434, http://b:11434/")
	got := lmcacheBlobPeersForCoordination()
	if len(got) != 2 || got[0] != "http://a:11434" || got[1] != "http://b:11434" {
		t.Fatalf("got %#v", got)
	}
	t.Setenv("ZEROLLAMA_LMCACHE_BLOB_PEERS", "http://explicit:11434")
	got = lmcacheBlobPeersForCoordination()
	if len(got) != 1 || got[0] != "http://explicit:11434" {
		t.Fatalf("explicit=%#v", got)
	}
}
