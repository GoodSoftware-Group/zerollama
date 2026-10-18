package cmd

import "testing"

func TestStorageRunResolveArgs(t *testing.T) {
	base, model, prompts, err := storageRunResolveArgs([]string{"http://192.168.10.39:18090", "qwen3.8:latest"})
	if err != nil {
		t.Fatal(err)
	}
	if base != "http://192.168.10.39:18090" || model != "qwen3.8:latest" || len(prompts) != 0 {
		t.Fatalf("got base=%q model=%q prompts=%v", base, model, prompts)
	}
	base, model, prompts, err = storageRunResolveArgs([]string{"http://s:18090", "m:tag", "hi"})
	if err != nil || model != "m:tag" || len(prompts) != 1 || prompts[0] != "hi" {
		t.Fatalf("prompt args: base=%q model=%q prompts=%v err=%v", base, model, prompts, err)
	}
	_, _, _, err = storageRunResolveArgs([]string{"http://s:18090"})
	if err == nil {
		t.Fatal("url alone should require model")
	}
}
