//go:build ignore

package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

func scan(path string) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(path, "err", err)
		return
	}
	defer f.Close()
	fi, _ := f.Stat()
	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		fmt.Println(path, "decode err", err)
		return
	}
	bc := int(meta.KV().BlockCount())
	layers := map[int]int{}
	post := 0
	for _, t := range meta.Tensors().Items() {
		if !strings.HasPrefix(t.Name, "blk.") {
			continue
		}
		parts := strings.Split(t.Name, ".")
		if len(parts) < 2 {
			continue
		}
		i, _ := strconv.Atoi(parts[1])
		layers[i]++
		if parts[2] == "post_attention_norm" || parts[2] == "ffn_norm" {
			post++
		}
	}
	max := -1
	for i := range layers {
		if i > max {
			max = i
		}
	}
	fmt.Printf("\n%s\n", path)
	fmt.Printf("  size GB: %.2f\n", float64(fi.Size())/1e9)
	fmt.Printf("  block_count meta: %d\n", bc)
	fmt.Printf("  tensor count: %d\n", len(meta.Tensors().Items()))
	fmt.Printf("  highest blk index: %d (layers with tensors: %d)\n", max, len(layers))
	fmt.Printf("  post_attention_norm/ffn_norm tensors: %d\n", post)
}

func main() {
	base := "/Users/user1/.lmstudio/models/lmstudio-community/gpt-oss-120b-GGUF"
	scan(base + "/gpt-oss-120b-MXFP4-00001-of-00002.gguf")
	scan(base + "/gpt-oss-120b-MXFP4-00002-of-00002.gguf")
	scan("/Users/user1/.ollama/models/blobs/sha256-01d8a3bc7efdaa331112e8a0a42ec9046fcee2a1dce910452aafaccb996759f3")
}
