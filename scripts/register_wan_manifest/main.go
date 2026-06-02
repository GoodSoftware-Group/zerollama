// Command register_wan_manifest writes a config-only Ollama manifest for Wan video presets.
//
// Usage: go run ./scripts/register_wan_manifest <model-name> <config.json>
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "usage: %s <model-name> <config.json>\n", os.Args[0])
		os.Exit(2)
	}
	name := model.ParseName(os.Args[1])
	if !name.IsValid() {
		fmt.Fprintf(os.Stderr, "invalid model name %q\n", os.Args[1])
		os.Exit(1)
	}
	raw, err := os.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read config: %v\n", err)
		os.Exit(1)
	}
	var cfg model.ConfigV2
	if err := json.Unmarshal(raw, &cfg); err != nil {
		fmt.Fprintf(os.Stderr, "parse config: %v\n", err)
		os.Exit(1)
	}
	if cfg.OS == "" {
		cfg.OS = "linux"
	}
	if cfg.Architecture == "" {
		cfg.Architecture = "amd64"
	}
	if cfg.RootFS.Type == "" {
		cfg.RootFS.Type = "layers"
	}
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(&cfg); err != nil {
		fmt.Fprintf(os.Stderr, "encode config: %v\n", err)
		os.Exit(1)
	}
	layer, err := manifest.NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "config layer: %v\n", err)
		os.Exit(1)
	}
	if err := manifest.WriteManifest(name, layer, nil); err != nil {
		fmt.Fprintf(os.Stderr, "write manifest: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("registered %s\n", name)
}
