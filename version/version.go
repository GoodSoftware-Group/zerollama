package version

// Version is the default when the binary is built without -ldflags.
// Why not 0.0.1: registry.ollama.ai returns 412 for ollama/0.0.x pulls (MLX/safetensors
// models like gemma4:26b-mlx). Match MinOllamaVersion floor for safetensors.
// Release builds should set this via -ldflags in scripts/build_zerollama_mac.sh.
var Version string = "0.19.0"
