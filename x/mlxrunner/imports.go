package mlxrunner

// Side-effect imports: each package's init() calls base.Register. A model
// that exists under x/models but is missing here loads as
// "unsupported architecture" (tiny-agent Qwen2 on 2026-08-24).
import (
	_ "github.com/ollama/ollama/x/models/cohere2_moe"
	_ "github.com/ollama/ollama/x/models/deepseekv4" // Flash CSA/HCA; without this import create/load is "unsupported architecture"
	_ "github.com/ollama/ollama/x/models/gemma3"
	_ "github.com/ollama/ollama/x/models/gemma4"
	_ "github.com/ollama/ollama/x/models/glm4_moe_lite"
	_ "github.com/ollama/ollama/x/models/granite"
	_ "github.com/ollama/ollama/x/models/lfm2"
	_ "github.com/ollama/ollama/x/models/llama"
	_ "github.com/ollama/ollama/x/models/qwen2"
	_ "github.com/ollama/ollama/x/models/qwen3"
	_ "github.com/ollama/ollama/x/models/qwen3_5"
	_ "github.com/ollama/ollama/x/models/qwen3_5_moe"
)
