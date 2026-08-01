# zerollama Agent Skills

29 `SKILL.md` packages describing how to use [zerollama](https://github.com/GoodSoftware-Group/zerollama) — generated from the server's own OpenAPI spec, CLI, and source, intended for distribution to any agent/tool that can consume the `SKILL.md` (frontmatter + markdown) format.

See [skills.json](skills.json) for the machine-readable manifest (name, description, version, category, tags, related_skills) used to generate this table.

## Building

This README and `skills.json` are generated — do not hand-edit them.

```bash
python3 .tools/build.py             # validate + regenerate README.md / skills.json
python3 .tools/build.py --strict     # same, but exit non-zero on any issue (CI)
python3 .tools/build.py --package    # also zip each skill into dist/<name>.zip
```

The validator checks: required frontmatter fields present, `name` matches its directory and is lowercase-hyphen-only (≤64 chars), `description` ≤1024 chars, body ≤500 lines, and every `related_skills` entry resolves to a real skill in this directory (catches stale/renamed cross-links before distribution).

## mlops

| Skill | Description |
|---|---|
| [`account-auth`](account-auth/SKILL.md) | Check zerollama/Eliza Cloud account identity and manage local signing keys via /api/me, /api/signout, and /api/user/keys. |
| [`agent-web-tools`](agent-web-tools/SKILL.md) | Use zerollama's experimental server-side web search and web fetch endpoints as agent tools. |
| [`anthropic-messages-compat`](anthropic-messages-compat/SKILL.md) | Talk to a zerollama server using the Anthropic Messages API wire format (/v1/messages) for tools built against Claude's API. |
| [`batch-inference`](batch-inference/SKILL.md) | Fan out multiple text chat completions to the same model in one call via a zerollama server's batch endpoint. |
| [`benchmark-model-speed`](benchmark-model-speed/SKILL.md) | Benchmark local model decode speed (chat) or generation wall time (image/video) on a zerollama server and cache results for zerollama ls. |
| [`cloud-model-routing`](cloud-model-routing/SKILL.md) | Route chat/embedding/message requests on a zerollama server to remote Eliza Cloud models using the :cloud model suffix, instead of local GGUF inference. |
| [`configure-zerollama-env`](configure-zerollama-env/SKILL.md) | Help a user navigate zerollama's ~1000 environment variables and YAML profiles (OLLAMA_*/ZEROLLAMA_*) — discover current effective config, prefer profiles over raw flags, and verify changes instead of guessing. |
| [`diagnose-server-health`](diagnose-server-health/SKILL.md) | Run zerollama doctor to diagnose local runtime readiness (venvs, libllama, MLX, sidecar health, model blob integrity) and apply safe auto-fixes. |
| [`distill-and-train`](distill-and-train/SKILL.md) | Fine-tune (LoRA/QLoRA) a local model on a zerollama server, including distilling a larger model into a smaller one via synthetic SFT data, plus the ternary QAT path for extreme quantization. |
| [`doctor-model`](doctor-model/SKILL.md) | Diagnose a specific local model's manifest/blob health (ok/repairable/orphaned/broken) and config-level footguns (quant label, missing generation config, context mismatch, no chat template) with zerollama doctor --models and the model-serving-minefield trap registry. |
| [`download-model`](download-model/SKILL.md) | Pull, register, list, and remove models on a zerollama server (GGUF chat models, image/video/audio backends). |
| [`fleet-management`](fleet-management/SKILL.md) | Run a zerollama fleet management node that routes agent requests to the best of several zerollama peers by warm-model status, via zerollama fleet serve. |
| [`fleet-vram-admission`](fleet-vram-admission/SKILL.md) | Inspect and dry-run zerollama's model admission/scheduling: capacity checks, co-residency planning, pins, and live fleet status before loading a model. |
| [`generate-embeddings`](generate-embeddings/SKILL.md) | Generate vector embeddings for text via a zerollama server, for RAG, semantic search, or clustering. |
| [`generate-image`](generate-image/SKILL.md) | Generate or edit images via a zerollama server (MLX fast path or ComfyUI backend). |
| [`generate-video`](generate-video/SKILL.md) | Generate text-to-video clips via a zerollama server's async OpenAI-compatible Videos API (Wan). |
| [`gpu-capability-discovery`](gpu-capability-discovery/SKILL.md) | Determine which GPU backend, autoconfig profile, and inference path a zerollama server picked (Metal/CUDA/MLX/inprocess llama-server) before choosing models or debugging performance. |
| [`hermes-provider`](hermes-provider/SKILL.md) | Wire Hermes to a zerollama server. |
| [`install-zerollama`](install-zerollama/SKILL.md) | Bootstrap and build zerollama from a fresh clone on macOS or Linux — prerequisites, tiered onboarding scripts, and daily-use verification. |
| [`launch-agent-integration`](launch-agent-integration/SKILL.md) | Wire up a coding agent CLI (Cline, OpenCode, Droid, Pi, Hermes, etc.) to a local zerollama server via zerollama launch, using one shared model inventory instead of per-integration config hacks. |
| [`lmstudio-cache-import`](lmstudio-cache-import/SKILL.md) | Import models already downloaded by LM Studio into zerollama without re-downloading, via zerollama list/pull cache discovery. |
| [`model-authoring`](model-authoring/SKILL.md) | Create custom model variants, quantize, copy, and repair manifests on a zerollama server (Modelfile-style /api/create, /api/copy, /api/repair). |
| [`model-suggester`](model-suggester/SKILL.md) | Pick the right zerollama model for a task: match required capabilities (tools/vision/embedding/thinking), context length, and available VRAM against local inventory, LM Studio cache, and cloud fallback before recommending a pull or a :cloud route. |
| [`openai-responses-compat`](openai-responses-compat/SKILL.md) | Talk to a zerollama server using OpenAI's Responses API wire format (/v1/responses) instead of Chat Completions. |
| [`rerank-candidates`](rerank-candidates/SKILL.md) | Score fixed candidate continuations against a shared prompt via a zerollama server, for classification, routing, or reranking without full generation. |
| [`speech-to-text`](speech-to-text/SKILL.md) | Transcribe audio to text via a zerollama server's OpenAI-compatible transcription API (Whisper or multimodal chat models). |
| [`text-to-speech`](text-to-speech/SKILL.md) | Synthesize speech audio from text via a zerollama server's OpenAI-compatible speech API (Piper, Chatterbox, Orpheus, Kokoro). |
| [`video-understanding-chat`](video-understanding-chat/SKILL.md) | Send video clips into a zerollama chat request for vision-language understanding (video_url/videos in /v1/chat/completions), distinct from generating video. |
| [`zerollama-integration`](zerollama-integration/SKILL.md) | Connect any agent harness to a zerollama server. |
