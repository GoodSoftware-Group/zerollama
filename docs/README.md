# Documentation

### Getting Started
* [Quickstart](https://docs.ollama.com/quickstart)
* [Examples](./examples.md)
* [Importing models](https://docs.ollama.com/import)
* [MacOS Documentation](https://docs.ollama.com/macos)
* [Linux Documentation](https://docs.ollama.com/linux)
* [Windows Documentation](https://docs.ollama.com/windows)
* [Docker Documentation](https://docs.ollama.com/docker)

### Reference

* [API Reference](https://docs.ollama.com/api)
* [Modelfile Reference](https://docs.ollama.com/modelfile)
* [OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
* [Anthropic Compatibility](./api/anthropic-compatibility.mdx)

### Multimodal & video (repo)

These live in-repo (not only on docs.ollama.com) because they explain **design rationale**—API shape, limits, and optional backends:

* [Video understanding (VLM)](./video-understanding.md) — **why** `video_url` / `videos` → ffmpeg → vision pipeline; **why** preflight and `video_spans` exist.
* [Optional multimodal backends](./multimodal-backends.md) — env + manifest; **why** both layers.
* [Video parity matrix](./video-parity.md) — **why** reference workloads for native vs SGLang.
* [Roadmap](./ROADMAP.md) — **why** Option 2 is phased (policy, templates, context, optional subprocess).

### Remote inference — Eliza Cloud (Zerollama)

* [Eliza Cloud](./eliza-cloud.md) — **why** default upstream is Eliza (not legacy ollama.com), **why** path rewrites and `X-API-Key`, **why** catalog merge + cache, **why** raw JSON on some routes, **why** account stubs off ollama.com.

### Resources

* [Troubleshooting Guide](https://docs.ollama.com/troubleshooting)
* [FAQ](https://docs.ollama.com/faq#faq)
* [Development guide](./development.md)
