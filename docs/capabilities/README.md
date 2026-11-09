# Capabilities

This folder is the **front door for working feature surfaces**.

Use capability docs to answer:

- What can zerollama do today?
- Which API or CLI surface should I use?
- Which backend paths are actually supported right now?
- Where is the deeper operator, roadmap, or upstream detail?

## Status legend

- `shipped` — working surface meant for normal product use
- `beta` — working, but still evolving or backend-dependent
- `lab` — real surface or backend, but still experimental / operator-heavy

## Core API capabilities

- [Tool calling](tool-calling.mdx)
- [Structured outputs](structured-outputs.mdx)
- [Streaming](streaming.mdx)
- [Thinking](thinking.mdx)
- [Vision](vision.mdx)
- [Embeddings](embeddings.mdx)
- [Web search](web-search.mdx)

## Product feature capabilities

- [Typed decisions](decisions.md)
- [Entity extract](extract.md)
- [Video](video.md)
- [Image](image.md)

## How this layer fits

- `capabilities/` = what works now
- `runbooks/` = how to operate it
- `design/` = how it works and why
- `roadmaps/` = what is still open
- `research/` = lab-only or future-facing work
- `upstream/` = where we are borrowing ideas from
- `handoffs/` = implementation history for maintainers
- `CHANGELOG.md` = what landed
