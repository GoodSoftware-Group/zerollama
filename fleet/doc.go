// Package fleet implements a thin management node for multi-zerollama deployments (F3).
//
// Why this exists: each zerollama process already owns local VRAM, queues, and load/evict
// decisions. Agents that see many hosts still need a cross-node view — "which box has model M
// warm?" — without scatter-gather probes or long-lived reservation quotes that waste GPU work
// when agents cancel. This package polls peer GET /api/status (F2), builds a warm-model map,
// and returns {url, node_id} for direct agent calls. It never loads models or evicts remotely.
//
// CLI: zerollama fleet serve — see docs/fleet-management.md.
package fleet
