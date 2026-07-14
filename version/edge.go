package version

// EdgeBuild is set via -ldflags when building with ./scripts/build/build_zerollama_edge.sh (-tags edge).
// Phase 16 v0: compile-time marker only; full ggml runner exclusion is future work.
var EdgeBuild = "false"

// IsEdgeBuild reports whether the binary was built with the edge compile marker.
func IsEdgeBuild() bool {
	return EdgeBuild == "true" || EdgeBuild == "1"
}
