package envconfig

// ANERepo returns the maderix/ane checkout used to build libane_bridge.
// Why envconfig, not discover-only: surfaced in zerollama envconfig alongside FLASH_MOE_REPO.
// See docs/ane-probe.md.
func ANERepo() string {
	return localRepoPath("ANE_REPO", "ane")
}
