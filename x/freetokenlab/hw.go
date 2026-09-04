package freetokenlab

// Paper Table 1 (arXiv:2608.16157) plus a Mac UMA inversion.
var (
	RTX5090Server  = Bandwidth{BP: 52.7, BH: 77.3}
	RTX4090        = Bandwidth{BP: 25.1, BH: 63.2}
	RTX3090        = Bandwidth{BP: 25.3, BH: 56.7}
	RTX5090Desktop = Bandwidth{BP: 49.0, BH: 53.8}
	RTX4060Laptop  = Bandwidth{BP: 11.8, BH: 47.5}
	RTXPro6000     = Bandwidth{BP: 51.5, BH: 178}
	// Unified memory: DMA and CPU experts share the same DRAM, so BP≈BH
	// and q* collapses to all fills (paper: BH→BP ⇒ q*→m).
	MacUMA = Bandwidth{BP: 90, BH: 90}
	// RTX 5080 is not in FreeToken Table 1. Until CT 1564 benches expert
	// DMA vs host GEMM, treat PCIe 5.0 fill like 5090-desktop and host
	// GEMM like 4090. Label stays *-est so it is not confused with paper.
	RTX5080Est = Bandwidth{BP: 49.0, BH: 63.2}
)

// Profiles used by the sim CLI and tests.
func Profiles() map[string]Bandwidth {
	return map[string]Bandwidth{
		"5090-server":  RTX5090Server,
		"4090":         RTX4090,
		"3090":         RTX3090,
		"5090-desktop": RTX5090Desktop,
		"4060-laptop":  RTX4060Laptop,
		"pro-6000":     RTXPro6000,
		"mac-uma":      MacUMA,
		"5080-est":     RTX5080Est,
	}
}
