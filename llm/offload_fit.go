package llm

// OffloadFits is the requireFull rule for a llama-server load beside a
// resident model. need == 0 means the estimate is missing, and the load is
// allowed. Otherwise free device memory must cover the estimate.
func OffloadFits(need, free uint64) bool {
	if need == 0 {
		return true
	}
	return free >= need
}
