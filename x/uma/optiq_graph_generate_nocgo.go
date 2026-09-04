//go:build darwin && uma && !umaoptiq

package uma

import "fmt"

func runOptiqGraphGenerateInProcess(prompt []int32, nGen int) ([]int32, error) {
	_ = prompt
	_ = nGen
	return nil, fmt.Errorf("optiq GRAPH generate dylib not linked (rebuild with -tags uma,umaoptiq)")
}
