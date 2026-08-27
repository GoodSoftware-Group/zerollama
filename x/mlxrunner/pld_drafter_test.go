package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPLDProposeEcho(t *testing.T) {
	skipIfNoMLX(t)
	// 3-gram key (shipping default): trailing [1,2,3] matches the earlier
	// site; draft is the tokens that followed ([4,5,6]).
	prompt := []int32{0, 1, 2, 3, 4, 5, 6, 9, 1, 2, 3}
	d := newPLDSession(prompt)
	c := d.propose(mlx.FromValues([]int32{3}, 1), 5)
	if c == nil {
		t.Fatal("expected PLD draft")
	}
	got := c.tokens.Ints()
	want := []int{4, 5, 6}
	if len(got) != len(want) {
		t.Fatalf("draft %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("draft %v want %v", got, want)
		}
	}
}
