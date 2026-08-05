package catalog

import "testing"

func TestRoleFromGGUFName(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		role   ModuleRole
		layer  int
		expert int
	}{
		{"token_embd.weight", RoleEmbed, -1, -1},
		{"output.weight", RoleLMHead, -1, -1},
		{"output_norm.weight", RoleNorm, -1, -1},
		{"blk.0.attn_q.weight", RoleAttn, 0, -1},
		{"blk.12.ffn_down.weight", RoleFFN, 12, -1},
		{"blk.3.ffn_gate_exps.5.weight", RoleExpert, 3, 5},
	}
	for _, tc := range cases {
		role, layer, expert := RoleFromGGUFName(tc.name)
		if role != tc.role || layer != tc.layer || expert != tc.expert {
			t.Errorf("%s: got %v %d %d want %v %d %d", tc.name, role, layer, expert, tc.role, tc.layer, tc.expert)
		}
	}
}

func TestRoleString(t *testing.T) {
	t.Parallel()
	if g := RoleString(RoleExpert, 3, 7); g != "layer.3.expert.7" {
		t.Fatalf("got %q", g)
	}
	if g := RoleString(RoleAttn, 0, -1); g != "layer.0.attn" {
		t.Fatalf("got %q", g)
	}
}
