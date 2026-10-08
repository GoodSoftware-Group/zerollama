package gemma4mlxmedia

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// RopeParams holds per-layer-type RoPE settings for vision towers.
type RopeParams struct {
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	RopeTheta           float32 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
}

// MLP is a GELU-tanh feed-forward block shared by vision/audio towers.
type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

// Forward runs the GELU-tanh MLP.
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := m.GateProj.Forward(x)
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.GeGLU(gate, up))
}
