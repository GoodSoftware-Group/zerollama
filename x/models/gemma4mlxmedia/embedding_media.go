package gemma4mlxmedia

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

// MultimodalEmbedder projects pooled features into the text embedding
// space: a scale-less RMSNorm then a linear.
type MultimodalEmbedder struct {
	Projection nn.LinearLayer
}

// ClippableLinear clamps a linear's input and output to checkpoint-provided
// bounds; the loader guarantees all four are present.
type ClippableLinear struct {
	Inner                                    nn.LinearLayer
	InputMin, InputMax, OutputMin, OutputMax *mlx.Array
}

func (c *ClippableLinear) Forward(x *mlx.Array) *mlx.Array {
	x = mlx.Clip(x, c.InputMin, c.InputMax)
	x = c.Inner.Forward(x)
	return mlx.Clip(x, c.OutputMin, c.OutputMax)
}

func (c *ClippableLinear) OutputDim() int32 { return c.Inner.OutputDim() }

// makeClippableLinear builds one tower projection: the weight sits under a
// .linear suffix, with optional clamp scalars beside it.
func makeClippableLinear(linears model.LinearFactory, tensors map[string]*mlx.Array, name string) (nn.LinearLayer, error) {
	inner := linears.Make(name + ".linear")
	if inner == nil {
		inner = linears.Make(name)
	}
	if inner == nil {
		return nil, fmt.Errorf("missing weight: %s", name)
	}

	c := &ClippableLinear{
		Inner:     inner,
		InputMin:  tensors[name+".input_min"],
		InputMax:  tensors[name+".input_max"],
		OutputMin: tensors[name+".output_min"],
		OutputMax: tensors[name+".output_max"],
	}
	if c.InputMin == nil && c.InputMax == nil && c.OutputMin == nil && c.OutputMax == nil {
		return inner, nil
	}
	if c.InputMin == nil || c.InputMax == nil || c.OutputMin == nil || c.OutputMax == nil {
		return nil, fmt.Errorf("weight %s has a partial clamp set", name)
	}
	return c, nil
}
