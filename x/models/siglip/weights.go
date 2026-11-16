package siglip

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
	if m.Config.Vision != nil {
		tower, err := loadVision(linears, tensors, m.Config.Vision)
		if err != nil {
			return err
		}
		m.Vision = tower
		m.Config.Vision = tower.cfg
	}
	if m.Config.Text != nil {
		tower, err := loadText(linears, tensors, m.Config.Text)
		if err != nil {
			return err
		}
		m.Text = tower
	}
	return nil
}

func loadVision(linears model.LinearFactory, tensors map[string]*mlx.Array, cfg *VisionConfig) (*visionTower, error) {
	return loadVisionKeys(linears, tensors, cfg, nil)
}

func loadVisionKeys(linears model.LinearFactory, tensors map[string]*mlx.Array, cfg *VisionConfig, extra []string) (*visionTower, error) {
	names := append([]string{}, extra...)
	names = append(names, "vision_model.embeddings.patch_embedding.weight", "model.vision_model.embeddings.patch_embedding.weight")
	prefix, err := findPrefix(tensors, names...)
	if err != nil {
		return nil, fmt.Errorf("vision tower: %w", err)
	}
	weight := tensors[prefix+"embeddings.patch_embedding.weight"]
	tower := &visionTower{
		cfg:      cfg,
		act:      activation(cfg.HiddenAct),
		PostNorm: loadLayerNorm(tensors, prefix+"post_layernorm", cfg.LayerNormEps),
	}
	if tower.PostNorm == nil {
		return nil, fmt.Errorf("missing %spost_layernorm.weight", prefix)
	}
	switch weight.NumDims() {
	case 4:
		// Torch stores conv weights OIHW. MLX conv2d wants O, kH, kW, I.
		tower.PatchWeight = mlx.Transpose(weight, 0, 2, 3, 1)
		tower.PatchBias = tensors[prefix+"embeddings.patch_embedding.bias"]
		cfg.NaFlex = false
	case 2:
		var err error
		tower.PatchLinear, err = mustLinear(linears, prefix+"embeddings.patch_embedding")
		if err != nil {
			return nil, err
		}
		cfg.NaFlex = true
		cfg.ImageSize = 0
	default:
		return nil, fmt.Errorf("patch embedding rank %d is neither conv nor linear", weight.NumDims())
	}
	pos := tensors[prefix+"embeddings.position_embedding.weight"]
	if pos == nil {
		return nil, fmt.Errorf("missing %sembeddings.position_embedding.weight", prefix)
	}
	if pos.Dim(0) != int(cfg.NumPatches) {
		cfg.NumPatches = int32(pos.Dim(0))
	}
	tower.Positions = nn.NewEmbedding(pos)
	tower.Blocks = make([]*encoderBlock, cfg.NumHiddenLayers)
	for i := range cfg.NumHiddenLayers {
		block, err := loadBlock(linears, tensors, fmt.Sprintf("%sencoder.layers.%d.", prefix, i), cfg.LayerNormEps)
		if err != nil {
			return nil, fmt.Errorf("vision layer %d: %w", i, err)
		}
		tower.Blocks[i] = block
	}
	if cfg.UseHead {
		head, err := loadMapHead(linears, tensors, prefix+"head.", cfg)
		if err != nil {
			return nil, err
		}
		tower.Head = head
	}
	if proj, err := maybeLinear(linears, "visual_projection", "model.visual_projection"); err != nil {
		return nil, err
	} else {
		tower.Projection = proj
	}
	return tower, nil
}

func loadText(linears model.LinearFactory, tensors map[string]*mlx.Array, cfg *TextConfig) (*textTower, error) {
	prefix, err := findPrefix(tensors, "text_model.embeddings.token_embedding.weight", "model.text_model.embeddings.token_embedding.weight")
	if err != nil {
		return nil, fmt.Errorf("text tower: %w", err)
	}
	tokens := model.MakeEmbeddingLayer(tensors, prefix+"embeddings.token_embedding", 0, 0, "", nil)
	if tokens == nil {
		return nil, fmt.Errorf("missing %sembeddings.token_embedding.weight", prefix)
	}
	pos := tensors[prefix+"embeddings.position_embedding.weight"]
	if pos == nil {
		return nil, fmt.Errorf("missing %sembeddings.position_embedding.weight", prefix)
	}
	if pos.Dim(0) < int(cfg.MaxPositionEmbeddings) {
		cfg.MaxPositionEmbeddings = int32(pos.Dim(0))
	}
	tower := &textTower{
		Tokens:    tokens,
		Positions: nn.NewEmbedding(pos),
		cfg:       cfg,
		act:       activation(cfg.HiddenAct),
		FinalNorm: loadLayerNorm(tensors, prefix+"final_layer_norm", cfg.LayerNormEps),
	}
	if tower.FinalNorm == nil {
		return nil, fmt.Errorf("missing %sfinal_layer_norm.weight", prefix)
	}
	tower.Blocks = make([]*encoderBlock, cfg.NumHiddenLayers)
	for i := range cfg.NumHiddenLayers {
		block, err := loadBlock(linears, tensors, fmt.Sprintf("%sencoder.layers.%d.", prefix, i), cfg.LayerNormEps)
		if err != nil {
			return nil, fmt.Errorf("text layer %d: %w", i, err)
		}
		tower.Blocks[i] = block
	}
	head, err := maybeLinear(linears, prefix+"head", "text_projection", "model.text_projection")
	if err != nil {
		return nil, err
	}
	tower.Head = head
	return tower, nil
}

func loadBlock(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, eps float32) (*encoderBlock, error) {
	block := &encoderBlock{Norm1: loadLayerNorm(tensors, prefix+"layer_norm1", eps), Norm2: loadLayerNorm(tensors, prefix+"layer_norm2", eps)}
	if block.Norm1 == nil || block.Norm2 == nil {
		return nil, fmt.Errorf("missing layer norm under %s", prefix)
	}
	var err error
	if block.Q, err = mustLinear(linears, prefix+"self_attn.q_proj"); err != nil {
		return nil, err
	}
	if block.K, err = mustLinear(linears, prefix+"self_attn.k_proj"); err != nil {
		return nil, err
	}
	if block.V, err = mustLinear(linears, prefix+"self_attn.v_proj"); err != nil {
		return nil, err
	}
	if block.Out, err = mustLinear(linears, prefix+"self_attn.out_proj"); err != nil {
		return nil, err
	}
	if block.FC1, err = mustLinear(linears, prefix+"mlp.fc1"); err != nil {
		return nil, err
	}
	if block.FC2, err = mustLinear(linears, prefix+"mlp.fc2"); err != nil {
		return nil, err
	}
	return block, nil
}

func loadMapHead(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, cfg *VisionConfig) (*mapHead, error) {
	probe := tensors[prefix+"probe"]
	if probe == nil {
		return nil, fmt.Errorf("missing %sprobe", prefix)
	}
	head := &mapHead{
		Probe: probe,
		Norm:  loadLayerNorm(tensors, prefix+"layernorm", cfg.LayerNormEps),
		heads: cfg.NumHeads,
		act:   activation(cfg.HiddenAct),
	}
	if head.Norm == nil {
		return nil, fmt.Errorf("missing %slayernorm.weight", prefix)
	}
	var err error
	if head.FC1, err = mustLinear(linears, prefix+"mlp.fc1"); err != nil {
		return nil, err
	}
	if head.FC2, err = mustLinear(linears, prefix+"mlp.fc2"); err != nil {
		return nil, err
	}
	if head.Out, err = mustLinear(linears, prefix+"attention.out_proj"); err != nil {
		return nil, err
	}
	if packed := tensors[prefix+"attention.in_proj_weight"]; packed != nil {
		head.InWeight = packed
		head.InBias = tensors[prefix+"attention.in_proj_bias"]
		return head, nil
	}
	if head.Q, err = mustLinear(linears, prefix+"attention.q_proj"); err != nil {
		return nil, err
	}
	if head.K, err = mustLinear(linears, prefix+"attention.k_proj"); err != nil {
		return nil, err
	}
	if head.V, err = mustLinear(linears, prefix+"attention.v_proj"); err != nil {
		return nil, err
	}
	return head, nil
}

func loadLayerNorm(tensors map[string]*mlx.Array, name string, eps float32) *nn.LayerNorm {
	weight := tensors[name+".weight"]
	if weight == nil {
		return nil
	}
	return &nn.LayerNorm{Weight: weight, Bias: tensors[name+".bias"], Eps: eps}
}

func findPrefix(tensors map[string]*mlx.Array, names ...string) (string, error) {
	for _, name := range names {
		if tensors[name] == nil {
			continue
		}
		for _, marker := range []string{"vision_model.", "text_model.", "vision_tower."} {
			if i := indexOf(name, marker); i >= 0 {
				return name[:i+len(marker)], nil
			}
		}
	}
	return "", fmt.Errorf("missing %s", names[0])
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func mustLinear(linears model.LinearFactory, path string) (nn.LinearLayer, error) {
	layer := linears.Make(path)
	if layer == nil {
		return nil, fmt.Errorf("missing %s.weight", path)
	}
	return layer, nil
}

func maybeLinear(linears model.LinearFactory, paths ...string) (nn.LinearLayer, error) {
	for _, path := range paths {
		if layer := linears.Make(path); layer != nil {
			return layer, nil
		}
	}
	return nil, nil
}
