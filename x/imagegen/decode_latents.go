package imagegen

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/imagegen/latents"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/zimage"
)

// ExecuteDecodeLatents runs VAE decode in a fresh process (subprocess of the imagegen runner).
func ExecuteDecodeLatents(args []string) error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: envconfig.LogLevel()})))

	fs := flag.NewFlagSet("imagegen-decode-latents", flag.ExitOnError)
	modelName := fs.String("model", "", "model name")
	latentsFile := fs.String("latents-file", "", "path to latent binary from denoise")
	denoiseMeta := fs.String("denoise-meta", "", "path to final-step handoff JSON (sample+noise)")
	output := fs.String("output", "", "path to write decoded float32 image tensor")
	width := fs.Int("width", 0, "image width")
	height := fs.Int("height", 0, "image height")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelName == "" || *output == "" {
		return fmt.Errorf("--model and --output are required")
	}
	if *denoiseMeta == "" && *latentsFile == "" {
		return fmt.Errorf("--denoise-meta or --latents-file is required")
	}
	if *denoiseMeta == "" && (*width <= 0 || *height <= 0) {
		return fmt.Errorf("--width and --height are required with --latents-file")
	}

	_ = os.Setenv("MLX_USE_CUDA_GRAPHS", "false")
	_ = os.Setenv("MLX_DISABLE_COMPILE", "1")

	// Model load/decode logs must not pollute stdout; parent reads {"ok":true} from stdout.
	stdout := os.Stdout
	os.Stdout = os.Stderr
	defer func() { os.Stdout = stdout }()

	if err := mlx.InitMLX(); err != nil {
		return fmt.Errorf("mlx init: %w", err)
	}

	m := &zimage.Model{}
	var img *mlx.Array
	var err error
	if *denoiseMeta != "" {
		meta, metaErr := latents.LoadFinalStepMeta(*denoiseMeta)
		if metaErr != nil {
			return metaErr
		}
		img, err = m.DecodeFromHandoff(*modelName, meta)
	} else {
		img, err = m.DecodeLatentsFromFile(*modelName, *latentsFile, int32(*width), int32(*height))
	}
	if err != nil {
		return err
	}
	defer img.Free()

	data := mlx.HostFloat32Slice(img)
	if len(data) == 0 {
		return fmt.Errorf("decoded image export returned no data")
	}
	if err := latents.SaveBin(*output, img.Shape(), data); err != nil {
		return fmt.Errorf("write decoded image: %w", err)
	}

	os.Stdout = stdout
	out := struct {
		OK bool `json:"ok"`
	}{OK: true}
	return json.NewEncoder(os.Stdout).Encode(out)
}
