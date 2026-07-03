package cmd

// bench_media.go — image and video_gen paths for zerollama bench.
//
// WHY separate from bench_cmd.go: chat bench measures EvalCount/EvalDuration decode tok/s;
// image/video measure wall seconds (subprocess diffusion or async /v1/videos jobs). Different
// timeouts, epoch caps, and cache fields (gen_sec vs tok_per_sec) — keeping media logic here
// avoids bloating the completion loop.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/openai"
)

const benchImagePrompt = "a red apple on a white table, product photo"

func benchImageOnce(ctx context.Context, client *api.Client, modelName string, warmup bool, loadTimeout, genTimeout time.Duration) (time.Duration, error) {
	stream := false
	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: benchImagePrompt,
		Stream: &stream,
	}
	timeout := genTimeout
	if warmup {
		timeout = loadTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var last api.GenerateResponse
	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if resp.Done {
			last = resp
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	if last.Image == "" {
		return 0, fmt.Errorf("no image in response")
	}
	if last.Metrics.TotalDuration > 0 {
		return last.Metrics.TotalDuration, nil
	}
	return 0, fmt.Errorf("no total_duration in response")
}

func benchImageModel(ctx context.Context, client *api.Client, m api.ListModelResponse, warmup, epochs int, loadTimeout, genTimeout time.Duration, minEpochs int) (benchModelResult, error) {
	for i := range warmup {
		if _, err := benchImageOnce(ctx, client, m.Name, true, loadTimeout, genTimeout); err != nil {
			fmt.Fprintf(os.Stderr, "warning: image warmup %d/%d for %s failed: %v\n", i+1, warmup, m.Name, err)
		}
	}
	timedEpochs := epochs
	if timedEpochs > 2 {
		timedEpochs = 2 // WHY cap: image runs are minutes-class on low-end GPUs
	}
	var secs []float64
	for epoch := range timedEpochs {
		d, err := benchImageOnce(ctx, client, m.Name, false, loadTimeout, genTimeout)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: image epoch %d/%d: %v\n", m.Name, epoch+1, timedEpochs, err)
			continue
		}
		if d > 0 {
			secs = append(secs, d.Seconds())
		}
	}
	// WHY clamp: minEpochs from caller may exceed timedEpochs cap (2); clamp so
	// "--min-epochs 3 --epochs 3" doesn't always fail for image models.
	effectiveMin := minEpochs
	if effectiveMin > timedEpochs {
		effectiveMin = timedEpochs
	}
	if len(secs) < effectiveMin {
		return benchModelResult{kind: benchKindImage, epochsTotal: timedEpochs}, fmt.Errorf("only %d/%d image epochs succeeded (need %d)", len(secs), timedEpochs, effectiveMin)
	}
	var sum float64
	for _, s := range secs {
		sum += s
	}
	return benchModelResult{
		kind:        benchKindImage,
		genSec:      sum / float64(len(secs)),
		epochsOK:    len(secs),
		epochsTotal: timedEpochs,
		partial:     len(secs) < timedEpochs,
	}, nil
}

func benchVideoOnce(ctx context.Context, modelName string, timeout time.Duration) (time.Duration, error) {
	host := envconfig.ConnectableHost().String()
	start := time.Now()

	body, err := json.Marshal(openai.VideoCreateRequest{
		Model:  modelName,
		Prompt: benchImagePrompt,
	})
	if err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, host+"/v1/videos", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		b, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("video create: %s", strings.TrimSpace(string(b)))
	}
	var created openai.Video
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		return 0, err
	}
	if created.ID == "" {
		return 0, fmt.Errorf("video create returned empty id")
	}

	deadline := start.Add(timeout)
	for time.Now().Before(deadline) {
		if ctx.Err() != nil {
			return 0, ctx.Err()
		}
		stReq, err := http.NewRequestWithContext(ctx, http.MethodGet, host+"/v1/videos/"+created.ID, nil)
		if err != nil {
			return 0, err
		}
		stResp, err := http.DefaultClient.Do(stReq)
		if err != nil {
			return 0, err
		}
		var v openai.Video
		decodeErr := json.NewDecoder(stResp.Body).Decode(&v)
		closeErr := stResp.Body.Close()
		if decodeErr != nil {
			return 0, decodeErr
		}
		if closeErr != nil {
			return 0, closeErr
		}
		switch v.Status {
		case "completed":
			return time.Since(start), nil
		case "failed", "cancelled":
			msg := "video generation failed"
			if v.Error != nil && v.Error.Message != "" {
				msg = v.Error.Message
			}
			return 0, fmt.Errorf("%s", msg)
		default:
			time.Sleep(2 * time.Second)
		}
	}
	return 0, fmt.Errorf("video generation timed out after %v", timeout)
}

func benchVideoModel(ctx context.Context, m api.ListModelResponse, videoTimeout time.Duration) (benchModelResult, error) {
	ctx, cancel := context.WithTimeout(ctx, videoTimeout)
	defer cancel()
	d, err := benchVideoOnce(ctx, m.Name, videoTimeout)
	if err != nil {
		return benchModelResult{kind: benchKindVideoGen, epochsTotal: 1}, err
	}
	return benchModelResult{
		kind:        benchKindVideoGen,
		genSec:      d.Seconds(),
		epochsOK:    1,
		epochsTotal: 1,
	}, nil
}
