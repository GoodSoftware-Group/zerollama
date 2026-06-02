// Package videogen implements zerollama run for video_gen models.
//
// Why poll HTTP instead of embedding Wan: the CLI stays thin—one code path with agents/SDKs
// that also use POST /v1/videos → GET status → GET content.
package videogen

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/openai"
)

// RunCLI submits a video job and polls until completion, saving the MP4 locally.
func RunCLI(cmd *cobra.Command, modelName, prompt string, keepAlive *api.Duration) error {
	_ = keepAlive
	host := envconfig.ConnectableHost()
	url := host.String()

	body, err := json.Marshal(openai.VideoCreateRequest{
		Model:  modelName,
		Prompt: prompt,
	})
	if err != nil {
		return err
	}
	resp, err := http.Post(url+"/v1/videos", "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("video create failed: %s", strings.TrimSpace(string(b)))
	}
	var created openai.Video
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		return err
	}
	fmt.Fprintf(cmd.OutOrStdout(), "video job %s queued\n", created.ID)

	deadline := time.Now().Add(3 * time.Hour)
	for time.Now().Before(deadline) {
		st, err := http.Get(url + "/v1/videos/" + created.ID)
		if err != nil {
			return err
		}
		var v openai.Video
		decodeErr := json.NewDecoder(st.Body).Decode(&v)
		closeErr := st.Body.Close()
		if decodeErr != nil {
			return decodeErr
		}
		if closeErr != nil {
			return closeErr
		}
		if v.Progress > 0 {
			fmt.Fprintf(cmd.OutOrStdout(), "progress %.1f%% status %s\n", v.Progress, v.Status)
		}
		switch v.Status {
		case "completed":
			return downloadContent(cmd, url, created.ID)
		case "failed", "cancelled":
			msg := "video generation failed"
			if v.Error != nil && v.Error.Message != "" {
				msg = v.Error.Message
			}
			return fmt.Errorf("%s", msg)
		default:
			if v.Status != "pending" && v.Status != "queued" && v.Status != "in_progress" {
				fmt.Fprintf(cmd.OutOrStdout(), "status %s\n", v.Status)
			}
		}
		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("timed out waiting for video job %s", created.ID)
}

func downloadContent(cmd *cobra.Command, baseURL, id string) error {
	resp, err := http.Get(baseURL + "/v1/videos/" + id + "/content")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("download failed: %s", strings.TrimSpace(string(b)))
	}
	out := filepath.Join(".", id+".mp4")
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	f.Close()
	fmt.Fprintf(cmd.OutOrStdout(), "wrote %s\n", out)
	return nil
}
