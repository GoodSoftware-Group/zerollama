// T11c — POST /api/train/recipes: offline ingest helper (no job submit, no GPU).
//
// WHY shell to training_recipes.py: one ingest engine (ShareGPT/OpenAI/Alpaca presets +
// CSV/JSONL). Reimplementing in Go would drift from Python unit tests.
// WHY not auto-submit /api/train/jobs: operators still choose host, T6 window, priority.
package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/x/trainingworker"
)

func (s *Server) trainHTTPRecipes(c *gin.Context) {
	var req struct {
		Content    string `json:"content"`
		Filename   string `json:"filename"`
		Format     string `json:"format"`
		Preset     string `json:"preset"`
		ModelName  string `json:"model_name"`
		PythonBin  string `json:"python"` // optional override (lab)
	}
	ct := c.ContentType()
	if strings.HasPrefix(ct, "multipart/form-data") {
		fh, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "multipart field 'file' required (or send JSON content)"})
			return
		}
		f, err := fh.Open()
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		defer f.Close()
		raw, err := io.ReadAll(io.LimitReader(f, 32<<20)) // 32 MiB cap
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		req.Content = string(raw)
		req.Filename = fh.Filename
		req.Format = c.PostForm("format")
		req.Preset = c.PostForm("preset")
		req.ModelName = c.PostForm("model_name")
	} else {
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}
	if strings.TrimSpace(req.Content) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "content required (JSON body or multipart file)"})
		return
	}
	if req.Filename == "" {
		req.Filename = "upload.json"
	}
	if req.Format == "" {
		req.Format = "auto"
	}
	if req.Preset == "" {
		req.Preset = "auto"
	}

	out, err := runTrainingRecipeStdin(recipeStdinReq{
		Content:   req.Content,
		Filename:  req.Filename,
		Format:    req.Format,
		Preset:    req.Preset,
		ModelName: req.ModelName,
		PythonBin: req.PythonBin,
	})
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.Data(http.StatusOK, "application/json", out)
}

type recipeStdinReq struct {
	Content, Filename, Format, Preset, ModelName, PythonBin string
}

func runTrainingRecipeStdin(req recipeStdinReq) ([]byte, error) {
	root, err := trainingworker.RepoRoot()
	if err != nil {
		return nil, err
	}
	if root == "" {
		return nil, fmt.Errorf("cannot locate repository root (set ZEROLLAMA_REPO or OLLAMA_TRAINING_PYTHONPATH)")
	}
	script := filepath.Join(root, "training_recipes.py")
	if st, err := os.Stat(script); err != nil || st.IsDir() {
		return nil, fmt.Errorf("training_recipes.py not found under %s", root)
	}
	py, err := resolveRecipePython(root, req.PythonBin)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(map[string]string{
		"content":     req.Content,
		"filename":    req.Filename,
		"format":      req.Format,
		"preset":      req.Preset,
		"model_name":  req.ModelName,
	})
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(py, script, "--stdin")
	cmd.Dir = root
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Env = os.Environ()
	done := make(chan error, 1)
	go func() { done <- cmd.Run() }()
	select {
	case err := <-done:
		if err != nil {
			msg := strings.TrimSpace(stderr.String())
			if msg == "" {
				msg = err.Error()
			}
			// Strip leading "error: " from Python helper for cleaner API errors.
			msg = strings.TrimPrefix(msg, "error: ")
			return nil, fmt.Errorf("%s", msg)
		}
	case <-time.After(60 * time.Second):
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("recipe ingest timed out")
	}
	out := bytes.TrimSpace(stdout.Bytes())
	if len(out) == 0 {
		return nil, fmt.Errorf("recipe ingest returned empty body")
	}
	if !json.Valid(out) {
		return nil, fmt.Errorf("recipe ingest returned non-JSON")
	}
	return out, nil
}

// resolveRecipePython mirrors cmd/train.go (keep in sync).
func resolveRecipePython(repoRoot, explicit string) (string, error) {
	if p := strings.TrimSpace(explicit); p != "" {
		return p, nil
	}
	if p := strings.TrimSpace(os.Getenv("TRAINING_PYTHON")); p != "" {
		return p, nil
	}
	if runtime.GOOS == "windows" {
		cand := filepath.Join(repoRoot, ".venv-training", "Scripts", "python.exe")
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
	} else {
		cand := filepath.Join(repoRoot, ".venv-training", "bin", "python")
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
	}
	if _, err := exec.LookPath("python3"); err == nil {
		return "python3", nil
	}
	if _, err := exec.LookPath("python"); err == nil {
		return "python", nil
	}
	return "", fmt.Errorf("no python3/python on PATH (or set TRAINING_PYTHON)")
}
