package server

import (
	"context"
	"time"

	"github.com/ollama/ollama/api"
)

func generateStatusChunk(model, status, detail string, position, queueDepth int) api.GenerateResponse {
	return api.GenerateResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		Done:       false,
		Status:     status,
		Position:   position,
		QueueDepth: queueDepth,
		Detail:     detail,
	}
}

func chatStatusChunk(model, status, detail string, position, queueDepth int) api.ChatResponse {
	return api.ChatResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		Done:       false,
		Message:    api.Message{Role: "assistant"},
		Status:     status,
		Position:   position,
		QueueDepth: queueDepth,
		Detail:     detail,
	}
}

func writeGenerateStatus(ch chan<- any, model, status, detail string, position, queueDepth int) {
	if ch == nil {
		return
	}
	ch <- generateStatusChunk(model, status, detail, position, queueDepth)
}

func writeChatStatus(ch chan<- any, model, status, detail string, position, queueDepth int) {
	if ch == nil {
		return
	}
	ch <- chatStatusChunk(model, status, detail, position, queueDepth)
}

func (s *Server) waitRunnerWithStatus(
	ctx context.Context,
	modelName string,
	ticket uint64,
	runnerCh <-chan *runnerRef,
	errCh <-chan error,
	statusCh chan<- any,
	writeStatus func(ch chan<- any, model, status, detail string, position, queueDepth int),
) (*runnerRef, error) {
	writeStatus(statusCh, modelName, "accepted", "request accepted", 0, 0)

	if ticket == 0 {
		select {
		case runner := <-runnerCh:
			if runner != nil && runner.loading {
				writeStatus(statusCh, modelName, "loading", "loading model into memory", 0, 0)
			}
			writeStatus(statusCh, modelName, "generating", "generating response", 0, 0)
			return runner, nil
		case err := <-errCh:
			return nil, err
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	ticker := time.NewTicker(400 * time.Millisecond)
	defer ticker.Stop()

	lastDetail := ""
	for {
		select {
		case runner := <-runnerCh:
			if runner != nil && runner.loading {
				writeStatus(statusCh, modelName, "loading", "loading model into memory", 0, 0)
			}
			writeStatus(statusCh, modelName, "generating", "generating response", 0, 0)
			return runner, nil
		case err := <-errCh:
			return nil, err
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			if s == nil || s.sched == nil {
				continue
			}
			status, detail, position, depth := s.sched.WaitStatus(ticket)
			if status == "" || detail == lastDetail {
				continue
			}
			lastDetail = detail
			writeStatus(statusCh, modelName, status, detail, position, depth)
		}
	}
}
