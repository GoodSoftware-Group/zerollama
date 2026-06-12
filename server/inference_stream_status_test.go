package server

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGenerateStatusChunkJSON(t *testing.T) {
	chunk := generateStatusChunk("llama3", "queued", "queued (#2 of 3)", 2, 3)
	b, err := json.Marshal(chunk)
	require.NoError(t, err)
	var m map[string]any
	require.NoError(t, json.Unmarshal(b, &m))
	require.Equal(t, "llama3", m["model"])
	require.Equal(t, "queued", m["status"])
	require.Equal(t, "queued (#2 of 3)", m["detail"])
	require.EqualValues(t, 2, m["position"])
	require.EqualValues(t, 3, m["queue_depth"])
	require.Equal(t, false, m["done"])
}

func TestChatStatusChunkJSON(t *testing.T) {
	chunk := chatStatusChunk("llama3", "accepted", "request accepted", 0, 0)
	b, err := json.Marshal(chunk)
	require.NoError(t, err)
	var m map[string]any
	require.NoError(t, json.Unmarshal(b, &m))
	require.Equal(t, "accepted", m["status"])
	require.Equal(t, false, m["done"])
	_, hasPosition := m["position"]
	require.False(t, hasPosition)
}

func TestPendingQueueFifoPosition(t *testing.T) {
	q := newPendingQueue(4)
	req1 := &LlmRequest{fifoSeq: 10}
	req2 := &LlmRequest{fifoSeq: 20}
	require.True(t, q.Push(req1))
	require.True(t, q.Push(req2))

	pos, depth := q.FifoPosition(10)
	require.Equal(t, 1, pos)
	require.Equal(t, 2, depth)

	pos, depth = q.FifoPosition(20)
	require.Equal(t, 2, pos)
	require.Equal(t, 2, depth)

	require.Equal(t, req1, q.Pop())
	pos, depth = q.FifoPosition(20)
	require.Equal(t, 1, pos)
	require.Equal(t, 1, depth)
}
