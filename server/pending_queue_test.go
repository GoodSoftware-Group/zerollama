package server

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPendingQueueDrainMatchingPreservesOrder(t *testing.T) {
	q := newPendingQueue(8)
	m1 := &LlmRequest{model: &Model{ModelPath: "/a"}}
	m2 := &LlmRequest{model: &Model{ModelPath: "/b"}}
	m3 := &LlmRequest{model: &Model{ModelPath: "/a"}}
	require.True(t, q.Push(m1))
	require.True(t, q.Push(m2))
	require.True(t, q.Push(m3))

	drained := q.DrainMatching("/a")
	require.Len(t, drained, 2)
	require.Same(t, m1, drained[0])
	require.Same(t, m3, drained[1])
	require.Equal(t, 1, q.Len())
	require.Same(t, m2, q.Pop())
}

func TestPendingQueueRemove(t *testing.T) {
	q := newPendingQueue(8)
	a := &LlmRequest{model: &Model{ModelPath: "/a"}}
	b := &LlmRequest{model: &Model{ModelPath: "/b"}}
	require.True(t, q.Push(a))
	require.True(t, q.Push(b))
	require.False(t, q.Remove(nil))
	require.True(t, q.Remove(a))
	require.Equal(t, 1, q.Len())
	require.False(t, q.Remove(a))
	require.Same(t, b, q.Pop())
}

func TestPendingQueuePopPreferringKeys(t *testing.T) {
	q := newPendingQueue(8)
	b := &LlmRequest{model: &Model{ModelPath: "/b"}}
	a2 := &LlmRequest{model: &Model{ModelPath: "/a"}}
	require.True(t, q.Push(b))
	require.True(t, q.Push(a2))

	prefer := map[string]struct{}{"/a": {}}
	got := q.PopPreferringKeys(prefer)
	require.Same(t, a2, got)
	require.Equal(t, 1, q.Len())
	require.Same(t, b, q.Pop())
}

func TestPendingQueueRequeueFront(t *testing.T) {
	q := newPendingQueue(8)
	b := &LlmRequest{model: &Model{ModelPath: "/b"}}
	require.True(t, q.Push(b))
	q.RequeueFront([]*LlmRequest{
		{model: &Model{ModelPath: "/a"}},
		{model: &Model{ModelPath: "/a2"}},
	})
	require.Equal(t, 3, q.Len())
	require.Equal(t, "/a", q.Pop().model.ModelPath)
	require.Equal(t, "/a2", q.Pop().model.ModelPath)
	require.Equal(t, "/b", q.Pop().model.ModelPath)
}

func TestPendingQueueCountsByModelKey(t *testing.T) {
	q := newPendingQueue(8)
	require.Nil(t, q.CountsByModelKey())
	require.True(t, q.Push(&LlmRequest{model: &Model{ModelPath: "/a"}}))
	require.True(t, q.Push(&LlmRequest{model: &Model{ModelPath: "/b"}}))
	require.True(t, q.Push(&LlmRequest{model: &Model{ModelPath: "/a"}}))
	got := q.CountsByModelKey()
	require.Equal(t, 2, got["/a"])
	require.Equal(t, 1, got["/b"])
}
