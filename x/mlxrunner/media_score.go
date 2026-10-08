package mlxrunner

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// scoreMediaItem is one media occurrence in a scoring request's token stream.
type scoreMediaItem struct {
	pos    int
	length int
	fold   uint32
	item   *base.PreparedItem
}

func (item *scoreMediaItem) atomic() bool { return item.item != nil && !item.item.Causal }

func foldValue(data []byte, dims []int) uint32 {
	h := fnv.New64a()
	h.Write(data)
	var b [8]byte
	for _, d := range dims {
		binary.LittleEndian.PutUint64(b[:], uint64(d))
		h.Write(b[:])
	}
	sum := h.Sum64()
	return (uint32(sum>>32) ^ uint32(sum)) | 1<<31
}

type scoreRequestMedia struct {
	model    base.MediaModel
	items    []scoreMediaItem
	inputLen int
	manifest []batch.MediaItem
	features []*mlx.Array
	scope    *mlx.Scope
	layout   []any
}

func (r *Runner) openScoreMedia(prepared *base.PreparedRequest, segments []base.Segment, layout any) (*scoreRequestMedia, error) {
	items, err := bindScoreItems(prepared, segments)
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}
	mm, ok := r.Model.(base.MediaModel)
	if !ok {
		return nil, fmt.Errorf("model does not support decision media")
	}
	m := &scoreRequestMedia{
		model:    mm,
		items:    items,
		inputLen: len(prepared.Tokens),
		manifest: make([]batch.MediaItem, len(items)),
		features: make([]*mlx.Array, len(items)),
		scope:    mlx.NewScope(),
	}
	if layout != nil {
		m.layout = []any{layout}
	}
	for i, item := range m.items {
		m.manifest[i] = batch.MediaItem{Pos: item.pos, Opaque: item.item.Opaque}
	}
	return m, nil
}

func (m *scoreRequestMedia) rowLayout() []any {
	if m == nil {
		return nil
	}
	return m.layout
}

func (m *scoreRequestMedia) extendChunk(pos, n int) int {
	if m == nil {
		return n
	}
	end := pos + n
	for i := range m.items {
		item := &m.items[i]
		if !item.atomic() {
			continue
		}
		if item.pos < end && end < item.pos+item.length {
			if item.pos > pos {
				return item.pos - pos
			}
			return min(item.pos+item.length, m.inputLen) - pos
		}
	}
	return n
}

func (m *scoreRequestMedia) batchMedia(pos, n int) []batch.MediaItem {
	if m == nil {
		return nil
	}
	for i, item := range m.items {
		if item.pos >= pos+n || item.pos+item.length <= pos {
			continue
		}
		if m.features[i] == nil {
			m.features[i] = mlx.ScopedArrays(func() []*mlx.Array {
				data := mlx.FromValues(item.item.MediaData, item.item.Dims...)
				return []*mlx.Array{m.model.EncodeMedia(item.item, data)}
			})[0]
			m.scope.Attach(m.features[i])
			item.item.MediaData = nil
		}
		m.manifest[i].Features = m.features[i]
	}
	return m.manifest
}

func (m *scoreRequestMedia) free(pos int) {
	if m == nil {
		return
	}
	for i, item := range m.items {
		if item.pos+item.length <= pos {
			item.item.MediaData = nil
			if m.features[i] != nil {
				m.scope.Discard(m.features[i])
				m.features[i] = nil
				m.manifest[i].Features = nil
			}
		}
	}
}

func (m *scoreRequestMedia) close() {
	if m == nil {
		return
	}
	for i := range m.features {
		m.features[i] = nil
		m.manifest[i].Features = nil
	}
	m.scope.Close()
}

func bindScoreItems(prepared *base.PreparedRequest, segments []base.Segment) ([]scoreMediaItem, error) {
	covered := make([]bool, len(segments))
	items := make([]scoreMediaItem, 0, len(prepared.Items))
	end := 0
	for i := range prepared.Items {
		item := &prepared.Items[i]
		rg := item.Range
		if rg[0] < end || rg[1] <= rg[0] || rg[1] > len(prepared.Tokens) {
			return nil, fmt.Errorf("media expansion has invalid range %v", rg)
		}
		if item.Source < 0 || item.Source >= len(segments) || segments[item.Source].Data == nil {
			return nil, fmt.Errorf("media expansion references non-media segment %d", item.Source)
		}
		covered[item.Source] = true
		end = rg[1]
		items = append(items, scoreMediaItem{
			pos:    rg[0],
			length: rg[1] - rg[0],
			fold:   foldValue(segments[item.Source].Data, item.Dims),
			item:   item,
		})
	}
	for s, seg := range segments {
		if seg.Data != nil && !covered[s] {
			return nil, errors.New("media expansion produced no tokens")
		}
	}
	return items, nil
}

func bindScoreMediaItems(prepared *base.PreparedRequest, segments []base.Segment) ([]scoreMediaItem, error) {
	return bindScoreItems(prepared, segments)
}
