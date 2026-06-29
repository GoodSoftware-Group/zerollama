package discover

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// PackANEMILWeightBlob wraps fp16 conv weights in the maderix BLOBFILE layout used by
// tools/ane-draft/draft_bench.m and ane-draft-daemon.
func PackANEMILWeightBlob(fp16Weights []byte) ([]byte, error) {
	if len(fp16Weights) == 0 || len(fp16Weights)%2 != 0 {
		return nil, fmt.Errorf("fp16 weights must be non-empty and 2-byte aligned")
	}
	wsize := len(fp16Weights)
	total := 64 + 64 + wsize
	buf := make([]byte, total)
	buf[0] = 0x01
	buf[4] = 0x02
	chunk := buf[64:]
	chunk[0] = 0xEF
	chunk[1] = 0xBE
	chunk[2] = 0xAD
	chunk[3] = 0xDE
	chunk[4] = 0x01
	chunk[10] = 0x08
	binary.LittleEndian.PutUint32(chunk[8:], uint32(wsize))  // matches maderix build_blob @+72
	binary.LittleEndian.PutUint32(chunk[16:], 128)         // fp16 payload starts at file byte 128
	copy(chunk[64:], fp16Weights)
	return buf, nil
}

// matmulExtractTranspose picks how to map GGUF rows/cols into MIL W[ic,oc].
// Llama FFN gate/up are [n_embd,n_ff]; down is [n_ff,n_embd] — the old shape[0]<=shape[1]
// heuristic mis-reads down (2688×768) as transposed.
func matmulExtractTranspose(tensor *ggml.Tensor, inCh, outCh int) bool {
	if tensor == nil || len(tensor.Shape) < 2 {
		return false
	}
	rows, cols := tensor.Shape[0], tensor.Shape[1]
	if rows >= uint64(inCh) && cols >= uint64(outCh) {
		return true
	}
	if cols >= uint64(inCh) && rows >= uint64(outCh) {
		return false
	}
	return convTransposeFromGGUF(tensor)
}

// ExtractMatmulRectFP16 packs W_gguf[ic,oc] in MIL W0 [ic,oc] row-major (not conv [oc,ic] layout).
func ExtractMatmulRectFP16(raw []byte, tensor *ggml.Tensor, inCh, outCh int) ([]byte, error) {
	if tensor == nil {
		return nil, fmt.Errorf("nil tensor")
	}
	if inCh <= 0 || outCh <= 0 {
		return nil, fmt.Errorf("inCh/outCh must be positive")
	}
	if len(tensor.Shape) < 2 {
		return nil, fmt.Errorf("tensor %q needs rank-2 shape, got %v", tensor.Name, tensor.Shape)
	}
	rows := tensor.Shape[0]
	cols := tensor.Shape[1]
	if rows < uint64(inCh) || cols < uint64(outCh) {
		if cols >= uint64(inCh) && rows >= uint64(outCh) {
			// Allow [out,in] storage when caller dims match transposed shape.
		} else {
			return nil, fmt.Errorf("tensor %q shape [%d,%d] smaller than %d×%d proxy", tensor.Name, rows, cols, inCh, outCh)
		}
	}
	transpose := matmulExtractTranspose(tensor, inCh, outCh)
	kind := ggml.TensorType(tensor.Kind)
	switch kind {
	case ggml.TensorTypeF16:
		return extractMatmulRectF16(raw, int(cols), inCh, outCh, transpose)
	case ggml.TensorTypeF32:
		return extractMatmulRectF32(raw, int(cols), inCh, outCh, transpose)
	case ggml.TensorTypeBF16:
		return extractMatmulRectBF16(raw, int(cols), inCh, outCh, transpose)
	default:
		return nil, fmt.Errorf("tensor %q kind %v unsupported for matmul MIL extract", tensor.Name, kind)
	}
}

func extractMatmulRectF16(raw []byte, cols, inCh, outCh int, transpose bool) ([]byte, error) {
	out := make([]byte, inCh*outCh*2)
	rowBytes := cols * 2
	for ic := 0; ic < inCh; ic++ {
		for oc := 0; oc < outCh; oc++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			srcOff := srcR*rowBytes + srcC*2
			if srcOff+2 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			copy(out[(ic*outCh+oc)*2:(ic*outCh+oc)*2+2], raw[srcOff:srcOff+2])
		}
	}
	return out, nil
}

func extractMatmulRectBF16(raw []byte, cols, inCh, outCh int, transpose bool) ([]byte, error) {
	out := make([]byte, inCh*outCh*2)
	rowBytes := cols * 2
	for ic := 0; ic < inCh; ic++ {
		for oc := 0; oc < outCh; oc++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			srcOff := srcR*rowBytes + srcC*2
			if srcOff+2 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			h := binary.LittleEndian.Uint16(raw[srcOff : srcOff+2])
			binary.LittleEndian.PutUint16(out[(ic*outCh+oc)*2:], float32ToFloat16Bits(bfloat16BitsToFloat32(h)))
		}
	}
	return out, nil
}

func extractMatmulRectF32(raw []byte, cols, inCh, outCh int, transpose bool) ([]byte, error) {
	out := make([]byte, inCh*outCh*2)
	rowFloats := cols
	for ic := 0; ic < inCh; ic++ {
		for oc := 0; oc < outCh; oc++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			off := (srcR*rowFloats + srcC) * 4
			if off+4 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			f := math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
			binary.LittleEndian.PutUint16(out[(ic*outCh+oc)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

// ExtractTopLeftRectFP16 reads the top-left inCh×outCh block from a rank-2 GGUF weight.
func ExtractTopLeftRectFP16(raw []byte, tensor *ggml.Tensor, inCh, outCh int, transpose bool) ([]byte, error) {
	if tensor == nil {
		return nil, fmt.Errorf("nil tensor")
	}
	if inCh <= 0 || outCh <= 0 {
		return nil, fmt.Errorf("inCh/outCh must be positive")
	}
	if len(tensor.Shape) < 2 {
		return nil, fmt.Errorf("tensor %q needs rank-2 shape, got %v", tensor.Name, tensor.Shape)
	}
	rows := tensor.Shape[0]
	cols := tensor.Shape[1]
	if rows < uint64(inCh) || cols < uint64(outCh) {
		return nil, fmt.Errorf("tensor %q shape [%d,%d] smaller than %d×%d proxy", tensor.Name, rows, cols, inCh, outCh)
	}
	kind := ggml.TensorType(tensor.Kind)
	switch kind {
	case ggml.TensorTypeF16:
		return extractRectF16(raw, int(cols), inCh, outCh, transpose)
	case ggml.TensorTypeF32:
		return extractRectF32(raw, int(cols), inCh, outCh, transpose)
	case ggml.TensorTypeBF16:
		return extractRectBF16(raw, int(cols), inCh, outCh, transpose)
	case ggml.TensorTypeQ8_0:
		return extractRectQ8_0(raw, int(rows), int(cols), inCh, outCh, transpose)
	default:
		return nil, fmt.Errorf("tensor %q kind %v unsupported for MIL extract", tensor.Name, kind)
	}
}

func extractRectF16(raw []byte, cols, inCh, outCh int, transpose bool) ([]byte, error) {
	out := make([]byte, inCh*outCh*2)
	rowBytes := cols * 2
	for oc := 0; oc < outCh; oc++ {
		for ic := 0; ic < inCh; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			srcOff := srcR*rowBytes + srcC*2
			if srcOff+2 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			copy(out[(oc*inCh+ic)*2:(oc*inCh+ic)*2+2], raw[srcOff:srcOff+2])
		}
	}
	return out, nil
}

func extractRectBF16(raw []byte, cols, inCh, outCh int, transpose bool) ([]byte, error) {
	out := make([]byte, inCh*outCh*2)
	rowBytes := cols * 2
	for oc := 0; oc < outCh; oc++ {
		for ic := 0; ic < inCh; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			srcOff := srcR*rowBytes + srcC*2
			if srcOff+2 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			h := binary.LittleEndian.Uint16(raw[srcOff : srcOff+2])
			binary.LittleEndian.PutUint16(out[(oc*inCh+ic)*2:], float32ToFloat16Bits(bfloat16BitsToFloat32(h)))
		}
	}
	return out, nil
}

func extractRectF32(raw []byte, cols, inCh, outCh int, transpose bool) ([]byte, error) {
	out := make([]byte, inCh*outCh*2)
	rowFloats := cols
	for oc := 0; oc < outCh; oc++ {
		for ic := 0; ic < inCh; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			off := (srcR*rowFloats + srcC) * 4
			if off+4 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			f := math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
			binary.LittleEndian.PutUint16(out[(oc*inCh+ic)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

func extractRectQ8_0(raw []byte, ne0, ne1, inCh, outCh int, transpose bool) ([]byte, error) {
	out := make([]byte, inCh*outCh*2)
	for oc := 0; oc < outCh; oc++ {
		for ic := 0; ic < inCh; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			f, err := q8_0Element(raw, ne0, srcR, srcC)
			if err != nil {
				return nil, err
			}
			binary.LittleEndian.PutUint16(out[(oc*inCh+ic)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

// ExtractTopLeftSquareFP16 reads the top-left channels×channels block from a GGUF
// weight matrix stored row-major. When transpose is true, output[oc,ic]=src[ic,oc]
// so the blob matches ANE conv weight layout [out_ch, in_ch] for x@W matmul slices.
func ExtractTopLeftSquareFP16(raw []byte, tensor *ggml.Tensor, channels int, transpose bool) ([]byte, error) {
	if tensor == nil {
		return nil, fmt.Errorf("nil tensor")
	}
	if channels <= 0 {
		return nil, fmt.Errorf("channels must be positive")
	}
	if len(tensor.Shape) < 2 {
		return nil, fmt.Errorf("tensor %q needs rank-2 shape, got %v", tensor.Name, tensor.Shape)
	}

	rows := tensor.Shape[0]
	cols := tensor.Shape[1]
	if rows < uint64(channels) || cols < uint64(channels) {
		return nil, fmt.Errorf("tensor %q shape [%d,%d] smaller than %d×%d proxy", tensor.Name, rows, cols, channels, channels)
	}

	kind := ggml.TensorType(tensor.Kind)
	switch kind {
	case ggml.TensorTypeF16:
		return extractSquareF16(raw, int(cols), channels, transpose)
	case ggml.TensorTypeF32:
		return extractSquareF32(raw, int(cols), channels, transpose)
	case ggml.TensorTypeBF16:
		return extractSquareBF16(raw, int(cols), channels, transpose)
	case ggml.TensorTypeQ8_0:
		return extractSquareQ8_0(raw, int(rows), int(cols), channels, transpose)
	default:
		return nil, fmt.Errorf("tensor %q kind %v unsupported for MIL extract (need f16/f32/bf16/q8_0)", tensor.Name, kind)
	}
}

const (
	qk8_0         = 32
	blockQ8_0Size = 2 + qk8_0 // fp16 scale + int8 quants
)

func q8_0Element(raw []byte, ne0, i0, i1 int) (float32, error) {
	if ne0%qk8_0 != 0 {
		return 0, fmt.Errorf("ne0 %d not divisible by %d", ne0, qk8_0)
	}
	blocksPerRow := ne0 / qk8_0
	rowBytes := blocksPerRow * blockQ8_0Size
	blockInRow := i0 / qk8_0
	qInBlock := i0 % qk8_0
	off := i1*rowBytes + blockInRow*blockQ8_0Size
	if off+blockQ8_0Size > len(raw) {
		return 0, fmt.Errorf("tensor data truncated at (%d,%d)", i0, i1)
	}
	d := fp16BitsToFloat32(binary.LittleEndian.Uint16(raw[off : off+2]))
	q := int8(raw[off+2+qInBlock])
	return float32(q) * d, nil
}

func extractSquareQ8_0(raw []byte, ne0, ne1, channels int, transpose bool) ([]byte, error) {
	if ne0 < channels || ne1 < channels {
		return nil, fmt.Errorf("tensor shape [%d,%d] smaller than %d×%d proxy", ne0, ne1, channels, channels)
	}
	out := make([]byte, channels*channels*2)
	for oc := 0; oc < channels; oc++ {
		for ic := 0; ic < channels; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			f, err := q8_0Element(raw, ne0, srcR, srcC)
			if err != nil {
				return nil, err
			}
			binary.LittleEndian.PutUint16(out[(oc*channels+ic)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

func fp16BitsToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := h & 0x3ff
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		f := float64(mant) / 1024.0 * math.Pow(2, -14)
		if sign != 0 {
			f = -f
		}
		return float32(f)
	case 31:
		if mant != 0 {
			return math.Float32frombits(sign | 0x7fc00000)
		}
		return math.Float32frombits(sign | 0x7f800000)
	default:
		f := math.Ldexp(1+float64(mant)/1024, int(exp)-15)
		if sign != 0 {
			f = -f
		}
		return float32(f)
	}
}

func extractSquareF16(raw []byte, cols, channels int, transpose bool) ([]byte, error) {
	need := channels * channels * 2
	out := make([]byte, need)
	rowBytes := cols * 2
	for oc := 0; oc < channels; oc++ {
		for ic := 0; ic < channels; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			srcOff := srcR*rowBytes + srcC*2
			if srcOff+2 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			copy(out[(oc*channels+ic)*2:(oc*channels+ic)*2+2], raw[srcOff:srcOff+2])
		}
	}
	return out, nil
}

func extractSquareBF16(raw []byte, cols, channels int, transpose bool) ([]byte, error) {
	need := channels * channels * 2
	out := make([]byte, need)
	rowBytes := cols * 2
	for oc := 0; oc < channels; oc++ {
		for ic := 0; ic < channels; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			srcOff := srcR*rowBytes + srcC*2
			if srcOff+2 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			h := binary.LittleEndian.Uint16(raw[srcOff : srcOff+2])
			f := bfloat16BitsToFloat32(h)
			binary.LittleEndian.PutUint16(out[(oc*channels+ic)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

func extractSquareF32(raw []byte, cols, channels int, transpose bool) ([]byte, error) {
	need := channels * channels * 2
	out := make([]byte, need)
	rowFloats := cols
	for oc := 0; oc < channels; oc++ {
		for ic := 0; ic < channels; ic++ {
			srcR, srcC := ic, oc
			if !transpose {
				srcR, srcC = oc, ic
			}
			idx := srcR*rowFloats + srcC
			off := idx * 4
			if off+4 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", srcR, srcC)
			}
			f := math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
			binary.LittleEndian.PutUint16(out[(oc*channels+ic)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

func bfloat16BitsToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// convTransposeFromGGUF returns true when a rank-2 tensor is stored [in, out] (llama FFN).
func convTransposeFromGGUF(tensor *ggml.Tensor) bool {
	if tensor == nil || len(tensor.Shape) < 2 {
		return false
	}
	return tensor.Shape[0] <= tensor.Shape[1]
}

func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint32((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff

	switch {
	case exp <= 0:
		if exp < -10 {
			return uint16(sign)
		}
		mant |= 0x800000
		shift := uint32(1 - exp)
		m := mant >> (shift + 13)
		if (mant>>(shift+12))&1 != 0 {
			m++
		}
		return uint16(sign | m)
	case exp >= 31:
		if mant != 0 {
			return uint16(sign | 0x7e00) // NaN
		}
		return uint16(sign | 0x7c00) // Inf
	default:
		return uint16(sign | (uint32(exp) << 10) | (mant >> 13))
	}
}

// DefaultProxyConvTensor picks the sidecar tensor used for lab conv proxy extract.
func DefaultProxyConvTensor() string {
	return "blk.0.ffn_gate.weight"
}

// DefaultProxyConv2Tensor is the second conv in the B6 two-layer proxy subgraph.
func DefaultProxyConv2Tensor() string {
	return "blk.0.ffn_up.weight"
}

// DefaultProxyConv3Tensor is the third conv in the B8 attn-gate proxy expansion.
func DefaultProxyConv3Tensor() string {
	return "blk.0.attn_gate.weight"
}

// DefaultProxyConv3TensorForArch picks B8 third conv tensor for a sidecar architecture.
func DefaultProxyConv3TensorForArch(arch string) string {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft", "qwen35", "qwen3", "eagle3":
		return "blk.0.attn_gate.weight"
	default:
		return DefaultProxyConv3Tensor()
	}
}

// proxyConv3TensorCandidates returns preferred B8 conv3 tensors, most specific first.
func proxyConv3TensorCandidates(arch string) []string {
	primary := DefaultProxyConv3TensorForArch(arch)
	return []string{
		primary,
		"blk.0.attn_gate.weight",
		"blk.0.attn_q.weight",
	}
}

// ResolveProxyConv3TensorForSidecar picks the first B8 conv3 tensor present in the sidecar GGUF.
func ResolveProxyConv3TensorForSidecar(sidecarPath string) (tensor string, arch string, err error) {
	arch, err = ProbeSidecarArchitecture(sidecarPath)
	if err != nil {
		return DefaultProxyConv3Tensor(), "", err
	}
	f, err := os.Open(sidecarPath)
	if err != nil {
		return DefaultProxyConv3TensorForArch(arch), arch, err
	}
	defer f.Close()
	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return DefaultProxyConv3TensorForArch(arch), arch, err
	}
	byName := make(map[string]struct{}, len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		if t != nil && t.Name != "" {
			byName[t.Name] = struct{}{}
		}
	}
	for _, cand := range proxyConv3TensorCandidates(arch) {
		if _, ok := byName[cand]; ok {
			return cand, arch, nil
		}
	}
	return "", arch, fmt.Errorf("no B8 conv3 proxy tensor in %s (arch=%s)", sidecarPath, arch)
}

// DefaultProxyConv3TensorForSidecar resolves B8 conv3 tensor from sidecar architecture.
func DefaultProxyConv3TensorForSidecar(sidecarPath string) (tensor string, err error) {
	tensor, _, err = ResolveProxyConv3TensorForSidecar(sidecarPath)
	return tensor, err
}

// DefaultProxyConv4Tensor is the fourth conv in the B9 ffn_down proxy expansion.
func DefaultProxyConv4Tensor() string {
	return "blk.0.ffn_down.weight"
}

// DefaultProxyConv4TensorForArch picks B9 fourth conv tensor for a sidecar architecture.
func DefaultProxyConv4TensorForArch(arch string) string {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft", "qwen35", "qwen3", "eagle3":
		return "blk.0.ffn_down.weight"
	default:
		return DefaultProxyConv4Tensor()
	}
}

// proxyConv4TensorCandidates returns preferred B9 conv4 tensors, most specific first.
func proxyConv4TensorCandidates(arch string) []string {
	primary := DefaultProxyConv4TensorForArch(arch)
	return []string{
		primary,
		"blk.0.ffn_down.weight",
	}
}

// ResolveProxyConv4TensorForSidecar picks the first B9 conv4 tensor present in the sidecar GGUF.
func ResolveProxyConv4TensorForSidecar(sidecarPath string) (tensor string, arch string, err error) {
	arch, err = ProbeSidecarArchitecture(sidecarPath)
	if err != nil {
		return DefaultProxyConv4Tensor(), "", err
	}
	f, err := os.Open(sidecarPath)
	if err != nil {
		return DefaultProxyConv4TensorForArch(arch), arch, err
	}
	defer f.Close()
	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return DefaultProxyConv4TensorForArch(arch), arch, err
	}
	byName := make(map[string]struct{}, len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		if t != nil && t.Name != "" {
			byName[t.Name] = struct{}{}
		}
	}
	for _, cand := range proxyConv4TensorCandidates(arch) {
		if _, ok := byName[cand]; ok {
			return cand, arch, nil
		}
	}
	return "", arch, fmt.Errorf("no B9 conv4 proxy tensor in %s (arch=%s)", sidecarPath, arch)
}

// DefaultProxyConv4TensorForSidecar resolves B9 conv4 tensor from sidecar architecture.
func DefaultProxyConv4TensorForSidecar(sidecarPath string) (tensor string, err error) {
	tensor, _, err = ResolveProxyConv4TensorForSidecar(sidecarPath)
	return tensor, err
}

// DefaultProxyConv5Tensor is the fifth conv in the B10 blk.1 ffn_gate proxy expansion.
func DefaultProxyConv5Tensor() string {
	return "blk.1.ffn_gate.weight"
}

// DefaultProxyConv5TensorForArch picks B10 fifth conv tensor for a sidecar architecture.
func DefaultProxyConv5TensorForArch(arch string) string {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft", "qwen35", "qwen3", "eagle3":
		return "blk.1.ffn_gate.weight"
	default:
		return DefaultProxyConv5Tensor()
	}
}

func proxyConv5TensorCandidates(arch string) []string {
	primary := DefaultProxyConv5TensorForArch(arch)
	return []string{
		primary,
		"blk.1.ffn_gate.weight",
	}
}

// ResolveProxyConv5TensorForSidecar picks the first B10 conv5 tensor present in the sidecar GGUF.
func ResolveProxyConv5TensorForSidecar(sidecarPath string) (tensor string, arch string, err error) {
	arch, err = ProbeSidecarArchitecture(sidecarPath)
	if err != nil {
		return DefaultProxyConv5Tensor(), "", err
	}
	f, err := os.Open(sidecarPath)
	if err != nil {
		return DefaultProxyConv5TensorForArch(arch), arch, err
	}
	defer f.Close()
	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return DefaultProxyConv5TensorForArch(arch), arch, err
	}
	byName := make(map[string]struct{}, len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		if t != nil && t.Name != "" {
			byName[t.Name] = struct{}{}
		}
	}
	for _, cand := range proxyConv5TensorCandidates(arch) {
		if _, ok := byName[cand]; ok {
			return cand, arch, nil
		}
	}
	return "", arch, fmt.Errorf("no B10 conv5 proxy tensor in %s (arch=%s)", sidecarPath, arch)
}

// DefaultProxyConv5TensorForSidecar resolves B10 conv5 tensor from sidecar architecture.
func DefaultProxyConv5TensorForSidecar(sidecarPath string) (tensor string, err error) {
	tensor, _, err = ResolveProxyConv5TensorForSidecar(sidecarPath)
	return tensor, err
}

// DefaultProxyConv6Tensor is the sixth conv in the B11 blk.1 ffn_up proxy expansion.
func DefaultProxyConv6Tensor() string {
	return "blk.1.ffn_up.weight"
}

// DefaultProxyConv6TensorForArch picks B11 sixth conv tensor for a sidecar architecture.
func DefaultProxyConv6TensorForArch(arch string) string {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft", "qwen35", "qwen3", "eagle3":
		return "blk.1.ffn_up.weight"
	default:
		return DefaultProxyConv6Tensor()
	}
}

func proxyConv6TensorCandidates(arch string) []string {
	primary := DefaultProxyConv6TensorForArch(arch)
	return []string{
		primary,
		"blk.1.ffn_up.weight",
	}
}

// ResolveProxyConv6TensorForSidecar picks the first B11 conv6 tensor present in the sidecar GGUF.
func ResolveProxyConv6TensorForSidecar(sidecarPath string) (tensor string, arch string, err error) {
	arch, err = ProbeSidecarArchitecture(sidecarPath)
	if err != nil {
		return DefaultProxyConv6Tensor(), "", err
	}
	f, err := os.Open(sidecarPath)
	if err != nil {
		return DefaultProxyConv6TensorForArch(arch), arch, err
	}
	defer f.Close()
	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return DefaultProxyConv6TensorForArch(arch), arch, err
	}
	byName := make(map[string]struct{}, len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		if t != nil && t.Name != "" {
			byName[t.Name] = struct{}{}
		}
	}
	for _, cand := range proxyConv6TensorCandidates(arch) {
		if _, ok := byName[cand]; ok {
			return cand, arch, nil
		}
	}
	return "", arch, fmt.Errorf("no B11 conv6 proxy tensor in %s (arch=%s)", sidecarPath, arch)
}

// DefaultProxyConv6TensorForSidecar resolves B11 conv6 tensor from sidecar architecture.
func DefaultProxyConv6TensorForSidecar(sidecarPath string) (tensor string, err error) {
	tensor, _, err = ResolveProxyConv6TensorForSidecar(sidecarPath)
	return tensor, err
}

// DefaultProxyConv7Tensor is the seventh conv in the B12 blk.1 attn_gate proxy expansion.
func DefaultProxyConv7Tensor() string {
	return "blk.1.attn_gate.weight"
}

// DefaultProxyConv7TensorForArch picks B12 seventh conv tensor for a sidecar architecture.
func DefaultProxyConv7TensorForArch(arch string) string {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft", "qwen35", "qwen3", "eagle3":
		return "blk.1.attn_gate.weight"
	default:
		return DefaultProxyConv7Tensor()
	}
}

func proxyConv7TensorCandidates(arch string) []string {
	primary := DefaultProxyConv7TensorForArch(arch)
	return []string{
		primary,
		"blk.1.attn_gate.weight",
	}
}

// ResolveProxyConv7TensorForSidecar picks the first B12 conv7 tensor present in the sidecar GGUF.
func ResolveProxyConv7TensorForSidecar(sidecarPath string) (tensor string, arch string, err error) {
	arch, err = ProbeSidecarArchitecture(sidecarPath)
	if err != nil {
		return DefaultProxyConv7Tensor(), "", err
	}
	f, err := os.Open(sidecarPath)
	if err != nil {
		return DefaultProxyConv7TensorForArch(arch), arch, err
	}
	defer f.Close()
	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return DefaultProxyConv7TensorForArch(arch), arch, err
	}
	byName := make(map[string]struct{}, len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		if t != nil && t.Name != "" {
			byName[t.Name] = struct{}{}
		}
	}
	for _, cand := range proxyConv7TensorCandidates(arch) {
		if _, ok := byName[cand]; ok {
			return cand, arch, nil
		}
	}
	return "", arch, fmt.Errorf("no B12 conv7 proxy tensor in %s (arch=%s)", sidecarPath, arch)
}

// DefaultProxyConv7TensorForSidecar resolves B12 conv7 tensor from sidecar architecture.
func DefaultProxyConv7TensorForSidecar(sidecarPath string) (tensor string, err error) {
	tensor, _, err = ResolveProxyConv7TensorForSidecar(sidecarPath)
	return tensor, err
}

// DefaultProxyConv8Tensor is the eighth conv in the B13 blk.1 ffn_down proxy expansion.
func DefaultProxyConv8Tensor() string {
	return "blk.1.ffn_down.weight"
}

// DefaultProxyConv8TensorForArch picks B13 eighth conv tensor for a sidecar architecture.
func DefaultProxyConv8TensorForArch(arch string) string {
	switch strings.ToLower(strings.TrimSpace(arch)) {
	case "dflash-draft", "qwen35", "qwen3", "eagle3":
		return "blk.1.ffn_down.weight"
	default:
		return DefaultProxyConv8Tensor()
	}
}

func proxyConv8TensorCandidates(arch string) []string {
	primary := DefaultProxyConv8TensorForArch(arch)
	return []string{
		primary,
		"blk.1.ffn_down.weight",
	}
}

// ResolveProxyConv8TensorForSidecar picks the first B13 conv8 tensor present in the sidecar GGUF.
func ResolveProxyConv8TensorForSidecar(sidecarPath string) (tensor string, arch string, err error) {
	arch, err = ProbeSidecarArchitecture(sidecarPath)
	if err != nil {
		return DefaultProxyConv8Tensor(), "", err
	}
	f, err := os.Open(sidecarPath)
	if err != nil {
		return DefaultProxyConv8TensorForArch(arch), arch, err
	}
	defer f.Close()
	meta, err := ggml.DecodeMetadata(f)
	if err != nil {
		return DefaultProxyConv8TensorForArch(arch), arch, err
	}
	byName := make(map[string]struct{}, len(meta.Tensors().Items()))
	for _, t := range meta.Tensors().Items() {
		if t != nil && t.Name != "" {
			byName[t.Name] = struct{}{}
		}
	}
	for _, cand := range proxyConv8TensorCandidates(arch) {
		if _, ok := byName[cand]; ok {
			return cand, arch, nil
		}
	}
	return "", arch, fmt.Errorf("no B13 conv8 proxy tensor in %s (arch=%s)", sidecarPath, arch)
}

// DefaultProxyConv8TensorForSidecar resolves B13 conv8 tensor from sidecar architecture.
func DefaultProxyConv8TensorForSidecar(sidecarPath string) (tensor string, err error) {
	tensor, _, err = ResolveProxyConv8TensorForSidecar(sidecarPath)
	return tensor, err
}

// ExtractNormVectorWeightBlob packs the first channels elements of a 1D norm weight.
func ExtractNormVectorWeightBlob(ggufPath, tensorName string, channels int) ([]byte, *ggml.Tensor, error) {
	if channels <= 0 {
		return nil, nil, fmt.Errorf("channels must be positive")
	}
	if tensorName == "" {
		return nil, nil, fmt.Errorf("empty tensor name")
	}

	raw, tensor, err := ggml.ReadTensorBytes(ggufPath, tensorName)
	if err != nil {
		return nil, nil, err
	}
	if len(tensor.Shape) != 1 {
		return nil, tensor, fmt.Errorf("tensor %q needs rank-1 shape, got %v", tensor.Name, tensor.Shape)
	}
	if tensor.Shape[0] < uint64(channels) {
		return nil, tensor, fmt.Errorf("tensor %q len %d < channels %d", tensor.Name, tensor.Shape[0], channels)
	}

	fp16, err := extractVectorFP16(raw, tensor, channels)
	if err != nil {
		return nil, tensor, err
	}
	blob, err := PackANEMILWeightBlob(fp16)
	if err != nil {
		return nil, tensor, err
	}
	return blob, tensor, nil
}

func extractVectorFP16(raw []byte, tensor *ggml.Tensor, channels int) ([]byte, error) {
	kind := ggml.TensorType(tensor.Kind)
	switch kind {
	case ggml.TensorTypeF16:
		if len(raw) < channels*2 {
			return nil, fmt.Errorf("tensor data truncated")
		}
		return append([]byte(nil), raw[:channels*2]...), nil
	case ggml.TensorTypeBF16:
		out := make([]byte, channels*2)
		for i := 0; i < channels; i++ {
			h := binary.LittleEndian.Uint16(raw[i*2:])
			binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(bfloat16BitsToFloat32(h)))
		}
		return out, nil
	case ggml.TensorTypeF32:
		out := make([]byte, channels*2)
		for i := 0; i < channels; i++ {
			f := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16Bits(f))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("tensor %q kind %v unsupported for norm extract", tensor.Name, kind)
	}
}
