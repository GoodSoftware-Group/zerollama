// In-process ANE prefill FFN-slice session.
// Modes:
//   1) fp16-blob matmul sandwich (single GEMM) — rectangular IC→OC
//   2) fused SwiGLU via 1×1 conv (gate+up+silu*up+down) — proven in maderix training
//
// MIL note: buildInfo braces must be {{ }} inside stringWithFormat; never put {{ in
// appendString (that emits literal double braces → InvalidMILProgram).
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

#include "ane_prefill_session.h"
#include "ane_bridge.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

struct ANEPrefillSession {
    int ic;
    int oc;       // matmul out channels; for swiglu equals ic (embed restore)
    int hidden;   // 0 = matmul-only
    int seq;
    bool swiglu;
    bool int8;
    bool int8Input;
    float int8Scale;
    size_t inBytes;
    size_t outBytes;
    double compileMs;
    int evalCount;
    ANEKernelHandle *kernel;
};

static mach_timebase_info_data_t g_tb;
static bool g_tb_init = false;

static double ticksToMs(uint64_t t) {
    if (!g_tb_init) {
        mach_timebase_info(&g_tb);
        g_tb_init = true;
    }
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

// Acts [1,IC,1,SEQ] + BLOBFILE W [1,1,IC,OC] → [1,OC,1,SEQ]
static NSString *genFP16BlobMIL(int ic, int oc, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        @"        tensor<fp16, [1,1,%d,%d]> W = const()[name=string(\"W\"), "
        @"val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        @"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        @"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=x)[name=string(\"a2\")];\n"
        @"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"
        @"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n"
        @"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
        @"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n"
        @"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n"
        @"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        @"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n"
        @"    } -> (y);\n"
        @"}\n",
        ic, seq,
        ic, oc, ic, oc,
        ic, seq, ic, seq,
        seq, ic,
        seq, oc,
        oc, seq,
        oc, seq,
        oc, seq];
}

// Fused SwiGLU FFN — maderix test_full_fused pattern (1×1 conv, multi BLOBFILE).
static NSString *genSwiGLUConvMIL(int ic, int hidden, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        @"        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        @"        tensor<fp16, [%d, %d, 1, 1]> Wg = const()[name = string(\"Wg\"), "
        @"val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wg.bin\"), offset = uint64(64)))];\n"
        @"        tensor<fp16, [%d, %d, 1, 1]> Wu = const()[name = string(\"Wu\"), "
        @"val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wu.bin\"), offset = uint64(64)))];\n"
        @"        tensor<fp16, [%d, %d, 1, 1]> Wd = const()[name = string(\"Wd\"), "
        @"val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wd.bin\"), offset = uint64(64)))];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wg, x = x)[name = string(\"cg\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wu, x = x)[name = string(\"cu\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = string(\"sg\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = string(\"si\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> hid = mul(x = silu, y = up)[name = string(\"gt\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wd, x = hid)[name = string(\"cd\")];\n"
        @"    } -> (y);\n"
        @"}\n",
        ic, seq,
        hidden, ic, hidden, ic,
        hidden, ic, hidden, ic,
        ic, hidden, ic, hidden,
        hidden, seq, hidden, seq,
        hidden, seq, hidden, seq, hidden, seq,
        ic, seq];
}

// int8 weight BLOBFILE + 1×1 conv (maderix constexpr_affine_dequantize).
static NSString *genInt8ConvMIL(int ic, int oc, int seq, float scale) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendString:
        @"        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"];
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize()"
        @"[axis = int32(0), name = string(\"W\"), "
        @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        @"(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64))), "
        @"scale = fp16(%a), zero_point = int8(0)];\n",
        oc, ic, oc, ic, (double)(_Float16)scale];
    [m appendFormat:
        @"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = W, x = x)[name = string(\"y\")];\n"
        @"    } -> (y);\n"
        @"}\n",
        oc, seq];
    return m;
}

// int8 SwiGLU: three constexpr_affine_dequantize weight BLOBFILEs.
// Optional W8A8 on input x and/or hid; optional 2D spatial tiling (sp0*sp1==seq).
// int8_input: main() takes tensor<int8> and only dequantizes (host already quantized).
static NSString *genInt8SwiGLUMIL(int ic, int hidden, int sp0, int sp1,
                                  float scaleG, float scaleU, float scaleD,
                                  float hid_scale, float x_scale,
                                  bool int8_input) {
    bool qhid = hid_scale > 0;
    bool qx = x_scale > 0 && !int8_input;
    bool dx = x_scale > 0 && int8_input;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    if (int8_input) {
        [m appendFormat:@"    func main<ios18>(tensor<int8, [1, %d, %d, %d]> x) {\n", ic, sp0, sp1];
    } else {
        [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n", ic, sp0, sp1];
    }
    [m appendString:
        @"        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        @"        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"];
    if (qx) {
        [m appendFormat:
            @"        fp16 x_qs = const()[name = string(\"x_qs\"), val = fp16(%a)];\n"
            @"        fp16 x_dqs = const()[name = string(\"x_dqs\"), val = fp16(%a)];\n",
            (double)(_Float16)x_scale, (double)(_Float16)x_scale];
    } else if (dx) {
        [m appendFormat:
            @"        fp16 x_dqs = const()[name = string(\"x_dqs\"), val = fp16(%a)];\n",
            (double)(_Float16)x_scale];
    }
    if (qhid) {
        [m appendFormat:
            @"        fp16 h_qs = const()[name = string(\"h_qs\"), val = fp16(%a)];\n"
            @"        fp16 h_dqs = const()[name = string(\"h_dqs\"), val = fp16(%a)];\n",
            (double)(_Float16)hid_scale, (double)(_Float16)hid_scale];
    }
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> Wg = constexpr_affine_dequantize()"
        @"[axis = int32(0), name = string(\"Wg\"), "
        @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        @"(BLOBFILE(path = string(\"@model_path/weights/wg.bin\"), offset = uint64(64))), "
        @"scale = fp16(%a), zero_point = int8(0)];\n",
        hidden, ic, hidden, ic, (double)(_Float16)scaleG];
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> Wu = constexpr_affine_dequantize()"
        @"[axis = int32(0), name = string(\"Wu\"), "
        @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        @"(BLOBFILE(path = string(\"@model_path/weights/wu.bin\"), offset = uint64(64))), "
        @"scale = fp16(%a), zero_point = int8(0)];\n",
        hidden, ic, hidden, ic, (double)(_Float16)scaleU];
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> Wd = constexpr_affine_dequantize()"
        @"[axis = int32(0), name = string(\"Wd\"), "
        @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        @"(BLOBFILE(path = string(\"@model_path/weights/wd.bin\"), offset = uint64(64))), "
        @"scale = fp16(%a), zero_point = int8(0)];\n",
        ic, hidden, ic, hidden, (double)(_Float16)scaleD];
    NSString *xin = @"x";
    if (qx) {
        [m appendFormat:
            @"        tensor<int8, [1, %d, %d, %d]> qxv = quantize(input = x, output_dtype = q_dtype, scale = x_qs)[name = string(\"qx\")];\n"
            @"        tensor<fp16, [1, %d, %d, %d]> x2 = dequantize(input = qxv, scale = x_dqs)[name = string(\"x2\")];\n",
            ic, sp0, sp1, ic, sp0, sp1];
        xin = @"x2";
    } else if (dx) {
        [m appendFormat:
            @"        tensor<fp16, [1, %d, %d, %d]> x2 = dequantize(input = x, scale = x_dqs)[name = string(\"x2\")];\n",
            ic, sp0, sp1];
        xin = @"x2";
    }
    [m appendFormat:
        @"        tensor<fp16, [1, %d, %d, %d]> gate = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wg, x = %@)[name = string(\"cg\")];\n"
        @"        tensor<fp16, [1, %d, %d, %d]> up = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wu, x = %@)[name = string(\"cu\")];\n"
        @"        tensor<fp16, [1, %d, %d, %d]> sig = sigmoid(x = gate)[name = string(\"sg\")];\n"
        @"        tensor<fp16, [1, %d, %d, %d]> silu = mul(x = gate, y = sig)[name = string(\"si\")];\n"
        @"        tensor<fp16, [1, %d, %d, %d]> hid = mul(x = silu, y = up)[name = string(\"gt\")];\n",
        hidden, sp0, sp1, xin,
        hidden, sp0, sp1, xin,
        hidden, sp0, sp1, hidden, sp0, sp1, hidden, sp0, sp1];
    if (qhid) {
        [m appendFormat:
            @"        tensor<int8, [1, %d, %d, %d]> qhid = quantize(input = hid, output_dtype = q_dtype, scale = h_qs)[name = string(\"qh\")];\n"
            @"        tensor<fp16, [1, %d, %d, %d]> hid2 = dequantize(input = qhid, scale = h_dqs)[name = string(\"dqh\")];\n"
            @"        tensor<fp16, [1, %d, %d, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
            @"pad_type = pt, strides = st, weight = Wd, x = hid2)[name = string(\"cd\")];\n",
            hidden, sp0, sp1, hidden, sp0, sp1, ic, sp0, sp1];
    } else {
        [m appendFormat:
            @"        tensor<fp16, [1, %d, %d, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
            @"pad_type = pt, strides = st, weight = Wd, x = hid)[name = string(\"cd\")];\n",
            ic, sp0, sp1];
    }
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

void ane_prefill_session_pick_tile(int seq, int *sp0, int *sp1) {
    if (!sp0 || !sp1 || seq <= 0) {
        return;
    }
    // Empirically strong for expert SwiGLU W8A8 (probe 2026-07-30).
    if (seq == 512) { *sp0 = 2; *sp1 = 256; return; }
    if (seq == 1024) { *sp0 = 4; *sp1 = 256; return; }
    if (seq == 2048) { *sp0 = 8; *sp1 = 256; return; }
    if (seq == 256) { *sp0 = 2; *sp1 = 128; return; }
    *sp0 = 1;
    *sp1 = seq;
}

// fp16: one [2H,IC] gate∥up conv + slice_by_size + down.
static NSString *genFusedGUSwiGLUMIL(int ic, int hidden, int seq) {
    int h2 = hidden * 2;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendString:
        @"        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"];
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> Wgu = const()[name = string(\"Wgu\"), "
        @"val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wgu.bin\"), offset = uint64(64)))];\n"
        @"        tensor<fp16, [%d, %d, 1, 1]> Wd = const()[name = string(\"Wd\"), "
        @"val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/wd.bin\"), offset = uint64(64)))];\n",
        h2, ic, h2, ic,
        ic, hidden, ic, hidden];
    [m appendFormat:
        @"        tensor<fp16, [1, %d, 1, %d]> gu = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wgu, x = x)[name = string(\"cgu\")];\n"
        @"        tensor<int32, [4]> bg = const()[name = string(\"bg\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [4]> sg = const()[name = string(\"sgz\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> gate = slice_by_size(x = gu, begin = bg, size = sg)[name = string(\"slg\")];\n"
        @"        tensor<int32, [4]> bu = const()[name = string(\"bu\"), val = tensor<int32, [4]>([0, %d, 0, 0])];\n"
        @"        tensor<int32, [4]> su = const()[name = string(\"suz\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> up = slice_by_size(x = gu, begin = bu, size = su)[name = string(\"slu\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = string(\"sg\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = string(\"si\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> hid = mul(x = silu, y = up)[name = string(\"gt\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wd, x = hid)[name = string(\"cd\")];\n"
        @"    } -> (y);\n"
        @"}\n",
        h2, seq,
        hidden, seq, hidden, seq,
        hidden, hidden, seq, hidden, seq,
        hidden, seq, hidden, seq, hidden, seq,
        ic, seq];
    return m;
}

// int8 fused gate∥up (+ optional W8A8 on hid before down).
static NSString *genInt8FusedGUSwiGLUMIL(int ic, int hidden, int seq,
                                         float scaleGU, float scaleD,
                                         bool w8a8_hid, float act_scale) {
    int h2 = hidden * 2;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendString:
        @"        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"];
    if (w8a8_hid) {
        [m appendFormat:
            @"        fp16 q_scale = const()[name = string(\"q_scale\"), val = fp16(%a)];\n"
            @"        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
            @"        fp16 dq_scale = const()[name = string(\"dq_scale\"), val = fp16(%a)];\n",
            (double)(_Float16)act_scale, (double)(_Float16)act_scale];
    }
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> Wgu = constexpr_affine_dequantize()"
        @"[axis = int32(0), name = string(\"Wgu\"), "
        @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        @"(BLOBFILE(path = string(\"@model_path/weights/wgu.bin\"), offset = uint64(64))), "
        @"scale = fp16(%a), zero_point = int8(0)];\n",
        h2, ic, h2, ic, (double)(_Float16)scaleGU];
    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> Wd = constexpr_affine_dequantize()"
        @"[axis = int32(0), name = string(\"Wd\"), "
        @"quantized_data = tensor<int8, [%d, %d, 1, 1]>"
        @"(BLOBFILE(path = string(\"@model_path/weights/wd.bin\"), offset = uint64(64))), "
        @"scale = fp16(%a), zero_point = int8(0)];\n",
        ic, hidden, ic, hidden, (double)(_Float16)scaleD];
    [m appendFormat:
        @"        tensor<fp16, [1, %d, 1, %d]> gu = conv(dilations = dl, groups = gr, pad = pd, "
        @"pad_type = pt, strides = st, weight = Wgu, x = x)[name = string(\"cgu\")];\n"
        @"        tensor<int32, [4]> bg = const()[name = string(\"bg\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [4]> sgz = const()[name = string(\"sgz\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> gate = slice_by_size(x = gu, begin = bg, size = sgz)[name = string(\"slg\")];\n"
        @"        tensor<int32, [4]> bu = const()[name = string(\"bu\"), val = tensor<int32, [4]>([0, %d, 0, 0])];\n"
        @"        tensor<int32, [4]> suz = const()[name = string(\"suz\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> up = slice_by_size(x = gu, begin = bu, size = suz)[name = string(\"slu\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = string(\"sg\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = string(\"si\")];\n"
        @"        tensor<fp16, [1, %d, 1, %d]> hid = mul(x = silu, y = up)[name = string(\"gt\")];\n",
        h2, seq,
        hidden, seq, hidden, seq,
        hidden, hidden, seq, hidden, seq,
        hidden, seq, hidden, seq, hidden, seq];
    if (w8a8_hid) {
        [m appendFormat:
            @"        tensor<int8, [1, %d, 1, %d]> qhid = quantize(input = hid, output_dtype = q_dtype, scale = q_scale)[name = string(\"qh\")];\n"
            @"        tensor<fp16, [1, %d, 1, %d]> hid2 = dequantize(input = qhid, scale = dq_scale)[name = string(\"dqh\")];\n"
            @"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
            @"pad_type = pt, strides = st, weight = Wd, x = hid2)[name = string(\"cd\")];\n",
            hidden, seq, hidden, seq, ic, seq];
    } else {
        [m appendFormat:
            @"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, pad = pd, "
            @"pad_type = pt, strides = st, weight = Wd, x = hid)[name = string(\"cd\")];\n",
            ic, seq];
    }
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static void fillSynth(float *dst, int n, float scale, int seedA, int seedB) {
    for (int i = 0; i < n; i++) {
        dst[i] = scale * (float)((i * seedA + seedB) % 97) / 97.0f;
        if (dst[i] < 0.01f) {
            dst[i] = 0.01f; // stay above fp16 denormal flush
        }
    }
}

ANEPrefillSession *ane_prefill_session_create(int ic, int oc, int seq,
                                              const float *weight_oc_ic) {
    if (ic <= 0 || oc <= 0 || seq <= 0) {
        return NULL;
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSession *s = (ANEPrefillSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->ic = ic;
    s->oc = oc;
    s->hidden = 0;
    s->seq = seq;
    s->swiglu = false;
    s->int8 = false;
    s->int8Scale = 0;
    s->inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
    s->outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);

    float *wICOC = (float *)malloc((size_t)ic * (size_t)oc * sizeof(float));
    if (!wICOC) {
        free(s);
        return NULL;
    }
    if (weight_oc_ic) {
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                wICOC[i * oc + o] = weight_oc_ic[o * ic + i];
            }
        }
    } else {
        for (int i = 0; i < ic * oc; i++) {
            wICOC[i] = 0.05f;
        }
    }

    size_t blobLen = 0;
    uint8_t *blob = ane_bridge_build_weight_blob(wICOC, ic, oc, &blobLen);
    free(wICOC);
    if (!blob) {
        free(s);
        return NULL;
    }

    NSString *mil = genFP16BlobMIL(ic, oc, seq);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(blob);
        free(s);
        return NULL;
    }

    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;
    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile(
        [milData bytes], [milData length],
        blob, blobLen,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(blob);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

ANEPrefillSession *ane_prefill_session_create_int8(int ic, int oc, int seq,
                                                   const float *weight_oc_ic) {
    if (ic <= 0 || oc <= 0 || seq <= 0) {
        return NULL;
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSession *s = (ANEPrefillSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->ic = ic;
    s->oc = oc;
    s->hidden = 0;
    s->seq = seq;
    s->swiglu = false;
    s->int8 = true;
    s->inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
    s->outBytes = (size_t)oc * (size_t)seq * sizeof(_Float16);

    // Conv blob is [OC, IC] row-major (= weight_oc_ic layout).
    float *wOCIC = (float *)malloc((size_t)oc * (size_t)ic * sizeof(float));
    if (!wOCIC) {
        free(s);
        return NULL;
    }
    if (weight_oc_ic) {
        memcpy(wOCIC, weight_oc_ic, (size_t)oc * (size_t)ic * sizeof(float));
    } else {
        fillSynth(wOCIC, oc * ic, 0.08f, 17, 3);
    }

    size_t blobLen = 0;
    float scale = 0;
    uint8_t *blob = ane_bridge_build_weight_blob_quantized(wOCIC, oc, ic, &scale, &blobLen);
    free(wOCIC);
    if (!blob) {
        free(s);
        return NULL;
    }
    s->int8Scale = scale;

    NSString *mil = genInt8ConvMIL(ic, oc, seq, scale);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(blob);
        free(s);
        return NULL;
    }

    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;
    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile(
        [milData bytes], [milData length],
        blob, blobLen,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(blob);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

ANEPrefillSession *ane_prefill_session_create_swiglu(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden) {
    if (ic <= 0 || hidden <= 0 || seq <= 0) {
        return NULL;
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSession *s = (ANEPrefillSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->ic = ic;
    s->oc = ic; // output restores embed width
    s->hidden = hidden;
    s->seq = seq;
    s->swiglu = true;
    s->inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
    s->outBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);

    size_t nGate = (size_t)hidden * (size_t)ic;
    size_t nDown = (size_t)ic * (size_t)hidden;
    float *Wg = (float *)malloc(nGate * sizeof(float));
    float *Wu = (float *)malloc(nGate * sizeof(float));
    float *Wd = (float *)malloc(nDown * sizeof(float));
    if (!Wg || !Wu || !Wd) {
        free(Wg); free(Wu); free(Wd); free(s);
        return NULL;
    }
    if (Wg_hidden_ic) {
        memcpy(Wg, Wg_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wg, (int)nGate, 0.08f, 17, 3);
    }
    if (Wu_hidden_ic) {
        memcpy(Wu, Wu_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wu, (int)nGate, 0.08f, 13, 7);
    }
    if (Wd_ic_hidden) {
        memcpy(Wd, Wd_ic_hidden, nDown * sizeof(float));
    } else {
        fillSynth(Wd, (int)nDown, 0.08f, 19, 11);
    }

    size_t lenG = 0, lenU = 0, lenD = 0;
    uint8_t *blobG = ane_bridge_build_weight_blob(Wg, hidden, ic, &lenG);
    uint8_t *blobU = ane_bridge_build_weight_blob(Wu, hidden, ic, &lenU);
    uint8_t *blobD = ane_bridge_build_weight_blob(Wd, ic, hidden, &lenD);
    free(Wg); free(Wu); free(Wd);
    if (!blobG || !blobU || !blobD) {
        free(blobG); free(blobU); free(blobD); free(s);
        return NULL;
    }

    NSString *mil = genSwiGLUConvMIL(ic, hidden, seq);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(blobG); free(blobU); free(blobD); free(s);
        return NULL;
    }

    const char *names[3] = {
        "@model_path/weights/wg.bin",
        "@model_path/weights/wu.bin",
        "@model_path/weights/wd.bin",
    };
    const uint8_t *datas[3] = { blobG, blobU, blobD };
    size_t lens[3] = { lenG, lenU, lenD };
    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;

    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile_multi_weights(
        [milData bytes], [milData length],
        names, datas, lens, 3,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(blobG); free(blobU); free(blobD);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

static void packGateUp(float *Wgu, const float *Wg, const float *Wu, int hidden, int ic) {
    size_t row = (size_t)ic;
    memcpy(Wgu, Wg, (size_t)hidden * row * sizeof(float));
    memcpy(Wgu + (size_t)hidden * row, Wu, (size_t)hidden * row * sizeof(float));
}

ANEPrefillSession *ane_prefill_session_create_swiglu_int8(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden) {
    if (ic <= 0 || hidden <= 0 || seq <= 0) {
        return NULL;
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSession *s = (ANEPrefillSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->ic = ic;
    s->oc = ic;
    s->hidden = hidden;
    s->seq = seq;
    s->swiglu = true;
    s->int8 = true;
    s->inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
    s->outBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);

    size_t nGate = (size_t)hidden * (size_t)ic;
    size_t nDown = (size_t)ic * (size_t)hidden;
    float *Wg = (float *)malloc(nGate * sizeof(float));
    float *Wu = (float *)malloc(nGate * sizeof(float));
    float *Wd = (float *)malloc(nDown * sizeof(float));
    if (!Wg || !Wu || !Wd) {
        free(Wg); free(Wu); free(Wd); free(s);
        return NULL;
    }
    if (Wg_hidden_ic) {
        memcpy(Wg, Wg_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wg, (int)nGate, 0.08f, 17, 3);
    }
    if (Wu_hidden_ic) {
        memcpy(Wu, Wu_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wu, (int)nGate, 0.08f, 13, 7);
    }
    if (Wd_ic_hidden) {
        memcpy(Wd, Wd_ic_hidden, nDown * sizeof(float));
    } else {
        fillSynth(Wd, (int)nDown, 0.08f, 19, 11);
    }

    float scaleG = 0, scaleU = 0, scaleD = 0;
    size_t lenG = 0, lenU = 0, lenD = 0;
    uint8_t *blobG = ane_bridge_build_weight_blob_quantized(Wg, hidden, ic, &scaleG, &lenG);
    uint8_t *blobU = ane_bridge_build_weight_blob_quantized(Wu, hidden, ic, &scaleU, &lenU);
    uint8_t *blobD = ane_bridge_build_weight_blob_quantized(Wd, ic, hidden, &scaleD, &lenD);
    free(Wg); free(Wu); free(Wd);
    if (!blobG || !blobU || !blobD) {
        free(blobG); free(blobU); free(blobD); free(s);
        return NULL;
    }
    s->int8Scale = scaleG;

    NSString *mil = genInt8SwiGLUMIL(ic, hidden, 1, seq, scaleG, scaleU, scaleD, 0, 0, false);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(blobG); free(blobU); free(blobD); free(s);
        return NULL;
    }

    const char *names[3] = {
        "@model_path/weights/wg.bin",
        "@model_path/weights/wu.bin",
        "@model_path/weights/wd.bin",
    };
    const uint8_t *datas[3] = { blobG, blobU, blobD };
    size_t lens[3] = { lenG, lenU, lenD };
    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;

    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile_multi_weights(
        [milData bytes], [milData length],
        names, datas, lens, 3,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(blobG); free(blobU); free(blobD);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

ANEPrefillSession *ane_prefill_session_create_swiglu_int8_w8a8(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    float hid_scale,
    float x_scale,
    int sp0, int sp1,
    bool int8_input) {
    if (ic <= 0 || hidden <= 0 || seq <= 0) {
        return NULL;
    }
    if (sp0 <= 0 || sp1 <= 0 || sp0 * sp1 != seq) {
        return NULL;
    }
    if (!(hid_scale > 0) && !(x_scale > 0)) {
        return NULL; // use create_swiglu_int8 for weight-only int8
    }
    if (int8_input && !(x_scale > 0)) {
        return NULL; // int8 input requires dequant scale
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSession *s = (ANEPrefillSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->ic = ic;
    s->oc = ic;
    s->hidden = hidden;
    s->seq = seq;
    s->swiglu = true;
    s->int8 = true;
    s->int8Input = int8_input;
    s->inBytes = (size_t)ic * (size_t)seq * (int8_input ? sizeof(int8_t) : sizeof(_Float16));
    s->outBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);

    size_t nGate = (size_t)hidden * (size_t)ic;
    size_t nDown = (size_t)ic * (size_t)hidden;
    float *Wg = (float *)malloc(nGate * sizeof(float));
    float *Wu = (float *)malloc(nGate * sizeof(float));
    float *Wd = (float *)malloc(nDown * sizeof(float));
    if (!Wg || !Wu || !Wd) {
        free(Wg); free(Wu); free(Wd); free(s);
        return NULL;
    }
    if (Wg_hidden_ic) {
        memcpy(Wg, Wg_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wg, (int)nGate, 0.08f, 17, 3);
    }
    if (Wu_hidden_ic) {
        memcpy(Wu, Wu_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wu, (int)nGate, 0.08f, 13, 7);
    }
    if (Wd_ic_hidden) {
        memcpy(Wd, Wd_ic_hidden, nDown * sizeof(float));
    } else {
        fillSynth(Wd, (int)nDown, 0.08f, 19, 11);
    }

    float scaleG = 0, scaleU = 0, scaleD = 0;
    size_t lenG = 0, lenU = 0, lenD = 0;
    uint8_t *blobG = ane_bridge_build_weight_blob_quantized(Wg, hidden, ic, &scaleG, &lenG);
    uint8_t *blobU = ane_bridge_build_weight_blob_quantized(Wu, hidden, ic, &scaleU, &lenU);
    uint8_t *blobD = ane_bridge_build_weight_blob_quantized(Wd, ic, hidden, &scaleD, &lenD);
    free(Wg); free(Wu); free(Wd);
    if (!blobG || !blobU || !blobD) {
        free(blobG); free(blobU); free(blobD); free(s);
        return NULL;
    }
    s->int8Scale = scaleG;

    NSString *mil = genInt8SwiGLUMIL(ic, hidden, sp0, sp1, scaleG, scaleU, scaleD,
                                     hid_scale, x_scale, int8_input);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(blobG); free(blobU); free(blobD); free(s);
        return NULL;
    }

    const char *names[3] = {
        "@model_path/weights/wg.bin",
        "@model_path/weights/wu.bin",
        "@model_path/weights/wd.bin",
    };
    const uint8_t *datas[3] = { blobG, blobU, blobD };
    size_t lens[3] = { lenG, lenU, lenD };
    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;

    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile_multi_weights(
        [milData bytes], [milData length],
        names, datas, lens, 3,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(blobG); free(blobU); free(blobD);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

ANEPrefillSession *ane_prefill_session_create_swiglu_fused_gu(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden) {
    if (ic <= 0 || hidden <= 0 || seq <= 0) {
        return NULL;
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSession *s = (ANEPrefillSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->ic = ic;
    s->oc = ic;
    s->hidden = hidden;
    s->seq = seq;
    s->swiglu = true;
    s->int8 = false;
    s->inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
    s->outBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);

    size_t nGate = (size_t)hidden * (size_t)ic;
    size_t nDown = (size_t)ic * (size_t)hidden;
    float *Wg = (float *)malloc(nGate * sizeof(float));
    float *Wu = (float *)malloc(nGate * sizeof(float));
    float *Wd = (float *)malloc(nDown * sizeof(float));
    float *Wgu = (float *)malloc(2 * nGate * sizeof(float));
    if (!Wg || !Wu || !Wd || !Wgu) {
        free(Wg); free(Wu); free(Wd); free(Wgu); free(s);
        return NULL;
    }
    if (Wg_hidden_ic) {
        memcpy(Wg, Wg_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wg, (int)nGate, 0.08f, 17, 3);
    }
    if (Wu_hidden_ic) {
        memcpy(Wu, Wu_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wu, (int)nGate, 0.08f, 13, 7);
    }
    if (Wd_ic_hidden) {
        memcpy(Wd, Wd_ic_hidden, nDown * sizeof(float));
    } else {
        fillSynth(Wd, (int)nDown, 0.08f, 19, 11);
    }
    packGateUp(Wgu, Wg, Wu, hidden, ic);
    free(Wg); free(Wu);

    size_t lenGU = 0, lenD = 0;
    uint8_t *blobGU = ane_bridge_build_weight_blob(Wgu, hidden * 2, ic, &lenGU);
    uint8_t *blobD = ane_bridge_build_weight_blob(Wd, ic, hidden, &lenD);
    free(Wgu); free(Wd);
    if (!blobGU || !blobD) {
        free(blobGU); free(blobD); free(s);
        return NULL;
    }

    NSString *mil = genFusedGUSwiGLUMIL(ic, hidden, seq);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(blobGU); free(blobD); free(s);
        return NULL;
    }

    const char *names[2] = {
        "@model_path/weights/wgu.bin",
        "@model_path/weights/wd.bin",
    };
    const uint8_t *datas[2] = { blobGU, blobD };
    size_t lens[2] = { lenGU, lenD };
    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;

    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile_multi_weights(
        [milData bytes], [milData length],
        names, datas, lens, 2,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(blobGU); free(blobD);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

ANEPrefillSession *ane_prefill_session_create_swiglu_int8_fused(
    int ic, int hidden, int seq,
    const float *Wg_hidden_ic,
    const float *Wu_hidden_ic,
    const float *Wd_ic_hidden,
    bool w8a8_hid,
    float act_scale) {
    if (ic <= 0 || hidden <= 0 || seq <= 0) {
        return NULL;
    }
    if (w8a8_hid && !(act_scale > 0)) {
        return NULL;
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSession *s = (ANEPrefillSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->ic = ic;
    s->oc = ic;
    s->hidden = hidden;
    s->seq = seq;
    s->swiglu = true;
    s->int8 = true;
    s->inBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);
    s->outBytes = (size_t)ic * (size_t)seq * sizeof(_Float16);

    size_t nGate = (size_t)hidden * (size_t)ic;
    size_t nDown = (size_t)ic * (size_t)hidden;
    float *Wg = (float *)malloc(nGate * sizeof(float));
    float *Wu = (float *)malloc(nGate * sizeof(float));
    float *Wd = (float *)malloc(nDown * sizeof(float));
    float *Wgu = (float *)malloc(2 * nGate * sizeof(float));
    if (!Wg || !Wu || !Wd || !Wgu) {
        free(Wg); free(Wu); free(Wd); free(Wgu); free(s);
        return NULL;
    }
    if (Wg_hidden_ic) {
        memcpy(Wg, Wg_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wg, (int)nGate, 0.08f, 17, 3);
    }
    if (Wu_hidden_ic) {
        memcpy(Wu, Wu_hidden_ic, nGate * sizeof(float));
    } else {
        fillSynth(Wu, (int)nGate, 0.08f, 13, 7);
    }
    if (Wd_ic_hidden) {
        memcpy(Wd, Wd_ic_hidden, nDown * sizeof(float));
    } else {
        fillSynth(Wd, (int)nDown, 0.08f, 19, 11);
    }
    packGateUp(Wgu, Wg, Wu, hidden, ic);
    free(Wg); free(Wu);

    float scaleGU = 0, scaleD = 0;
    size_t lenGU = 0, lenD = 0;
    uint8_t *blobGU = ane_bridge_build_weight_blob_quantized(Wgu, hidden * 2, ic, &scaleGU, &lenGU);
    uint8_t *blobD = ane_bridge_build_weight_blob_quantized(Wd, ic, hidden, &scaleD, &lenD);
    free(Wgu); free(Wd);
    if (!blobGU || !blobD) {
        free(blobGU); free(blobD); free(s);
        return NULL;
    }
    s->int8Scale = scaleGU;

    NSString *mil = genInt8FusedGUSwiGLUMIL(ic, hidden, seq, scaleGU, scaleD, w8a8_hid, act_scale);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(blobGU); free(blobD); free(s);
        return NULL;
    }

    const char *names[2] = {
        "@model_path/weights/wgu.bin",
        "@model_path/weights/wd.bin",
    };
    const uint8_t *datas[2] = { blobGU, blobD };
    size_t lens[2] = { lenGU, lenD };
    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;

    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile_multi_weights(
        [milData bytes], [milData length],
        names, datas, lens, 2,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(blobGU); free(blobD);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

void ane_prefill_session_destroy(ANEPrefillSession *s) {
    if (!s) {
        return;
    }
    if (s->kernel) {
        ane_bridge_free(s->kernel);
    }
    free(s);
}

bool ane_prefill_session_ready(const ANEPrefillSession *s) {
    return s && s->kernel;
}

int ane_prefill_session_ic(const ANEPrefillSession *s) { return s ? s->ic : 0; }
int ane_prefill_session_oc(const ANEPrefillSession *s) { return s ? s->oc : 0; }
int ane_prefill_session_hidden(const ANEPrefillSession *s) { return s ? s->hidden : 0; }
int ane_prefill_session_seq(const ANEPrefillSession *s) { return s ? s->seq : 0; }
bool ane_prefill_session_is_swiglu(const ANEPrefillSession *s) { return s && s->swiglu; }
bool ane_prefill_session_is_int8(const ANEPrefillSession *s) { return s && s->int8; }
bool ane_prefill_session_is_int8_input(const ANEPrefillSession *s) {
    return s && s->int8Input;
}
float ane_prefill_session_int8_scale(const ANEPrefillSession *s) {
    return s ? s->int8Scale : 0;
}

uint32_t ane_prefill_session_input_surface_id(const ANEPrefillSession *s) {
    if (!s || !s->kernel) {
        return 0;
    }
    return ane_bridge_input_surface_id(s->kernel, 0);
}

uint32_t ane_prefill_session_output_surface_id(const ANEPrefillSession *s) {
    if (!s || !s->kernel) {
        return 0;
    }
    return ane_bridge_output_surface_id(s->kernel, 0);
}

size_t ane_prefill_session_input_bytes(const ANEPrefillSession *s) {
    return s ? s->inBytes : 0;
}

size_t ane_prefill_session_output_bytes(const ANEPrefillSession *s) {
    return s ? s->outBytes : 0;
}

bool ane_prefill_session_write_acts_fp16(ANEPrefillSession *s,
                                         const void *fp16_ic_seq,
                                         size_t bytes) {
    if (!s || !s->kernel || !fp16_ic_seq || s->int8Input || bytes != s->inBytes) {
        return false;
    }
    ane_bridge_write_input(s->kernel, 0, fp16_ic_seq, bytes);
    return true;
}

bool ane_prefill_session_write_acts_int8(ANEPrefillSession *s,
                                         const void *int8_ic_seq,
                                         size_t bytes) {
    if (!s || !s->kernel || !int8_ic_seq || !s->int8Input || bytes != s->inBytes) {
        return false;
    }
    ane_bridge_write_input(s->kernel, 0, int8_ic_seq, bytes);
    return true;
}

bool ane_prefill_session_eval(ANEPrefillSession *s) {
    if (!s || !s->kernel) {
        return false;
    }
    if (!ane_bridge_eval(s->kernel)) {
        return false;
    }
    s->evalCount++;
    return true;
}

bool ane_prefill_session_read_out_fp16(ANEPrefillSession *s, void *dst, size_t bytes) {
    if (!s || !s->kernel || !dst || bytes != s->outBytes) {
        return false;
    }
    ane_bridge_read_output(s->kernel, 0, dst, bytes);
    return true;
}

bool ane_prefill_session_read_out_f32(ANEPrefillSession *s, float *dst, size_t n) {
    if (!s || !s->kernel || !dst || n == 0) {
        return false;
    }
    if (s->outBytes != n * sizeof(_Float16)) {
        return false;
    }
    uint32_t sid = ane_bridge_output_surface_id(s->kernel, 0);
    if (sid == 0) {
        return false;
    }
    IOSurfaceRef surf = IOSurfaceLookup(sid);
    if (!surf) {
        return false;
    }
    IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *src = (const _Float16 *)IOSurfaceGetBaseAddress(surf);
    for (size_t i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
    IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
    CFRelease(surf);
    return true;
}

double ane_prefill_session_compile_ms(const ANEPrefillSession *s) {
    return s ? s->compileMs : 0;
}

int ane_prefill_session_eval_count(const ANEPrefillSession *s) {
    return s ? s->evalCount : 0;
}
