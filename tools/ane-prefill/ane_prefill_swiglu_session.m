// In-process ANE SwiGLU session — gate+up+silu*+down, multi BLOBFILE weights.
// Shared act transpose for gate/up (ANE rejects reshaping the same input twice).
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

#include "ane_prefill_swiglu_session.h"
#include "ane_bridge.h"

#include <stdlib.h>
#include <string.h>

struct ANEPrefillSwiGLUSession {
    int dim;
    int hidden;
    int seq;
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

static NSString *genSwiGLUMIL(int dim, int hidden, int seq) {
    NSMutableString *m = [NSMutableString string];
    // Double braces in buildInfo are required by ANE MIL parser.
    [m appendString:@"program(1.3)\n"];
    [m appendString:@"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> Wg = const()[name=string(\"Wg\"), "
        @"val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/wg.bin\"), offset=uint64(64)))];\n",
        dim, hidden, dim, hidden];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> Wu = const()[name=string(\"Wu\"), "
        @"val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/wu.bin\"), offset=uint64(64)))];\n",
        dim, hidden, dim, hidden];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> Wd = const()[name=string(\"Wd\"), "
        @"val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/wd.bin\"), offset=uint64(64)))];\n",
        hidden, dim, hidden, dim];
    // Shared x→xt (do not reshape x twice — ANE name collision).
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n",
        dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=x)[name=string(\"a2\")];\n",
        dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xt = transpose(perm=pm,x=a2)[name=string(\"xt\")];\n",
        seq, dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> gm = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wg)[name=string(\"gm\")];\n",
        seq, hidden];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> um = matmul(transpose_x=bF,transpose_y=bF,x=xt,y=Wu)[name=string(\"um\")];\n",
        seq, hidden];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> gt = transpose(perm=pm,x=gm)[name=string(\"gt\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> ut = transpose(perm=pm,x=um)[name=string(\"ut\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> hg = reshape(shape=rh,x=gt)[name=string(\"hg\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> hu = reshape(shape=rh,x=ut)[name=string(\"hu\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sg = sigmoid(x=hg)[name=string(\"sg\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> si = mul(x=hg,y=sg)[name=string(\"si\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sw = mul(x=si,y=hu)[name=string(\"sw\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<int32, [4]> rg = const()[name=string(\"rg\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> s2 = reshape(shape=rg,x=sw)[name=string(\"s2\")];\n",
        hidden, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> st = transpose(perm=pm,x=s2)[name=string(\"st\")];\n",
        seq, hidden];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> ym = matmul(transpose_x=bF,transpose_y=bF,x=st,y=Wd)[name=string(\"ym\")];\n",
        seq, dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=ym)[name=string(\"yt\")];\n",
        dim, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n",
        dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n",
        dim, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static float *copyOrFill(const float *src, int n, float fill) {
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!out) {
        return NULL;
    }
    if (src) {
        memcpy(out, src, (size_t)n * sizeof(float));
    } else {
        for (int i = 0; i < n; i++) {
            out[i] = fill;
        }
    }
    return out;
}

ANEPrefillSwiGLUSession *ane_prefill_swiglu_session_create(
    int dim, int hidden, int seq,
    const float *Wg_dim_hidden,
    const float *Wu_dim_hidden,
    const float *Wd_hidden_dim) {
    if (dim <= 0 || hidden <= 0 || seq <= 0) {
        return NULL;
    }
    if (ane_bridge_init() != 0) {
        return NULL;
    }

    ANEPrefillSwiGLUSession *s = (ANEPrefillSwiGLUSession *)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->dim = dim;
    s->hidden = hidden;
    s->seq = seq;
    s->inBytes = (size_t)dim * (size_t)seq * sizeof(_Float16);
    s->outBytes = (size_t)dim * (size_t)seq * sizeof(_Float16);

    float *Wg = copyOrFill(Wg_dim_hidden, dim * hidden, 0.05f);
    float *Wu = copyOrFill(Wu_dim_hidden, dim * hidden, 0.05f);
    float *Wd = copyOrFill(Wd_hidden_dim, hidden * dim, 0.05f);
    if (!Wg || !Wu || !Wd) {
        free(Wg); free(Wu); free(Wd); free(s);
        return NULL;
    }

    size_t lg = 0, lu = 0, ld = 0;
    uint8_t *bg = ane_bridge_build_weight_blob(Wg, dim, hidden, &lg);
    uint8_t *bu = ane_bridge_build_weight_blob(Wu, dim, hidden, &lu);
    uint8_t *bd = ane_bridge_build_weight_blob(Wd, hidden, dim, &ld);
    free(Wg); free(Wu); free(Wd);
    if (!bg || !bu || !bd) {
        free(bg); free(bu); free(bd); free(s);
        return NULL;
    }

    NSString *mil = genSwiGLUMIL(dim, hidden, seq);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        free(bg); free(bu); free(bd); free(s);
        return NULL;
    }

    const char *names[] = {
        "@model_path/weights/wg.bin",
        "@model_path/weights/wu.bin",
        "@model_path/weights/wd.bin",
    };
    const uint8_t *datas[] = {bg, bu, bd};
    size_t lens[] = {lg, lu, ld};
    size_t inBytes = s->inBytes;
    size_t outBytes = s->outBytes;

    uint64_t t0 = mach_absolute_time();
    s->kernel = ane_bridge_compile_multi_weights(
        [milData bytes], [milData length],
        names, datas, lens, 3,
        1, &inBytes, 1, &outBytes);
    s->compileMs = ticksToMs(mach_absolute_time() - t0);
    free(bg); free(bu); free(bd);

    if (!s->kernel) {
        free(s);
        return NULL;
    }
    return s;
}

void ane_prefill_swiglu_session_destroy(ANEPrefillSwiGLUSession *s) {
    if (!s) {
        return;
    }
    if (s->kernel) {
        ane_bridge_free(s->kernel);
    }
    free(s);
}

bool ane_prefill_swiglu_session_ready(const ANEPrefillSwiGLUSession *s) {
    return s && s->kernel;
}

int ane_prefill_swiglu_session_dim(const ANEPrefillSwiGLUSession *s) {
    return s ? s->dim : 0;
}
int ane_prefill_swiglu_session_hidden(const ANEPrefillSwiGLUSession *s) {
    return s ? s->hidden : 0;
}
int ane_prefill_swiglu_session_seq(const ANEPrefillSwiGLUSession *s) {
    return s ? s->seq : 0;
}

uint32_t ane_prefill_swiglu_session_input_surface_id(const ANEPrefillSwiGLUSession *s) {
    if (!s || !s->kernel) {
        return 0;
    }
    return ane_bridge_input_surface_id(s->kernel, 0);
}

size_t ane_prefill_swiglu_session_input_bytes(const ANEPrefillSwiGLUSession *s) {
    return s ? s->inBytes : 0;
}

size_t ane_prefill_swiglu_session_output_bytes(const ANEPrefillSwiGLUSession *s) {
    return s ? s->outBytes : 0;
}

bool ane_prefill_swiglu_session_write_acts_fp16(ANEPrefillSwiGLUSession *s,
                                                const void *fp16_dim_seq,
                                                size_t bytes) {
    if (!s || !s->kernel || !fp16_dim_seq || bytes != s->inBytes) {
        return false;
    }
    ane_bridge_write_input(s->kernel, 0, fp16_dim_seq, bytes);
    return true;
}

bool ane_prefill_swiglu_session_eval(ANEPrefillSwiGLUSession *s) {
    if (!s || !s->kernel) {
        return false;
    }
    if (!ane_bridge_eval(s->kernel)) {
        return false;
    }
    s->evalCount++;
    return true;
}

bool ane_prefill_swiglu_session_read_out_fp16(ANEPrefillSwiGLUSession *s,
                                              void *dst, size_t bytes) {
    if (!s || !s->kernel || !dst || bytes != s->outBytes) {
        return false;
    }
    ane_bridge_read_output(s->kernel, 0, dst, bytes);
    return true;
}

double ane_prefill_swiglu_session_compile_ms(const ANEPrefillSwiGLUSession *s) {
    return s ? s->compileMs : 0;
}

int ane_prefill_swiglu_session_eval_count(const ANEPrefillSwiGLUSession *s) {
    return s ? s->evalCount : 0;
}
