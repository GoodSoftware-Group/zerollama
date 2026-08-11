// Metal ggml↔ANE channel layout (lab). See ane_ffn_layout_metal.h.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "ane_ffn_layout_metal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> unpackF16;
    id<MTLComputePipelineState> packI8F32;
    id<MTLComputePipelineState> packI8F16;
    bool ready;
} LayoutMetalState;

static LayoutMetalState g_lm = {0};

static NSString *kShaderSrc = @""
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void channel_f16_to_ggml_f16(\n"
    "    device half *src [[buffer(0)]],\n"
    "    device half *dst [[buffer(1)]],\n"
    "    constant uint &oc [[buffer(2)]],\n"
    "    constant uint &seq [[buffer(3)]],\n"
    "    uint2 gid [[thread_position_in_grid]]) {\n"
    "    uint t = gid.x;\n"
    "    uint o = gid.y;\n"
    "    if (t >= seq || o >= oc) return;\n"
    "    dst[t * oc + o] = src[o * seq + t];\n"
    "}\n"
    "kernel void ggml_f32_to_channel_i8(\n"
    "    device float *src [[buffer(0)]],\n"
    "    device char *dst [[buffer(1)]],\n"
    "    constant float &inv_scale [[buffer(2)]],\n"
    "    constant uint &ic [[buffer(3)]],\n"
    "    constant uint &seq [[buffer(4)]],\n"
    "    uint2 gid [[thread_position_in_grid]]) {\n"
    "    uint t = gid.x;\n"
    "    uint i = gid.y;\n"
    "    if (t >= seq || i >= ic) return;\n"
    "    float v = src[t * ic + i] * inv_scale;\n"
    "    v = clamp(v, -128.0f, 127.0f);\n"
    "    int q = (int)floor(v + (v >= 0.0f ? 0.5f : -0.5f));\n"
    "    dst[i * seq + t] = (char)q;\n"
    "}\n"
    "kernel void ggml_f16_to_channel_i8(\n"
    "    device half *src [[buffer(0)]],\n"
    "    device char *dst [[buffer(1)]],\n"
    "    constant float &inv_scale [[buffer(2)]],\n"
    "    constant uint &ic [[buffer(3)]],\n"
    "    constant uint &seq [[buffer(4)]],\n"
    "    uint2 gid [[thread_position_in_grid]]) {\n"
    "    uint t = gid.x;\n"
    "    uint i = gid.y;\n"
    "    if (t >= seq || i >= ic) return;\n"
    "    float v = (float)src[t * ic + i] * inv_scale;\n"
    "    v = clamp(v, -128.0f, 127.0f);\n"
    "    int q = (int)floor(v + (v >= 0.0f ? 0.5f : -0.5f));\n"
    "    dst[i * seq + t] = (char)q;\n"
    "}\n";

static id<MTLComputePipelineState> make_pipe(id<MTLDevice> device, NSString *name) {
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kShaderSrc options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "ane_ffn_layout_metal: library: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [device newComputePipelineStateWithFunction:fn error:&err];
}

bool ane_ffn_layout_metal_ready(void) {
    if (g_lm.ready) return true;
    g_lm.device = MTLCreateSystemDefaultDevice();
    if (!g_lm.device) return false;
    g_lm.queue = [g_lm.device newCommandQueue];
    if (!g_lm.queue) return false;
    g_lm.unpackF16 = make_pipe(g_lm.device, @"channel_f16_to_ggml_f16");
    g_lm.packI8F32 = make_pipe(g_lm.device, @"ggml_f32_to_channel_i8");
    g_lm.packI8F16 = make_pipe(g_lm.device, @"ggml_f16_to_channel_i8");
    if (!g_lm.unpackF16 || !g_lm.packI8F32 || !g_lm.packI8F16) {
        return false;
    }
    g_lm.ready = true;
    return true;
}

static id<MTLBuffer> map_surface_buf(uint32_t sid, size_t bytes, IOSurfaceRef *outSurf) {
    IOSurfaceRef surf = IOSurfaceLookup(sid);
    if (!surf) return nil;
    IOSurfaceLock(surf, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surf);
    if (!base) {
        IOSurfaceUnlock(surf, 0, NULL);
        CFRelease(surf);
        return nil;
    }
    id<MTLBuffer> buf = [g_lm.device newBufferWithBytesNoCopy:base
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!buf) {
        IOSurfaceUnlock(surf, 0, NULL);
        CFRelease(surf);
        return nil;
    }
    *outSurf = surf;
    return buf;
}

static void release_surface_buf(IOSurfaceRef surf) {
    if (!surf) return;
    IOSurfaceUnlock(surf, 0, NULL);
    CFRelease(surf);
}

bool ane_ffn_layout_metal_unpack_out_f16(
    uint32_t out_surface_id,
    int oc, int seq,
    void *dst_ggml_f16) {
    if (!ane_ffn_layout_metal_ready() || !dst_ggml_f16 || oc <= 0 || seq <= 0) {
        return false;
    }
    size_t bytes = (size_t)oc * (size_t)seq * sizeof(uint16_t);
    IOSurfaceRef surf = NULL;
    id<MTLBuffer> src = map_surface_buf(out_surface_id, bytes, &surf);
    if (!src) return false;

    id<MTLBuffer> dst = [g_lm.device newBufferWithBytesNoCopy:dst_ggml_f16
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    // Fallback if dst not page-aligned for no-copy.
    bool copied = false;
    if (!dst) {
        dst = [g_lm.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        copied = true;
    }
    if (!dst) {
        release_surface_buf(surf);
        return false;
    }

    uint32_t oc_u = (uint32_t)oc;
    uint32_t seq_u = (uint32_t)seq;
    id<MTLCommandBuffer> cmd = [g_lm.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_lm.unpackF16];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&oc_u length:sizeof(oc_u) atIndex:2];
    [enc setBytes:&seq_u length:sizeof(seq_u) atIndex:3];
    NSUInteger tw = MIN((NSUInteger)16, (NSUInteger)seq);
    NSUInteger th = MIN((NSUInteger)16, (NSUInteger)oc);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)seq, (NSUInteger)oc, 1)
   threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    bool ok = cmd.status == MTLCommandBufferStatusCompleted;
    if (ok && copied) {
        memcpy(dst_ggml_f16, [dst contents], bytes);
    }
    release_surface_buf(surf);
    return ok;
}

static bool pack_i8_common(
    uint32_t in_surface_id,
    id<MTLBuffer> srcBuf,
    id<MTLComputePipelineState> pipe,
    int ic, int seq,
    float scale) {
    if (!ane_ffn_layout_metal_ready() || !srcBuf || ic <= 0 || seq <= 0 || !(scale > 0)) {
        return false;
    }
    size_t bytes = (size_t)ic * (size_t)seq; // int8
    IOSurfaceRef surf = NULL;
    id<MTLBuffer> dst = map_surface_buf(in_surface_id, bytes, &surf);
    if (!dst) return false;

    float inv = 1.0f / scale;
    uint32_t ic_u = (uint32_t)ic;
    uint32_t seq_u = (uint32_t)seq;
    id<MTLCommandBuffer> cmd = [g_lm.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:srcBuf offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&inv length:sizeof(inv) atIndex:2];
    [enc setBytes:&ic_u length:sizeof(ic_u) atIndex:3];
    [enc setBytes:&seq_u length:sizeof(seq_u) atIndex:4];
    NSUInteger tw = MIN((NSUInteger)16, (NSUInteger)seq);
    NSUInteger th = MIN((NSUInteger)16, (NSUInteger)ic);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)seq, (NSUInteger)ic, 1)
   threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    bool ok = cmd.status == MTLCommandBufferStatusCompleted;
    release_surface_buf(surf);
    return ok;
}

bool ane_ffn_layout_metal_pack_in_i8_f32(
    uint32_t in_surface_id,
    const float *src_ggml_f32,
    int ic, int seq,
    float scale) {
    if (!src_ggml_f32 || !ane_ffn_layout_metal_ready()) return false;
    size_t bytes = (size_t)ic * (size_t)seq * sizeof(float);
    id<MTLBuffer> src = [g_lm.device newBufferWithBytesNoCopy:(void *)src_ggml_f32
                                                        length:bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
    if (!src) {
        src = [g_lm.device newBufferWithBytes:src_ggml_f32
                                       length:bytes
                                      options:MTLResourceStorageModeShared];
    }
    if (!src) return false;
    return pack_i8_common(in_surface_id, src, g_lm.packI8F32, ic, seq, scale);
}

bool ane_ffn_layout_metal_pack_in_i8_f16(
    uint32_t in_surface_id,
    const void *src_ggml_f16,
    int ic, int seq,
    float scale) {
    if (!src_ggml_f16) return false;
    if (!ane_ffn_layout_metal_ready()) return false;
    size_t bytes = (size_t)ic * (size_t)seq * sizeof(uint16_t);
    id<MTLBuffer> src = [g_lm.device newBufferWithBytesNoCopy:(void *)src_ggml_f16
                                                        length:bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
    if (!src) {
        src = [g_lm.device newBufferWithBytes:src_ggml_f16
                                       length:bytes
                                      options:MTLResourceStorageModeShared];
    }
    if (!src) return false;
    return pack_i8_common(in_surface_id, src, g_lm.packI8F16, ic, seq, scale);
}
