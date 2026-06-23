// Shared conv-stack MIL for ANE peak/matmul-style benchmarks (maderix pattern).
#pragma once

#import <Foundation/Foundation.h>

static NSString *aneGenConvStackMIL(int ch, int sp, int depth) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        @"{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ch, sp];
    [m appendString:
        @"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"
        @"            string x_to_fp16_dtype_0 = const()[name = string(\"x_to_fp16_dtype_0\"), val = string(\"fp16\")];\n"];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, 1, %d]> x_to_fp16 = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string(\"cast_in\")];\n",
        ch, sp];

    NSUInteger chunkSize = 64 + (NSUInteger)ch * (NSUInteger)ch * 2;
    NSString *prev = @"x_to_fp16";
    for (int i = 0; i < depth; i++) {
        [m appendFormat:
            @"            tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name = string(\"W%d\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n",
            ch, ch, i, i, ch, ch, (unsigned long)(64 + (NSUInteger)i * chunkSize)];
        NSString *out = [NSString stringWithFormat:@"c%d", i];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, 1, %d]> %@ = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W%d, x = %@)[name = string(\"%@\")];\n",
            ch, sp, out, i, prev, out];
        prev = out;
    }
    [m appendString:@"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:
        @"            tensor<fp32, [1, %d, 1, %d]> c = cast(dtype = to_fp32, x = %@)[name = string(\"cast_out\")];\n",
        ch, sp, prev];
    [m appendString:@"        } -> (c);\n}\n"];
    return m;
}

static NSData *aneBuildConvStackWeightBlob(int ch, int depth) {
    NSUInteger wsize = (NSUInteger)ch * (NSUInteger)ch * 2;
    NSUInteger chunkSize = 64 + wsize;
    NSUInteger total = 64 + chunkSize * (NSUInteger)depth;
    uint8_t *buf = calloc(total, 1);
    if (!buf) {
        return nil;
    }
    buf[0] = 0x01;
    buf[4] = 0x02;
    for (int i = 0; i < depth; i++) {
        uint8_t *chunk = buf + 64 + (NSUInteger)i * chunkSize;
        chunk[0] = 0xEF;
        chunk[1] = 0xBE;
        chunk[2] = 0xAD;
        chunk[3] = 0xDE;
        chunk[4] = 0x01;
        chunk[10] = 0x08;
        uint16_t *fp16 = (uint16_t *)(chunk + 64);
        for (NSUInteger j = 0; j < wsize / 2; j++) {
            fp16[j] = 0x1c00; // ~0.05 fp16 — deep stacks stay finite
        }
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static double aneConvStackGFLOP(int ch, int sp, int depth) {
    // Each depth-wise conv: ~2 * ch * ch * sp FMAs.
    return (double)depth * 2.0 * (double)ch * (double)ch * (double)sp / 1e9;
}
