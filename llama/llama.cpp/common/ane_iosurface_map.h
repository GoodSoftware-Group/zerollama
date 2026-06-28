// IOSurface → MTLBuffer mapping (mirrors ggml_metal_buffer_map_iosurface page alignment).
#pragma once

#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

static inline id<MTLBuffer> ane_ggml_map_iosurface_base(id<MTLDevice> device,
                                                        void * ptr,
                                                        size_t size,
                                                        void ** mappedBase,
                                                        size_t * mappedSize) {
    const size_t size_page = (size_t) sysconf(_SC_PAGESIZE);

    void * alignedPtr = ptr;
    size_t alignedSize = size;

    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        alignedPtr = (void *) ((char *) ptr - offs);
        alignedSize += offs;
    }

    if ((alignedSize % size_page) != 0) {
        alignedSize += (size_page - (alignedSize % size_page));
    }

    if (mappedBase) {
        *mappedBase = alignedPtr;
    }
    if (mappedSize) {
        *mappedSize = alignedSize;
    }

    if (alignedSize == 0) {
        return nil;
    }

    return [device newBufferWithBytesNoCopy:alignedPtr
                                     length:alignedSize
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

static inline IOSurfaceRef ane_ggml_surface_from_id(uint32_t surfaceID) {
    return IOSurfaceLookup(surfaceID);
}
