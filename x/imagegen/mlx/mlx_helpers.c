// mlx_helpers.c - manually maintained MLX helpers (not auto-generated)

#include "mlx/c/array.h"
#include "mlx_dynamic.h"
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#define GET_SYM(handle, name) (void*)GetProcAddress((HMODULE)(handle), name)
#else
#include <dlfcn.h>
#define GET_SYM(handle, name) dlsym(handle, name)
#endif

static int (*mlx_array_detach_ptr)(mlx_array arr) = NULL;
static const uint16_t* (*mlx_array_data_bfloat16_ptr)(mlx_array arr) = NULL;

static void init_detach_ptr(void) {
    if (mlx_array_detach_ptr != NULL) {
        return;
    }
    void* handle = mlx_get_handle();
    if (handle == NULL) {
        return;
    }
    mlx_array_detach_ptr = (int (*)(mlx_array))GET_SYM(handle, "mlx_array_detach");
    if (mlx_array_detach_ptr == NULL) {
        fprintf(stderr, "MLX: mlx_array_detach symbol not found in library\n");
    }
    mlx_array_data_bfloat16_ptr = (const uint16_t* (*)(mlx_array))GET_SYM(handle, "mlx_array_data_bfloat16");
}

int mlx_go_array_detach(mlx_array arr) {
    init_detach_ptr();
    if (mlx_array_detach_ptr == NULL) {
        return 1;
    }
    return mlx_array_detach_ptr(arr);
}

const uint16_t* mlx_go_array_data_bfloat16(mlx_array arr) {
    init_detach_ptr();
    if (mlx_array_data_bfloat16_ptr == NULL) {
        return NULL;
    }
    return mlx_array_data_bfloat16_ptr(arr);
}

// mlx_go_trim_cuda_pool trims the default CUDA memory pool on device 0
// to release pool-reserved-but-freed memory back to the OS.
// This can free 100-300 MB between pipeline stages on tight VRAM hosts.
// Safe no-op on non-CUDA builds (symbol loaded via dlopen).
void mlx_go_trim_cuda_pool(void) {
#ifndef _WIN32
    static int (*trim_fn)(void* pool, size_t min_bytes_to_keep) = NULL;
    static int (*get_pool_fn)(void** pool_out, int device) = NULL;
    static int (*set_attr_fn)(void* pool, int attr, void* value) = NULL;
    static int loaded = 0;
    if (!loaded) {
        loaded = 1;
        void* cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
        if (!cudart) cudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
        if (cudart) {
            trim_fn = dlsym(cudart, "cudaMemPoolTrimTo");
            get_pool_fn = dlsym(cudart, "cudaDeviceGetDefaultMemPool");
            set_attr_fn = dlsym(cudart, "cudaMemPoolSetAttribute");
        }
    }
    if (trim_fn && get_pool_fn) {
        void* pool = NULL;
        if (get_pool_fn(&pool, 0) == 0 && pool != NULL) {
            // Set release threshold to 0 so the pool returns memory immediately
            // when trim is called, rather than holding reservation for reuse.
            if (set_attr_fn) {
                size_t zero = 0;
                // cudaMemPoolAttrReleaseThreshold = 1
                set_attr_fn(pool, 1, &zero);
            }
            trim_fn(pool, 0);
        }
    }
#endif
}

// mlx_go_set_cuda_pool_threshold sets cudaMemPoolAttrReleaseThreshold for the
// default CUDA memory pool. Setting to 0 causes the pool to immediately release
// freed memory to the system, maximally reducing peak VRAM at cost of some
// allocation latency. Call once at startup before model loading.
void mlx_go_set_cuda_pool_threshold(size_t threshold) {
#ifndef _WIN32
    static int (*get_pool_fn)(void** pool_out, int device) = NULL;
    static int (*set_attr_fn)(void* pool, int attr, void* value) = NULL;
    static int loaded = 0;
    if (!loaded) {
        loaded = 1;
        void* cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
        if (!cudart) cudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
        if (cudart) {
            get_pool_fn = dlsym(cudart, "cudaDeviceGetDefaultMemPool");
            set_attr_fn = dlsym(cudart, "cudaMemPoolSetAttribute");
        }
    }
    if (get_pool_fn && set_attr_fn) {
        void* pool = NULL;
        if (get_pool_fn(&pool, 0) == 0 && pool != NULL) {
            // cudaMemPoolAttrReleaseThreshold = 1
            set_attr_fn(pool, 1, &threshold);
        }
    }
#endif
}

// mlx_go_export_latents_bin_d2h is implemented in libmlxc (cudaMemcpy D2H export).
int mlx_go_export_latents_bin_d2h(const char* path, mlx_array gpu) {
    init_detach_ptr();
    static int (*export_fn)(const char*, mlx_array) = NULL;
    static int loaded = 0;
    if (!loaded) {
        loaded = 1;
        void* handle = mlx_get_handle();
        if (handle != NULL) {
            export_fn = (int (*)(const char*, mlx_array))GET_SYM(
                handle, "mlx_go_export_latents_bin_d2h");
        }
    }
    if (export_fn == NULL) {
        return -1;
    }
    return export_fn(path, gpu);
}
