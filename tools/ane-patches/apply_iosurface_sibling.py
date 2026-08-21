#!/usr/bin/env python3
"""Apply ggml IOSurface hook to elizaOS llama.cpp sibling (minimal, idempotent)."""
from __future__ import annotations

import pathlib
import sys


def patch_once(path: pathlib.Path, needle: str, repl: str, label: str, *, required: bool = True) -> None:
    text = path.read_text()
    if repl.strip() in text:
        print(f"  skip {label} (already applied)")
        return
    if needle not in text:
        if not required:
            print(f"  skip {label} (anchor missing — optional)")
            return
        raise SystemExit(f"anchor missing for {label} in {path}")
    path.write_text(text.replace(needle, repl, 1))
    print(f"  patched {label}")


def fix_metal_device_name_check(metal_cpp: pathlib.Path) -> None:
    """Upgrade early IOSurface patches that used strcmp(MTL) — device names are MTL0, MTL1, …"""
    text = metal_cpp.read_text()
    old = (
        '    if (strcmp(device->iface.get_name(device), GGML_METAL_NAME) != 0) {\n'
        '        GGML_LOG_ERROR("%s: device is not Metal\\n", __func__);'
    )
    new = (
        '    const char * name = device->iface.get_name(device);\n'
        '    if (!name || strncmp(name, GGML_METAL_NAME, strlen(GGML_METAL_NAME)) != 0) {\n'
        '        GGML_LOG_ERROR("%s: device is not Metal (name=%s)\\n", __func__, name ? name : "null");'
    )
    if old in text:
        metal_cpp.write_text(text.replace(old, new, 1))
        print("  fixed metal.cpp IOSurface device name check (strcmp → strncmp)")


def main() -> None:
    root = pathlib.Path(sys.argv[1])

    device_m = root / "ggml/src/ggml-metal/ggml-metal-device.m"
    patch_once(
        device_m,
        '#include <Foundation/Foundation.h>\n\n#include <Metal/Metal.h>',
        '#include <Foundation/Foundation.h>\n\n#import <IOSurface/IOSurface.h>\n\n#include <Metal/Metal.h>',
        "device.m IOSurface import",
    )
    patch_once(
        device_m,
        "    // pointers to global device\n    ggml_metal_device_t dev;\n};",
        "    // pointers to global device\n    ggml_metal_device_t dev;\n\n    // optional IOSurface backing (retained until buffer free)\n    IOSurfaceRef iosurface;\n};",
        "device.m struct iosurface field",
    )
    patch_once(
        device_m,
        "    return res;\n}\n\nvoid ggml_metal_buffer_free(ggml_metal_buffer_t buf) {",
        """    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map_iosurface(ggml_metal_device_t dev, uint32_t surface_id, size_t size, size_t max_tensor_size) {
    IOSurfaceRef surface = IOSurfaceLookup(surface_id);
    if (!surface) {
        GGML_LOG_ERROR("%s: IOSurfaceLookup(%u) failed (same-process only)\\n", __func__, surface_id);
        return NULL;
    }

    IOSurfaceLock(surface, 0, NULL);
    void * base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        IOSurfaceUnlock(surface, 0, NULL);
        CFRelease(surface);
        GGML_LOG_ERROR("%s: IOSurfaceGetBaseAddress(%u) failed\\n", __func__, surface_id);
        return NULL;
    }

    ggml_metal_buffer_t res = ggml_metal_buffer_map(dev, base, size, max_tensor_size);

    IOSurfaceUnlock(surface, 0, NULL);

    if (!res) {
        CFRelease(surface);
        return NULL;
    }

    res->iosurface = (IOSurfaceRef) CFRetain(surface);
    CFRelease(surface);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {""",
        "device.m map_iosurface",
    )
    patch_once(
        device_m,
        "    if (buf->is_shared && buf->owned) {\n#if TARGET_OS_OSX\n        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);\n#else\n        free(buf->all_data);\n#endif\n    }\n\n    free(buf);\n}",
        "    if (buf->is_shared && buf->owned) {\n#if TARGET_OS_OSX\n        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);\n#else\n        free(buf->all_data);\n#endif\n    }\n\n    if (buf->iosurface) {\n        CFRelease(buf->iosurface);\n        buf->iosurface = NULL;\n    }\n\n    free(buf);\n}",
        "device.m buffer_free iosurface",
    )

    device_h = root / "ggml/src/ggml-metal/ggml-metal-device.h"
    patch_once(
        device_h,
        "ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);\n",
        "ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);\n"
        "ggml_metal_buffer_t ggml_metal_buffer_map_iosurface(ggml_metal_device_t dev, uint32_t surface_id, size_t size, size_t max_tensor_size);\n",
        "device.h declare",
    )

    metal_h = root / "ggml/include/ggml-metal.h"
    patch_once(
        metal_h,
        "#include <stdbool.h>\n",
        "#include <stdbool.h>\n#include <stdint.h>\n",
        "metal.h stdint",
    )
    patch_once(
        metal_h,
        "GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);\n\n#ifdef __cplusplus\n}\n#endif",
        """GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_dev_buffer_from_iosurface(
        ggml_backend_dev_t device,
        uint32_t surface_id,
        size_t size,
        size_t max_tensor_size);

#ifdef __cplusplus
}
#endif""",
        "metal.h public API",
    )

    metal_cpp = root / "ggml/src/ggml-metal/ggml-metal.cpp"
    patch_once(
        metal_cpp,
        """static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {""",
        """static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_from_iosurface(
        ggml_backend_dev_t dev, uint32_t surface_id, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map_iosurface(ctx_dev, surface_id, size, max_tensor_size);
    if (!res) {
        return nullptr;
    }

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(
            ggml_backend_metal_buffer_type_mapped(props_dev->device),
            ggml_backend_metal_buffer_shared_i,
            res,
            size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {""",
        "metal.cpp device buffer_from_iosurface",
    )
    patch_once(
        metal_cpp,
        """static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }

    return NULL;

    GGML_UNUSED(reg);
}""",
        """static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }
    if (strcmp(name, "ggml_backend_dev_buffer_from_iosurface") == 0) {
        return (void *)ggml_backend_dev_buffer_from_iosurface;
    }

    return NULL;

    GGML_UNUSED(reg);
}""",
        "metal.cpp proc address",
        required=False,
    )
    if "ggml_backend_dev_buffer_from_iosurface" in metal_cpp.read_text():
        has_public = "GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_dev_buffer_from_iosurface(" in metal_cpp.read_text()
        if has_public:
            print("  skip metal.cpp public API (already applied)")
        else:
            patch_once(
                metal_cpp,
                "    return &reg;\n}\n\nGGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)",
                """    return &reg;
}

GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_dev_buffer_from_iosurface(
        ggml_backend_dev_t device,
        uint32_t surface_id,
        size_t size,
        size_t max_tensor_size) {
    if (!device || !device->iface.get_name) {
        return nullptr;
    }

    const char * name = device->iface.get_name(device);
    if (!name || strncmp(name, GGML_METAL_NAME, strlen(GGML_METAL_NAME)) != 0) {
        GGML_LOG_ERROR("%s: device is not Metal (name=%s)\\n", __func__, name ? name : "null");
        return nullptr;
    }

    return ggml_backend_metal_device_buffer_from_iosurface(device, surface_id, size, max_tensor_size);
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)""",
                "metal.cpp public API",
            )
    else:
        patch_once(
            metal_cpp,
            "    return &reg;\n}\n\nGGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)",
            """    return &reg;
}

GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_dev_buffer_from_iosurface(
        ggml_backend_dev_t device,
        uint32_t surface_id,
        size_t size,
        size_t max_tensor_size) {
    if (!device || !device->iface.get_name) {
        return nullptr;
    }

    const char * name = device->iface.get_name(device);
    if (!name || strncmp(name, GGML_METAL_NAME, strlen(GGML_METAL_NAME)) != 0) {
        GGML_LOG_ERROR("%s: device is not Metal (name=%s)\\n", __func__, name ? name : "null");
        return nullptr;
    }

    return ggml_backend_metal_device_buffer_from_iosurface(device, surface_id, size, max_tensor_size);
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)""",
            "metal.cpp public API",
        )

    cmake = root / "ggml/src/ggml-metal/CMakeLists.txt"
    patch_once(
        cmake,
        "find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)\n",
        "find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)\nfind_library(IOSURFACE_FRAMEWORK IOSurface REQUIRED)\n",
        "CMake IOSurface find",
    )
    patch_once(
        cmake,
        "                      ${METALKIT_FRAMEWORK}\n                      )",
        "                      ${METALKIT_FRAMEWORK}\n                      ${IOSURFACE_FRAMEWORK}\n                      )",
        "CMake IOSurface link",
    )

    fix_metal_device_name_check(root / "ggml/src/ggml-metal/ggml-metal.cpp")
    apply_rsets_pause_resume(root)


def apply_rsets_pause_resume(root: pathlib.Path) -> None:
    """P70 (lab, Jul 2026): pause/resume the residency-set keep-alive heartbeat.

    Investigated as a candidate fix for an intermittent SIGSEGV in Metal resource-list code
    during ANE dflash chain-17 lab runs (docs/ane-draft-inprocess.md "Known issues"). Did NOT
    fix that crash on its own, but is a real, independently useful, ref-counted primitive — kept
    so it survives a fresh llama.cpp checkout + re-patch.
    """
    device_m = root / "ggml/src/ggml-metal/ggml-metal-device.m"
    patch_once(
        device_m,
        "    // background heartbeat thread to keep the residency sets alive\n"
        "    atomic_bool d_stop;\n"
        "    atomic_int  d_loop;\n\n"
        "    dispatch_group_t d_group;\n};",
        "    // background heartbeat thread to keep the residency sets alive\n"
        "    atomic_bool d_stop;\n"
        "    atomic_int  d_loop;\n\n"
        "    // P70 (lab): let callers temporarily suppress requestResidency calls while they hold\n"
        "    // a long host-side compute window on a buffer whose residency set is registered\n"
        "    // here, without fully tearing down the set. Ref-counted so nested pause/resume from\n"
        "    // multiple call sites (or accidental double-pause) cannot leave the heartbeat stuck off.\n"
        "    atomic_int  d_paused;\n\n"
        "    dispatch_group_t d_group;\n};",
        "device.m rsets d_paused field",
    )
    patch_once(
        device_m,
        "    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);\n"
        "    atomic_store_explicit(&res->d_loop, res->loops_per_s*res->keep_alive_s, memory_order_relaxed);",
        "    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);\n"
        "    atomic_store_explicit(&res->d_loop, res->loops_per_s*res->keep_alive_s, memory_order_relaxed);\n"
        "    atomic_store_explicit(&res->d_paused, 0, memory_order_relaxed);",
        "device.m rsets init d_paused",
    )
    patch_once(
        device_m,
        "              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {\n"
        "                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {\n"
        "                      [res->lock lock];",
        "              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {\n"
        "                  if (atomic_load_explicit(&res->d_paused, memory_order_relaxed) <= 0 &&\n"
        "                      atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {\n"
        "                      [res->lock lock];",
        "device.m rsets heartbeat pause check",
    )
    patch_once(
        device_m,
        "void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {\n"
        "    if (dev->rsets == NULL) {\n"
        "        return;\n"
        "    }\n\n"
        "    atomic_store_explicit(&dev->rsets->d_loop, dev->rsets->loops_per_s*dev->rsets->keep_alive_s, memory_order_relaxed);\n"
        "}",
        "void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {\n"
        "    if (dev->rsets == NULL) {\n"
        "        return;\n"
        "    }\n\n"
        "    atomic_store_explicit(&dev->rsets->d_loop, dev->rsets->loops_per_s*dev->rsets->keep_alive_s, memory_order_relaxed);\n"
        "}\n\n"
        "// P70 (lab): pause/resume the background requestResidency heartbeat. Ref-counted: the\n"
        "// heartbeat stays paused while the count is > 0. Callers must pair every pause with a\n"
        "// resume (e.g. via a scope guard) even on early-return/error paths. This does not touch\n"
        "// [lock]/[data] membership — it only skips the periodic requestResidency calls, so\n"
        "// add/rm from other threads are unaffected and still fully synchronized by\n"
        "// dev->rsets->lock as before.\n"
        "void ggml_metal_device_rsets_pause(ggml_metal_device_t dev) {\n"
        "    if (dev->rsets == NULL) {\n"
        "        return;\n"
        "    }\n\n"
        "    atomic_fetch_add_explicit(&dev->rsets->d_paused, 1, memory_order_relaxed);\n"
        "}\n\n"
        "void ggml_metal_device_rsets_resume(ggml_metal_device_t dev) {\n"
        "    if (dev->rsets == NULL) {\n"
        "        return;\n"
        "    }\n\n"
        "    const int prev = atomic_fetch_sub_explicit(&dev->rsets->d_paused, 1, memory_order_relaxed);\n"
        "    if (prev <= 0) {\n"
        "        // guard against unbalanced resume (should not happen if callers pair pause/resume)\n"
        "        atomic_store_explicit(&dev->rsets->d_paused, 0, memory_order_relaxed);\n"
        "    }\n"
        "}",
        "device.m rsets_pause/resume impl",
    )

    device_h = root / "ggml/src/ggml-metal/ggml-metal-device.h"
    patch_once(
        device_h,
        "void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);",
        "void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);\n\n"
        "// P70 (lab): temporarily suppress the background requestResidency heartbeat. Ref-counted;\n"
        "// every ggml_metal_device_rsets_pause() must be paired with exactly one\n"
        "// ggml_metal_device_rsets_resume(), including on error/early-return paths.\n"
        "void ggml_metal_device_rsets_pause (ggml_metal_device_t dev);\n"
        "void ggml_metal_device_rsets_resume(ggml_metal_device_t dev);",
        "device.h rsets_pause/resume declare",
    )

    metal_h = root / "ggml/include/ggml-metal.h"
    metal_h_text = metal_h.read_text()
    rsets_decl_marker = "ggml_backend_metal_dev_rsets_pause"
    if rsets_decl_marker in metal_h_text:
        print("  skip metal.h rsets_pause/resume declare (already applied)")
    elif "#ifdef __cplusplus\n}\n#endif" not in metal_h_text:
        raise SystemExit(f"anchor missing for metal.h rsets_pause/resume declare in {metal_h}")
    else:
        # Append a standalone extern "C" block after the existing one, rather than editing
        # inside it — keeps the original IOSurface patch's own idempotency check intact.
        addition = (
            "\n#ifdef __cplusplus\n"
            "extern \"C\" {\n"
            "#endif\n\n"
            "// P70 (lab): pause/resume the background residency-set keep-alive heartbeat for this\n"
            "// Metal device. Callers holding a long host-side compute window (no GPU submissions on\n"
            "// this device) can pause the heartbeat to avoid racing its requestResidency calls\n"
            "// against the next command encoder/resource-list mutation, then must resume it\n"
            "// afterwards (including on early-return / error paths) via a scope guard. Ref-counted;\n"
            "// safe to call from any thread. No-op if device is not Metal or has no residency-set\n"
            "// support.\n"
            "GGML_BACKEND_API void ggml_backend_metal_dev_rsets_pause (ggml_backend_dev_t device);\n"
            "GGML_BACKEND_API void ggml_backend_metal_dev_rsets_resume(ggml_backend_dev_t device);\n\n"
            "#ifdef __cplusplus\n"
            "}\n"
            "#endif\n"
        )
        metal_h.write_text(metal_h_text + addition)
        print("  patched metal.h rsets_pause/resume declare")

    metal_cpp = root / "ggml/src/ggml-metal/ggml-metal.cpp"
    patch_once(
        metal_cpp,
        "    return ggml_backend_metal_device_buffer_from_iosurface(device, surface_id, size, max_tensor_size);\n"
        "}\n\nGGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)",
        "    return ggml_backend_metal_device_buffer_from_iosurface(device, surface_id, size, max_tensor_size);\n"
        "}\n\n"
        "GGML_BACKEND_API void ggml_backend_metal_dev_rsets_pause(ggml_backend_dev_t device) {\n"
        "    if (!device || !device->iface.get_name) {\n"
        "        return;\n"
        "    }\n"
        "    const char * name = device->iface.get_name(device);\n"
        "    if (!name || strncmp(name, GGML_METAL_NAME, strlen(GGML_METAL_NAME)) != 0) {\n"
        "        return;\n"
        "    }\n"
        "    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) device->context;\n"
        "    ggml_metal_device_rsets_pause(ctx_dev);\n"
        "}\n\n"
        "GGML_BACKEND_API void ggml_backend_metal_dev_rsets_resume(ggml_backend_dev_t device) {\n"
        "    if (!device || !device->iface.get_name) {\n"
        "        return;\n"
        "    }\n"
        "    const char * name = device->iface.get_name(device);\n"
        "    if (!name || strncmp(name, GGML_METAL_NAME, strlen(GGML_METAL_NAME)) != 0) {\n"
        "        return;\n"
        "    }\n"
        "    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) device->context;\n"
        "    ggml_metal_device_rsets_resume(ctx_dev);\n"
        "}\n\n"
        "GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)",
        "metal.cpp rsets_pause/resume impl",
        required=False,
    )


if __name__ == "__main__":
    main()
