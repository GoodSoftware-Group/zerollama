import pathlib, sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
changed = False

old_malloc = '        cu::device(device).make_current();\n        if (mem_pools_[device]) { // supports memory pools\n          CHECK_CUDA_ERROR(cudaMallocAsync(&data, size, stream));\n        } else {\n          CHECK_CUDA_ERROR(cudaMalloc(&data, size));\n        }'
new_malloc = '        cu::device(device).make_current();\n        // Use synchronous cudaMalloc instead of cudaMallocAsync. The async memory\n        // pool reserves address space that counts against physical VRAM on 16GB GPUs.\n        CHECK_CUDA_ERROR(cudaMalloc(&data, size));'

if "Use synchronous cudaMalloc" in text:
    print("Skip: malloc already sync")
elif old_malloc in text:
    text = text.replace(old_malloc, new_malloc, 1)
    print("Patched cudaMallocAsync -> cudaMalloc")
    changed = True
else:
    print("ERROR: malloc async block not found — MLX allocator shape changed", file=sys.stderr)
    raise SystemExit(1)

old_free = '  if (buf.device == -1) {\n    unified_free(buf.data);\n  } else {\n    // Free asynchronously when memory pools is supported.\n    if (mem_pools_[buf.device]) {\n      if (!stream) {\n        stream = free_streams_[buf.device];\n      }\n      CHECK_CUDA_ERROR(cudaFreeAsync(buf.data, stream));\n    } else {\n      CHECK_CUDA_ERROR(cudaFree(buf.data));\n    }\n  }'
new_free = '  if (buf.device == -1) {\n    unified_free(buf.data);\n  } else {\n    // zerollama: sync cudaFree — pairs with forced cudaMalloc (no async pool).\n    (void)stream;\n    CHECK_CUDA_ERROR(cudaFree(buf.data));\n  }'

if "zerollama: sync cudaFree" in text:
    print("Skip: free_async already sync")
elif old_free in text:
    text = text.replace(old_free, new_free, 1)
    print("Patched cudaFreeAsync -> cudaFree")
    changed = True
else:
    print("ERROR: free_async pool block not found — MLX allocator shape changed", file=sys.stderr)
    raise SystemExit(1)

if "memory_limit_ = total_memory_ * 0.95;" in text:
    text = text.replace(
        "memory_limit_ = total_memory_ * 0.95;",
        "memory_limit_ = total_memory_ * 0.90;",
        1,
    )
    print("Patched memory_limit 0.95 -> 0.90")
    changed = True
elif "memory_limit_ = total_memory_ * 0.90;" in text:
    print("Skip: memory_limit already 0.90")
else:
    print("ERROR: memory_limit assignment not found", file=sys.stderr)
    raise SystemExit(1)

old_recycle = '  if (get_cache_memory() < max_pool_size_) {\n    buffer_cache_.recycle_to_cache(buf);\n  } else {\n    free_cuda_buffer(buf);\n  }'
new_recycle = '  // Always return memory to the driver (see zerollama imagegen checkpoint frees).\n  free_cuda_buffer(buf);'

if "Always return memory to the driver" in text:
    print("Skip: recycle already disabled")
elif old_recycle in text:
    text = text.replace(old_recycle, new_recycle, 1)
    print("Patched allocator: disable buffer cache recycle")
    changed = True
else:
    print("ERROR: recycle block not found — MLX allocator shape changed", file=sys.stderr)
    raise SystemExit(1)

path.write_text(text)
print(("Updated" if changed else "Unchanged") + ":", path)
