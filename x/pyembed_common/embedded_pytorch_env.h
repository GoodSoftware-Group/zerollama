#ifndef OLLAMA_EMBEDDED_PYTORCH_ENV_H
#define OLLAMA_EMBEDDED_PYTORCH_ENV_H

/*
 * Prepare the process library search path before Py_Initialize when embedded
 * Python will import torch. ggml/MLX serve scripts often prepend /usr/hostlibs
 * (older libcudnn) to LD_LIBRARY_PATH; PyTorch wheels ship a matching cuDNN in
 * site-packages/torch/lib and must win the dynamic linker search order.
 *
 * repo_root: zerollama repository (may contain .venv-training).
 * py_major/py_minor: embedded libpython version (from Py_GetVersion() before init).
 */
void embedded_prepare_pytorch_ld_path_ex(const char *repo_root, int py_major, int py_minor);

#endif
