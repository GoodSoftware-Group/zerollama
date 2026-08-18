# Third-party notices (x/video-c)

## antirez/h3.c (MIT)

Portable MiniMax-H3 host layout / schedule code under `family_h3/h3_host.c` and
`include/h3_host.h` is vendored from [antirez/h3.c](https://github.com/antirez/h3.c)
(`h3_host.c` / `h3_host.h`). The velocity-reuse helper `family_h3/h3_reuse.c` is
extracted from the same project's `h3_dit.c` (`h3_dit_reuse_schedule`).
Audio VAE host constants (upsample rates, hop length) and CPU reference kernels
in `family_h3/h3_audio_vae_host.c` match antirez `h3_audio_vae.c` /
`tests/test_audio_gpu.c` (weight-norm, Conv1d, alias-free Snake); Kaiser-sinc
recompute matches `minimax-h3-mlx` / alias-free-torch. Host BigVGAN decode in
`family_h3/h3_audio_vae_decode.c` follows the antirez `h3_audio_vae_decode`
stage graph on CPU. AdaLN cache sizing in
`family_h3/h3_adaln_host.c` follows `minimax-h3-mlx/adaln.py`.

## NicoLab28 ClipProj (MIT)

`family_h3/h3_clipproj_host.c` implements the published affine (+ optional MLP /
`sink_out`) projection from [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3).
Control matrices used in rematch tests are downloaded under
`~/.zerollama/third_party/h3/clipproj/` (not vendored in-git). See
[docs/h3-clipproj.md](../../docs/h3-clipproj.md).

Full upstream license: `../h3.c/LICENSE` when the sibling rematch tree is cloned.
