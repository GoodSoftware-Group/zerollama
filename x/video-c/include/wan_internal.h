/*
 * wan_internal.h — shared internals for wan-c modules.
 */
#pragma once

#include "gguf_min.h"
#include "safetensors_min.h"
#include "zip_weight.h"
#include "uma_buf_load.h"
#include "wan.h"
#include "wan_config.h"

#include "uma/client.h"
#include "uma_wan_ops.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wan_caps {
  int gemm_f16;     /* k_ops GEMM_F16 */
  int layernorm;    /* k_ops LAYERNORM_MUL */
  int affine;       /* k_ops AFFINE_MUL_ADD */
  int group_norm;   /* k_ops GROUP_NORM */
  int rope3;        /* F0784 k_ops ROPE3 */
  int conv2d;       /* F0784 k_ops CONV2D */
  int ct2d;         /* F0801 CONV_TRANSPOSE2D / CT2D */
  int conv3d;       /* F0784/F0814 CONV3D */
  int unpatchify;   /* F0784 k_ops UNPATCHIFY3D */
  int attn_full;    /* F0783 ATTN_NAMED kind=full (HELP graph_wan) */
  int attn_bias;    /* F0945 ATTN_NAMED kind=bias|unscaled (umt5) */
  int gelu_tanh_mul; /* GELU_TANH_MUL (T5 gated FFN gate) */
  int silu_mul;     /* SILU_MUL (F0782 recipe) */
  int residual_add; /* RESIDUAL_ADD */
  int prefer_ext;   /* UMA_WAN_EXT=1 — EXT_CALL fallback / experimental */
  int ext_ready;    /* EXT_REGISTER succeeded for Wan kinds */
  int rope3_ext;    /* F0778 EXT_ROPE3 available */
  int layout_ext;   /* F0813 EXT_TOK3 / EXT_NCDHW3 */
  int tok3;         /* F0826 k_ops NCDHW_TOKENS / TOK3 */
  int ncdhw3;       /* F0826 k_ops TOKENS_NCDHW / NCDHW3 */
  int form_repeat;  /* F0822 GRAPH form=repeat (HELP) */
  int mech;         /* F0891 level=mech buffer binds */
  int sinusoid;     /* F0906 SINUSOID / TIMESTEP_EMB */
  int ct3d;         /* F0905 CONV_TRANSPOSE3D / CT3D */
  int gelu;         /* GELU / GELU_TANH (Wan DiT FFN) */
  int row_copy;     /* ROW_COPY / RCOPY (F0988 DiT chunk; prefer ATTN t=) */
  int ffn_gelu;     /* F0993 FFN_GELU Wan GELU(tanh) dense fuse */
  int head_rmsnorm; /* F0704 HEAD_RMSNORM (Wan qk_norm) */
  int channel_rms;  /* F0984 CHANNEL_RMS / RMS_CHANNEL (VAE) */
  int silu;         /* F0987 SILU / SWISH (VAE tip; not SILU_MUL) */
  int causal_pad3d; /* F0984 CAUSAL_PAD3D / CPAD3D */
  int nearest;      /* F0987 NEAREST / NEAREST2D */
} wan_caps;

struct wan_ctx {
  char ckpt_dir[4096];
  char uma_sock[256];
  char ext_sock[256];
  UmaClient *uma;
  uma_buf_pool *bufs;
  gguf_file *gguf;
  st_file *st;     /* diffusion_pytorch_model.safetensors (DiT as-is) */
  zw_file *t5_zip; /* models_t5_*.pth via indices/t5_embed_index.json */
  zw_file *vae_zip; /* Wan2.1_VAE.pth via indices/vae_index.json */
  wan_model_config cfg;
  int local_mode;
  wan_caps caps;
  /* Pipeline latent grid (NCDHW D,H,W with N=1,C=z_channels). */
  int gen_lt;
  int gen_lh;
  int gen_lw;
  /* DiT patch token grid after patch_embedding (tp,hp,wp); 0 = unset. */
  int gen_tp;
  int gen_hp;
  int gen_wp;
  /* Continuous flow timestep for DiT time MLP (Wan t≈sigma*1000). */
  float gen_t;
  /* Host f32 weight borrow cache (wan_borrow_tensor_f32). */
  void *weight_cache;
  /* F0994: persistent DiT block BANK (keys blocks.{i}.*). */
  int dit_persist_blocks; /* >0 when banked */
  int dit_persist_ready;
  int t5_persist_blocks; /* F1020: >0 when T5 weights banked */
  int t5_persist_ready;
  /* F1002: VAE HEADT tip weights banked (decoder.head / resample / u14). */
  int vae_headt_ready;
  /* Sticky projected text ctx (cond + uncond) across UniPC steps. */
  const float *dit_tctx_src[2];
  size_t dit_tctx_src_n[2];
  int dit_tctx_Tk[2];
  int dit_tctx_D[2];
  float *dit_tctx_pack[2];
  int dit_tctx_on_broker[2];
};

int wan_env_local(void);
int wan_env_prefer_ext(void);
/* F0909: default "GPU" (Metal GEMM_F16); WAN_GEMM_CPU=1 → "CPU". */
const char *wan_gemm_role(const wan_ctx *ctx);

int wan_probe_caps(wan_ctx *ctx);
int wan_ext_setup(wan_ctx *ctx);

int wan_submit_graph(UmaClient *c, const char *nodes);

int wan_graph_gemm_f32(wan_ctx *ctx, const char *bx, const char *by,
                       const char *bw, int M, int N, int K);
int wan_graph_layernorm(wan_ctx *ctx, const char *bx, const char *by,
                        const char *bw, int rows, int D);
int wan_graph_affine(wan_ctx *ctx, const char *bx, const char *by,
                     const char *bscale, const char *bshift, int rows, int D);
int wan_graph_groupnorm(wan_ctx *ctx, const char *bx, const char *by, int N,
                        int C, int spatial, int G);

/* F0784 ROPE3 k_ops, else EXT_ROPE3, else host. */
int wan_graph_rope3(wan_ctx *ctx, const char *bx, const char *by, int T, int H,
                    int HD);
/* One GRAPH: RoPE Q and K (shared freqs). */
int wan_graph_rope3_qk(wan_ctx *ctx, const char *bq, const char *bqr,
                       const char *bk, const char *bkr, int T, int H, int HD);
/* Ensure sticky RoPE freq bufs; fills gt/gh/gw when non-NULL. */
int wan_rope3_ensure_freqs(wan_ctx *ctx, int T, int HD, int *gt, int *gh,
                           int *gw);

/* F0784 CONV2D k_ops, else EXT_C2D (same-size), else host. */
int wan_graph_conv2d(wan_ctx *ctx, const char *bx, const char *by,
                     const char *bw, const char *bbias, int N, int Cin, int Hin,
                     int Win, int Cout, int KH, int KW, int stride, int pad);

/* F0801 CONV_TRANSPOSE2D / CT2D k_ops (ny can exceed nx). */
int wan_graph_ct2d(wan_ctx *ctx, const char *bx, const char *by, const char *bw,
                   const char *bbias, int N, int Cin, int Hin, int Win,
                   int Cout, int KH, int KW, int stride, int pad, int out_pad);

/* F0826 mainline layout: kind=N_C_D_H_W (aliases TOK3 / NCDHW3). */
int wan_graph_tok3(wan_ctx *ctx, const char *bx, const char *by,
                   const char *kind);
int wan_graph_ncdhw3(wan_ctx *ctx, const char *bx, const char *by,
                     const char *kind);

/* F0784 CONV3D — kind=N_Cin_Din_Hin_Win_Cout_KD_KH_KW_sd_sh_sw_pd_ph_pw */
int wan_graph_conv3d(wan_ctx *ctx, const char *bx, const char *by,
                     const char *bw, const char *kind);

/* F1001 HEADT tip: CHANNEL_RMS→SILU→CAUSAL_PAD3D→CONV3D (+optional bias gate). */
int wan_graph_vae_headt(wan_ctx *ctx, const char *bx, const char *by,
                        const char *bw, const char *bgamma, const char *bbias,
                        const char *brms, const char *bsil, const char *bpad,
                        int Cin, int Cout, int T, int H, int W);

/* F1012: HEADT with broker feat_cache on CAUSAL_PAD3D (mid=out=bcache, t=). */
int wan_graph_vae_headt_cache(wan_ctx *ctx, const char *bx, const char *by,
                              const char *bw, const char *bgamma,
                              const char *bbias, const char *brms,
                              const char *bsil, const char *bpad,
                              const char *bcache, int cache_t, int Cin,
                              int Cout, int T, int H, int W);

/* Brick 7: dual HEADT (+ optional resid) one GRAPH — same Cin==Cout. */
int wan_graph_vae_resblock_dual_headt(
    wan_ctx *ctx, const char *bx, const char *bmid, const char *by,
    const char *bw1, const char *bg1, const char *bb1, const char *brms1,
    const char *bsil1, const char *bpad1, const char *bcache1, int cache_t1,
    const char *bw2, const char *bg2, const char *bb2, const char *brms2,
    const char *bsil2, const char *bpad2, const char *bcache2, int cache_t2,
    int C, int T, int H, int W, int add_resid);

/* F1001 resample: NEAREST×2 → CONV2D (N=1, planar CHW treated as D=Cin). */
int wan_graph_vae_nearest_conv2d(wan_ctx *ctx, const char *blo, const char *bhi,
                                 const char *by, const char *bw,
                                 const char *bbias, int Cin, int Cout, int H,
                                 int W);

/* F0905 CONV_TRANSPOSE3D / CT3D — kind=18-int tuple (…_od_oh_ow). */
int wan_graph_ct3d(wan_ctx *ctx, const char *bx, const char *by, const char *bw,
                   const char *kind);

/* F0906 SINUSOID / TIMESTEP_EMB — x=t[N] y=emb[N,D] D even. */
int wan_graph_sinusoid(wan_ctx *ctx, const char *bt, const char *by, int N,
                       int D);

/* F0784 UNPATCHIFY3D k_ops, else EXT_UP3, else host. */
int wan_graph_unpatchify3d(wan_ctx *ctx, const char *bx, const char *by, int B,
                           int C, int T, int H, int W, int pt, int ph, int pw);

/* F0783 ATTN_NAMED kind=full (self or cross; Tk may be < T). */
int wan_graph_attn_full(wan_ctx *ctx, const char *bq, const char *bk,
                        const char *bv, const char *bout, int T, int Tk, int H,
                        int KV, int HD);
/* F0992: ATTN_NAMED with t= row window into full-seq q/out (Tk = full K). */
int wan_graph_attn_full_row(wan_ctx *ctx, const char *bq, const char *bk,
                            const char *bv, const char *bout, int T, int Tk,
                            int H, int KV, int HD, int t_row);
int wan_graph_silu_mul(wan_ctx *ctx, const char *bgate, const char *bup,
                       const char *by, int D);
/* F0945 umt5 biased (unscaled) attention: ATTN_NAMED kind=bias, mid=[H,T,Tk]. */
int wan_graph_attn_bias(wan_ctx *ctx, const char *bq, const char *bk,
                        const char *bv, const char *bbias, const char *bout,
                        int T, int Tk, int H, int KV, int HD);
/* F0987 gated GELU(tanh): y = gelu(gate)*up (T5FeedForward gate). */
int wan_graph_gelu_tanh_mul(wan_ctx *ctx, const char *bgate, const char *bup,
                            const char *by, int D);
int wan_graph_gelu(wan_ctx *ctx, const char *bx, const char *by, int D);
/* F0993: FFN_GELU y[t:t+M]=Wd@gelu(Wu@x[t:t+M]); mid holds M×ffn scratch. */
int wan_graph_ffn_gelu(wan_ctx *ctx, const char *bx, const char *by,
                       const char *bwu, const char *bwd, const char *bmid,
                       int M, int D, int ffn, int t_row);
int wan_graph_head_rmsnorm(wan_ctx *ctx, const char *bx, const char *bw, int H,
                           int HD);
int wan_graph_row_copy(wan_ctx *ctx, const char *bx, const char *by, int N,
                       int D, int src_row, int dst_row);
int wan_graph_residual_add(wan_ctx *ctx, const char *by, const char *bx, int D);
int wan_graph_copy(wan_ctx *ctx, const char *by, const char *bx, int D);

/* Chained DiT scaffold: LN → AdaLN → GEMM (RoPE/ATTN layered by caller). */
int wan_graph_dit_ln_affine_gemm(wan_ctx *ctx, const char *ba, const char *bb,
                                 const char *bw, const char *bs, const char *bt,
                                 int rows, int D);

void wan_fill_rope_freqs(float *freq, int npos, int HD);
/* Linear fallback: grid (T,1,1). Prefer wan_rope3_tokens_grid. */
int wan_rope3_tokens(float *tokens, int T, int H, int HD);
/* Wan patch order: idx=((od*H+oh)*W+ow) with grid (T,H,W)=(grid_t,grid_h,grid_w). */
int wan_rope3_tokens_grid(float *tokens, int T, int H, int HD, int grid_t,
                          int grid_h, int grid_w);
int wan_rope_axis_dim(int HD);

float *wan_load_tensor_f32(wan_ctx *ctx, const char *name, size_t *nelems_out);
/* Cached borrow — do not free; cleared in wan_ctx_close. */
const float *wan_borrow_tensor_f32(wan_ctx *ctx, const char *name,
                                   size_t *nelems_out);
void wan_weight_cache_clear(wan_ctx *ctx);
/* True if tensor exists in safetensors / zip index / GGUF. */
int wan_gguf_has(wan_ctx *ctx, const char *name);
void wan_fill_eye_nt(float *w, int N, int K);
/* Load gguf_name into BANK/BUF as buf_name; eye fallback if missing/size mismatch. */
int wan_put_weight_or_eye(wan_ctx *ctx, const char *buf_name,
                          const char *bank_key, const char *gguf_name, int N,
                          int K);
int wan_put_weight_raw(wan_ctx *ctx, const char *buf_name, const char *bank_key,
                       const char *gguf_name, size_t expect_nelems);

/* text may be NULL when ids provided; n = text_dim or text_len*text_dim. */
int wan_t5_encode(wan_ctx *ctx, const char *text, float *out, size_t n);
int wan_t5_encode_ids(wan_ctx *ctx, const int32_t *ids, size_t n_ids, float *out,
                      size_t n);
int wan_dit_denoise(wan_ctx *ctx, float *latent, size_t n, int step,
                    const float *text_emb, size_t text_n);
int wan_vae_decode(wan_ctx *ctx, const float *latent, size_t latent_n,
                   float *rgb, size_t rgb_n, int width, int height, int frames);

int wan_pipeline_t2v(wan_ctx *ctx, const wan_gen_params *p, float *rgb_out,
                     size_t rgb_cap, size_t *rgb_len);

#ifdef __cplusplus
}
#endif
