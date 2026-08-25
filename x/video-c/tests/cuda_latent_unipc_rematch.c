/* One latent UniPC denoise step on CUDA: 30-block DiT + head/unpatch + sched_unipc.
 * Fixture: dumps/latent_unipc_fixture/ from tools/gen_latent_unipc_fixture.py
 * Pager N=2; geometry latent [16,2,8,8] → T=32 (grid 2×4×4), patch (1,2,2).
 */
#include "backend_ops.h"
#include "dit_pager.h"
#include "dit_resident.h"
#include "safetensors_min.h"
#include "sched_unipc.h"
#include "wan_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  N_BLOCKS = 30,
  T = 32,
  TK = 512,
  H = 12,
  HD = 128,
  D = H * HD,
  FFN = 8960,
  GT = 2,
  GH = 4,
  GW = 4,
  C_LAT = 16,
  LT = 2,
  LH = 8,
  LW = 8,
  PT = 1,
  PH = 2,
  PW = 2,
  OUT_PER = C_LAT * PT * PH * PW /* 64 */
};

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float cosine(const float *a, const float *b, size_t n) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < n; i++) {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
  }
  if (na < 1e-30 || nb < 1e-30) return 0.f;
  return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static int read_f32(const char *path, float *dst, size_t n) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "FAIL open %s\n", path);
    return -1;
  }
  size_t got = fread(dst, sizeof(float), n, f);
  fclose(f);
  return got == n ? 0 : -1;
}

static const char *ckpt_dir(void) {
  const char *e = getenv("WAN_CKPT");
  if (e && e[0]) return e;
  return "/root/.zerollama/third_party/wan/Wan2.1-T2V-1.3B";
}

static const char *fix_dir(void) {
  const char *e = getenv("WAN_LATENT_FIXDIR");
  if (e && e[0]) return e;
  return "dumps/latent_unipc_fixture";
}

static void bank_key(char *o, size_t n, const char *pfx, int block) {
  snprintf(o, n, "%s_b%d", pfx, block);
}

static int load_linear(st_file *sf, wan_backend *b, const char *st_name,
                       const char *bank, int out, int in, float *tmp_oi,
                       float *tmp_io) {
  const st_tensor_t *t = st_find_tensor(sf, st_name);
  size_t n = (size_t)out * (size_t)in;
  if (!t || st_tensor_to_f32(sf, t, tmp_oi, n) != 0) return -1;
  wan_op_transpose_oi_f32(tmp_io, tmp_oi, out, in);
  return b->vt->bank_put(b, bank, tmp_io, n * sizeof(float));
}

static int load_vec(st_file *sf, wan_backend *b, const char *st_name,
                    const char *bank, int n, float *tmp) {
  const st_tensor_t *t = st_find_tensor(sf, st_name);
  if (!t || st_tensor_to_f32(sf, t, tmp, (size_t)n) != 0) return -1;
  return b->vt->bank_put(b, bank, tmp, (size_t)n * sizeof(float));
}

static const char *BLOCK_PFX[] = {
    "Wq",  "Wk",  "Wv",  "Wo",  "Wqc", "Wkc", "Wvc", "Woc", "Wu",  "Wd",
    "Bq",  "Bk",  "Bv",  "Bo",  "Bqc", "Bkc", "Bvc", "Boc", "Bu",  "Bd",
    "Nq",  "Nk",  "Nqc", "Nkc", "N3w", "N3b", "Mod"};
enum { N_BLOCK_PFX = 27 };

static int load_block(st_file *sf, wan_backend *b, int block, float *tmp_oi,
                      float *tmp_io, float *tmp_v) {
  char st[128], bk[64];
  #define LOAD_LIN(sfx, pfx, out, in)                                          \
    do {                                                                       \
      snprintf(st, sizeof(st), "blocks.%d." sfx, block);                       \
      bank_key(bk, sizeof(bk), pfx, block);                                    \
      if (load_linear(sf, b, st, bk, out, in, tmp_oi, tmp_io)) return -1;      \
    } while (0)
  #define LOAD_V(sfx, pfx, n)                                                  \
    do {                                                                       \
      snprintf(st, sizeof(st), "blocks.%d." sfx, block);                       \
      bank_key(bk, sizeof(bk), pfx, block);                                    \
      if (load_vec(sf, b, st, bk, n, tmp_v)) return -1;                        \
    } while (0)
  LOAD_LIN("self_attn.q.weight", "Wq", D, D);
  LOAD_LIN("self_attn.k.weight", "Wk", D, D);
  LOAD_LIN("self_attn.v.weight", "Wv", D, D);
  LOAD_LIN("self_attn.o.weight", "Wo", D, D);
  LOAD_LIN("cross_attn.q.weight", "Wqc", D, D);
  LOAD_LIN("cross_attn.k.weight", "Wkc", D, D);
  LOAD_LIN("cross_attn.v.weight", "Wvc", D, D);
  LOAD_LIN("cross_attn.o.weight", "Woc", D, D);
  LOAD_LIN("ffn.0.weight", "Wu", FFN, D);
  LOAD_LIN("ffn.2.weight", "Wd", D, FFN);
  LOAD_V("self_attn.q.bias", "Bq", D);
  LOAD_V("self_attn.k.bias", "Bk", D);
  LOAD_V("self_attn.v.bias", "Bv", D);
  LOAD_V("self_attn.o.bias", "Bo", D);
  LOAD_V("cross_attn.q.bias", "Bqc", D);
  LOAD_V("cross_attn.k.bias", "Bkc", D);
  LOAD_V("cross_attn.v.bias", "Bvc", D);
  LOAD_V("cross_attn.o.bias", "Boc", D);
  LOAD_V("ffn.0.bias", "Bu", FFN);
  LOAD_V("ffn.2.bias", "Bd", D);
  LOAD_V("self_attn.norm_q.weight", "Nq", D);
  LOAD_V("self_attn.norm_k.weight", "Nk", D);
  LOAD_V("cross_attn.norm_q.weight", "Nqc", D);
  LOAD_V("cross_attn.norm_k.weight", "Nkc", D);
  LOAD_V("norm3.weight", "N3w", D);
  LOAD_V("norm3.bias", "N3b", D);
  snprintf(st, sizeof(st), "blocks.%d.modulation", block);
  {
    const st_tensor_t *t = st_find_tensor(sf, st);
    if (!t || st_tensor_to_f32(sf, t, tmp_v, (size_t)6 * D) != 0) return -1;
    bank_key(bk, sizeof(bk), "Mod", block);
    if (b->vt->bank_put(b, bk, tmp_v, (size_t)6 * D * sizeof(float))) return -1;
  }
  #undef LOAD_LIN
  #undef LOAD_V
  return 0;
}

static int evict_block(wan_backend *b, int block) {
  for (int i = 0; i < N_BLOCK_PFX; i++) {
    char k[64];
    bank_key(k, sizeof(k), BLOCK_PFX[i], block);
    if (b->vt->bank_evict(b, k)) return -1;
  }
  return 0;
}

static int bind_block(wan_backend *b, int block) {
  for (int i = 0; i < N_BLOCK_PFX; i++) {
    char k[64];
    bank_key(k, sizeof(k), BLOCK_PFX[i], block);
    if (b->vt->bank_bind(b, k, BLOCK_PFX[i])) return -1;
  }
  return 0;
}

static size_t block_weight_bytes(void) {
  size_t d2 = (size_t)D * D * sizeof(float);
  size_t df = (size_t)D * FFN * sizeof(float);
  size_t dv = (size_t)D * sizeof(float);
  size_t fv = (size_t)FFN * sizeof(float);
  return 8 * d2 + 2 * df + 8 * dv + fv + dv + 4 * dv + 2 * dv + 6 * dv;
}

static int run_block(wan_backend *b, dit_pager *pg, st_file *sf, const float *e0,
                     float *tmp_oi, float *tmp_io, float *tmp_v, float *mod6,
                     unsigned block) {
  int evicted = -1;
  dit_pager_stats before = dit_pager_get_stats(pg);
  int slot = dit_pager_touch(pg, block, &evicted);
  if (slot < 0) return -1;
  int was_hit = dit_pager_get_stats(pg).hits > before.hits;
  if (!was_hit) {
    if (evicted >= 0 && evict_block(b, evicted) != 0) return -2;
    if (load_block(sf, b, (int)block, tmp_oi, tmp_io, tmp_v) != 0) return -3;
    dit_pager_set_slot_bytes(pg, slot, block_weight_bytes());
  }
  if (bind_block(b, (int)block) != 0) return -4;

  size_t dbytes = (size_t)D * sizeof(float);
  if (b->vt->buf_get(b, "Mod", mod6, 6 * dbytes)) return -5;
  for (int i = 0; i < 6 * D; i++) mod6[i] += e0[i];
  if (b->vt->buf_put(b, "SH", mod6 + 0 * D, dbytes) ||
      b->vt->buf_put(b, "SC", mod6 + 1 * D, dbytes) ||
      b->vt->buf_put(b, "GSA", mod6 + 2 * D, dbytes) ||
      b->vt->buf_put(b, "SH2", mod6 + 3 * D, dbytes) ||
      b->vt->buf_put(b, "SC2", mod6 + 4 * D, dbytes) ||
      b->vt->buf_put(b, "GFF", mod6 + 5 * D, dbytes))
    return -6;

  if (b->vt->layernorm(b, "X", "LN", NULL, T, D) ||
      b->vt->affine_mul_add(b, "LN", "AD", "SC", "SH", T, D))
    return -10;
  if (b->vt->gemm_f32(b, "AD", "Wq", "Q0", T, D, D) ||
      b->vt->bias_add(b, "Q0", "Q", "Bq", T, D) ||
      b->vt->rmsnorm(b, "Q", "Qr", "Nq", T, D))
    return -11;
  if (b->vt->gemm_f32(b, "AD", "Wk", "K0", T, D, D) ||
      b->vt->bias_add(b, "K0", "K", "Bk", T, D) ||
      b->vt->rmsnorm(b, "K", "Kr", "Nk", T, D))
    return -12;
  if (b->vt->gemm_f32(b, "AD", "Wv", "V0", T, D, D) ||
      b->vt->bias_add(b, "V0", "V", "Bv", T, D))
    return -13;
  if (b->vt->rope3(b, "Qr", "Qrope", T, H, HD, GT, GH, GW) ||
      b->vt->rope3(b, "Kr", "Krope", T, H, HD, GT, GH, GW) ||
      b->vt->attn_sdpa(b, "Qrope", "Krope", "V", "Attn", T, T, H, HD) ||
      b->vt->gemm_f32(b, "Attn", "Wo", "O0", T, D, D) ||
      b->vt->bias_add(b, "O0", "Dsa", "Bo", T, D) ||
      b->vt->gated_residual(b, "X", "Dsa", "GSA", "X1", T, D))
    return -14;

  if (b->vt->layernorm(b, "X1", "LN3r", NULL, T, D) ||
      b->vt->scale_bias(b, "LN3r", "LN3", "N3w", "N3b", T, D) ||
      b->vt->gemm_f32(b, "LN3", "Wqc", "Qc0", T, D, D) ||
      b->vt->bias_add(b, "Qc0", "Qc", "Bqc", T, D) ||
      b->vt->rmsnorm(b, "Qc", "Qcr", "Nqc", T, D) ||
      b->vt->gemm_f32(b, "CTX", "Wkc", "Kc0", TK, D, D) ||
      b->vt->bias_add(b, "Kc0", "Kc", "Bkc", TK, D) ||
      b->vt->rmsnorm(b, "Kc", "Kcr", "Nkc", TK, D) ||
      b->vt->gemm_f32(b, "CTX", "Wvc", "Vc0", TK, D, D) ||
      b->vt->bias_add(b, "Vc0", "Vc", "Bvc", TK, D) ||
      b->vt->attn_sdpa(b, "Qcr", "Kcr", "Vc", "XAttn", T, TK, H, HD) ||
      b->vt->gemm_f32(b, "XAttn", "Woc", "Oc0", T, D, D) ||
      b->vt->bias_add(b, "Oc0", "Dxa", "Boc", T, D) ||
      b->vt->gated_residual(b, "X1", "Dxa", "ONES", "X2", T, D))
    return -20;

  if (b->vt->layernorm(b, "X2", "LN2", NULL, T, D) ||
      b->vt->affine_mul_add(b, "LN2", "AD2", "SC2", "SH2", T, D) ||
      b->vt->gemm_f32(b, "AD2", "Wu", "MID0", T, FFN, D) ||
      b->vt->bias_add(b, "MID0", "MID", "Bu", T, FFN) ||
      b->vt->gelu_tanh(b, "MID", "GELU", (size_t)T * FFN) ||
      b->vt->gemm_f32(b, "GELU", "Wd", "D0", T, D, FFN) ||
      b->vt->bias_add(b, "D0", "Dff", "Bd", T, D) ||
      b->vt->gated_residual(b, "X2", "Dff", "GFF", "Y", T, D) ||
      b->vt->sync(b))
    return -30;

  size_t xb = (size_t)T * D * sizeof(float);
  float *yhost = malloc(xb);
  if (!yhost) return -40;
  if (b->vt->buf_get(b, "Y", yhost, xb) || b->vt->buf_put(b, "X", yhost, xb)) {
    free(yhost);
    return -41;
  }
  free(yhost);
  return was_hit ? 1 : 0;
}

/* Wan unpatchify: head vec [pt,ph,pw,C] → latent [C,F,H,W]. */
static void unpatchify(float *latent, const float *head, int tp, int hp, int wp) {
  size_t latent_n = (size_t)C_LAT * LT * LH * LW;
  memset(latent, 0, latent_n * sizeof(float));
  for (int tt = 0; tt < tp; tt++)
    for (int th = 0; th < hp; th++)
      for (int tw = 0; tw < wp; tw++) {
        size_t ti =
            (((size_t)tt * (size_t)hp + (size_t)th) * (size_t)wp + (size_t)tw);
        const float *vec = head + ti * (size_t)OUT_PER;
        for (int pti = 0; pti < PT; pti++)
          for (int phi = 0; phi < PH; phi++)
            for (int pwi = 0; pwi < PW; pwi++)
              for (int c = 0; c < C_LAT; c++) {
                size_t vi =
                    (((((size_t)pti * (size_t)PH + (size_t)phi) * (size_t)PW +
                       (size_t)pwi) *
                      (size_t)C_LAT) +
                     (size_t)c);
                int t = tt * PT + pti;
                int h = th * PH + phi;
                int ww = tw * PW + pwi;
                size_t oi =
                    (((((size_t)c * (size_t)LT + (size_t)t) * (size_t)LH +
                       (size_t)h) *
                      (size_t)LW) +
                     (size_t)ww);
                latent[oi] = vec[vi];
              }
      }
}

static int run_head_unpatch(wan_backend *b, st_file *sf, const float *e,
                            float *tmp_oi, float *tmp_io, float *tmp_v,
                            float *mod2, float *head_tok, float *model_out) {
  if (load_linear(sf, b, "head.head.weight", "Wh", OUT_PER, D, tmp_oi,
                  tmp_io))
    return -1;
  if (load_vec(sf, b, "head.head.bias", "Bh", OUT_PER, tmp_v)) return -2;
  {
    const st_tensor_t *t = st_find_tensor(sf, "head.modulation");
    if (!t || st_tensor_to_f32(sf, t, mod2, (size_t)2 * D) != 0) return -3;
  }
  /* Head: (modulation + e.unsqueeze(1)).chunk(2) — same e added to both rows. */
  for (int i = 0; i < D; i++) {
    mod2[i] += e[i];
    mod2[D + i] += e[i];
  }
  size_t dbytes = (size_t)D * sizeof(float);
  if (b->vt->buf_put(b, "SHh", mod2 + 0 * D, dbytes) ||
      b->vt->buf_put(b, "SCh", mod2 + 1 * D, dbytes))
    return -4;
  if (b->vt->layernorm(b, "X", "LNh", NULL, T, D) ||
      b->vt->affine_mul_add(b, "LNh", "ADh", "SCh", "SHh", T, D) ||
      b->vt->gemm_f32(b, "ADh", "Wh", "H0", T, OUT_PER, D) ||
      b->vt->bias_add(b, "H0", "HeadTok", "Bh", T, OUT_PER) ||
      b->vt->sync(b))
    return -5;
  if (b->vt->buf_get(b, "HeadTok", head_tok,
                     (size_t)T * OUT_PER * sizeof(float)))
    return -6;
  unpatchify(model_out, head_tok, GT, GH, GW);
  return 0;
}

int main(void) {
  const char *fix = fix_dir();
  char path[1024];
  snprintf(path, sizeof(path), "%s/diffusion_pytorch_model.safetensors",
           ckpt_dir());
  st_file *sf = st_open(path);
  if (!sf) {
    fprintf(stderr, "FAIL open %s\n", path);
    return 1;
  }

  int n_resident = wan_dit_resident_slots();
  if (n_resident <= 0) n_resident = 2;
  wan_backend *b = wan_backend_cuda_create(0);
  dit_pager *pg = dit_pager_create((unsigned)n_resident);
  if (!b || !pg) return 1;

  size_t xb = (size_t)T * D;
  size_t cb = (size_t)TK * D;
  size_t e0n = (size_t)6 * D;
  size_t lat_n = (size_t)C_LAT * LT * LH * LW;
  size_t oi = (size_t)FFN * D * sizeof(float);
  if (oi < (size_t)D * D * sizeof(float)) oi = (size_t)D * D * sizeof(float);
  if (oi < (size_t)OUT_PER * D * sizeof(float))
    oi = (size_t)OUT_PER * D * sizeof(float);

  float *X = malloc(xb * sizeof(float));
  float *E = malloc((size_t)D * sizeof(float));
  float *E0 = malloc(e0n * sizeof(float));
  float *CTX = malloc(cb * sizeof(float));
  float *ref_blk = malloc(xb * sizeof(float));
  float *out_blk = malloc(xb * sizeof(float));
  float *noise = malloc(lat_n * sizeof(float));
  float *py_out = malloc(lat_n * sizeof(float));
  float *py_s1 = malloc(lat_n * sizeof(float));
  float *model_out = malloc(lat_n * sizeof(float));
  float *head_tok = malloc((size_t)T * OUT_PER * sizeof(float));
  float *ones = malloc((size_t)D * sizeof(float));
  float *tmp_oi = malloc(oi);
  float *tmp_io = malloc(oi);
  float *tmp_v = malloc(e0n * sizeof(float));
  float *mod6 = malloc(e0n * sizeof(float));
  float *mod2 = malloc((size_t)2 * D * sizeof(float));
  if (!X || !E || !E0 || !CTX || !ref_blk || !out_blk || !noise || !py_out ||
      !py_s1 || !model_out || !head_tok || !ones || !tmp_oi || !tmp_io ||
      !tmp_v || !mod6 || !mod2)
    return 1;

  char p[1024];
  #define RD(name, dst, n)                                                     \
    do {                                                                       \
      snprintf(p, sizeof(p), "%s/" name, fix);                                 \
      if (read_f32(p, dst, n)) return 1;                                       \
    } while (0)
  RD("noise.f32", noise, lat_n);
  RD("x_in.f32", X, xb); /* reference tokens after PyTorch patch */
  RD("e.f32", E, (size_t)D);
  RD("e0.f32", E0, e0n);
  RD("context.f32", CTX, cb);
  RD("py_after_blocks.f32", ref_blk, xb);
  RD("py_model_out.f32", py_out, lat_n);
  RD("py_latent_s1.f32", py_s1, lat_n);
  #undef RD

  /* C patch_embedding: noise → tokens (host Conv3d; unlock for full CUDA path). */
  {
    size_t wne = (size_t)D * C_LAT * PT * PH * PW;
    float *pw = malloc(wne * sizeof(float));
    float *pb = malloc((size_t)D * sizeof(float));
    float *tok = malloc(xb * sizeof(float));
    if (!pw || !pb || !tok) return 1;
    const st_tensor_t *tw = st_find_tensor(sf, "patch_embedding.weight");
    const st_tensor_t *tb = st_find_tensor(sf, "patch_embedding.bias");
    if (!tw || !tb || st_tensor_to_f32(sf, tw, pw, wne) != 0 ||
        st_tensor_to_f32(sf, tb, pb, (size_t)D) != 0) {
      fprintf(stderr, "FAIL load patch_embedding\n");
      return 1;
    }
    if (wan_op_patch_embed_f32(tok, noise, pw, pb, C_LAT, LT, LH, LW, D, PT, PH,
                               PW) != 0) {
      fprintf(stderr, "FAIL patch_embed\n");
      return 1;
    }
    float cos_p = cosine(tok, X, xb);
    printf("ok: patch_embed cosine=%.6f\n", cos_p);
    if (cos_p < 0.99f) {
      fprintf(stderr, "FAIL patch_embed cosine %g\n", cos_p);
      return 1;
    }
    memcpy(X, tok, xb * sizeof(float)); /* drive trunk from C patch */
    free(pw);
    free(pb);
    free(tok);
  }

  for (int i = 0; i < D; i++) ones[i] = 1.f;
  if (b->vt->buf_put(b, "X", X, xb * sizeof(float)) ||
      b->vt->buf_put(b, "CTX", CTX, cb * sizeof(float)) ||
      b->vt->buf_put(b, "ONES", ones, (size_t)D * sizeof(float)))
    return 1;

  double t0 = now_s();
  for (int bi = 0; bi < N_BLOCKS; bi++) {
    if (run_block(b, pg, sf, E0, tmp_oi, tmp_io, tmp_v, mod6, (unsigned)bi) <
        0) {
      fprintf(stderr, "FAIL block %d\n", bi);
      return 1;
    }
    if ((bi + 1) % 5 == 0 || bi + 1 == N_BLOCKS)
      fprintf(stderr, "cuda latent: %d/%d\n", bi + 1, N_BLOCKS);
  }
  if (b->vt->buf_get(b, "X", out_blk, xb * sizeof(float))) return 1;
  float cos_blk = cosine(out_blk, ref_blk, xb);
  printf("ok: after_blocks cosine=%.6f\n", cos_blk);
  if (cos_blk < 0.99f) {
    fprintf(stderr, "FAIL after_blocks cosine %g\n", cos_blk);
    return 1;
  }

  if (run_head_unpatch(b, sf, E, tmp_oi, tmp_io, tmp_v, mod2, head_tok,
                       model_out) != 0) {
    fprintf(stderr, "FAIL head/unpatch\n");
    return 1;
  }
  float cos_mo = cosine(model_out, py_out, lat_n);
  printf("ok: model_out (head+unpatch) cosine=%.6f\n", cos_mo);
  if (cos_mo < 0.99f) {
    fprintf(stderr, "FAIL model_out cosine %g\n", cos_mo);
    return 1;
  }

  float *sample = malloc(lat_n * sizeof(float));
  float *pred = malloc(lat_n * sizeof(float));
  if (!sample || !pred) return 1;
  memcpy(sample, noise, lat_n * sizeof(float));
  /* Drive UniPC from C model_out (already rematched); still compare to py_s1. */
  memcpy(pred, model_out, lat_n * sizeof(float));
  sched_unipc *sched = sched_unipc_create(4, 5.0f);
  if (!sched) return 1;
  if (sched_unipc_step(sched, 0, pred, sample, lat_n) != 0) {
    fprintf(stderr, "FAIL sched_unipc_step\n");
    return 1;
  }
  float cos_s1 = cosine(sample, py_s1, lat_n);
  double ms = (now_s() - t0) * 1e3;
  dit_pager_stats st = dit_pager_get_stats(pg);
  printf("ok: latent_s1 UniPC cosine=%.6f N=%d pager_peak=%.1fMiB "
         "evict=%llu wall=%.1fms\n",
         cos_s1, n_resident, dit_pager_peak_bytes(pg) / (1024.0 * 1024.0),
         (unsigned long long)st.evictions, ms);
  if (cos_s1 < 0.99f) {
    fprintf(stderr, "FAIL latent_s1 cosine %g\n", cos_s1);
    return 1;
  }
  if (st.evictions < 1) {
    fprintf(stderr, "FAIL expected pager evictions\n");
    return 1;
  }

  snprintf(p, sizeof(p), "%s/c_model_out.f32", fix);
  FILE *f = fopen(p, "wb");
  if (f) {
    fwrite(model_out, sizeof(float), lat_n, f);
    fclose(f);
  }
  snprintf(p, sizeof(p), "%s/c_latent_s1.f32", fix);
  f = fopen(p, "wb");
  if (f) {
    fwrite(sample, sizeof(float), lat_n, f);
    fclose(f);
  }

  sched_unipc_destroy(sched);
  wan_backend_destroy(b);
  dit_pager_destroy(pg);
  st_close(sf);
  free(X);
  free(E);
  free(E0);
  free(CTX);
  free(ref_blk);
  free(out_blk);
  free(noise);
  free(py_out);
  free(py_s1);
  free(model_out);
  free(head_tok);
  free(ones);
  free(tmp_oi);
  free(tmp_io);
  free(tmp_v);
  free(mod6);
  free(mod2);
  free(sample);
  free(pred);
  return 0;
}
