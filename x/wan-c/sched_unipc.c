#include "sched_unipc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Flow UniPC matched to Wan FlowUniPCMultistepScheduler:
 *   schedule: linspace(1, 1/1000, steps+1)[:-1] → shift → append 0
 *   convert:  x0 = x - σ · v   (flow_prediction, predict_x0)
 *   solver:   UniP + UniC B(h=bh2), solver_order≤3
 *   disable_corrector: WAN_UNIPC_DISABLE_CORRECTOR=0,1,...
 *   order override:    WAN_UNIPC_ORDER=2|3 (default 3)
 */

#define WAN_UNIPC_ORDER_MAX 3
#define WAN_UNIPC_DISABLE_MAX 16

struct sched_unipc {
  int steps;
  float shift;
  float *sigmas;
  int n_sigmas;
  int solver_order;
  float *m_hist[WAN_UNIPC_ORDER_MAX]; /* x0 preds; newest at n_hist-1 */
  int n_hist;
  float *last_sample;
  float *x0_cur;
  size_t buf_n;
  int lower_order_nums;
  int this_order;
  int disable_corrector[WAN_UNIPC_DISABLE_MAX];
  int n_disable;
  int has_last;
};

float sched_unipc_warp_sigma(float sigma, float shift) {
  if (shift <= 0.0f)
    return sigma;
  return shift * sigma / (1.0f + (shift - 1.0f) * sigma);
}

static float sigma_lambda(float sigma) {
  float a = fmaxf(1.0f - sigma, 1e-12f);
  float s = fmaxf(sigma, 1e-12f);
  return logf(a) - logf(s);
}

static float clamp_rk(float rk) {
  if (fabsf(rk) < 1e-12f)
    return (rk < 0.f) ? -1e-12f : 1e-12f;
  return rk;
}

/* Solve A x = b for n×n (n≤3) via Gaussian elimination with partial pivot. */
static int solve_lin(float *A, float *b, float *x, int n) {
  float M[WAN_UNIPC_ORDER_MAX][WAN_UNIPC_ORDER_MAX];
  float rhs[WAN_UNIPC_ORDER_MAX];
  if (n < 1 || n > WAN_UNIPC_ORDER_MAX)
    return -1;
  for (int i = 0; i < n; i++) {
    rhs[i] = b[i];
    for (int j = 0; j < n; j++)
      M[i][j] = A[i * n + j];
  }
  for (int k = 0; k < n; k++) {
    int piv = k;
    float best = fabsf(M[k][k]);
    for (int i = k + 1; i < n; i++) {
      float v = fabsf(M[i][k]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (best < 1e-12f)
      return -1;
    if (piv != k) {
      for (int j = 0; j < n; j++) {
        float tmp = M[k][j];
        M[k][j] = M[piv][j];
        M[piv][j] = tmp;
      }
      float tb = rhs[k];
      rhs[k] = rhs[piv];
      rhs[piv] = tb;
    }
    float diag = M[k][k];
    for (int j = k; j < n; j++)
      M[k][j] /= diag;
    rhs[k] /= diag;
    for (int i = 0; i < n; i++) {
      if (i == k)
        continue;
      float f = M[i][k];
      for (int j = k; j < n; j++)
        M[i][j] -= f * M[k][j];
      rhs[i] -= f * rhs[k];
    }
  }
  for (int i = 0; i < n; i++)
    x[i] = rhs[i];
  return 0;
}

/* Build Vandermonde R (order×order) and b from rks[0..order-1]; bh2 predict_x0. */
static void build_R_b(const float *rks, int order, float hh, float B_h,
                      float *R, float *b) {
  float h_phi_1 = expm1f(hh);
  float h_phi_k = h_phi_1 / hh - 1.0f;
  float factorial_i = 1.0f;
  for (int i = 1; i <= order; i++) {
    int row = i - 1;
    for (int j = 0; j < order; j++)
      R[row * order + j] = powf(rks[j], (float)(i - 1));
    b[row] = h_phi_k * factorial_i / B_h;
    factorial_i *= (float)(i + 1);
    h_phi_k = h_phi_k / hh - 1.0f / factorial_i;
  }
}

static int corrector_disabled(const sched_unipc *s, int step_minus_1) {
  for (int i = 0; i < s->n_disable; i++) {
    if (s->disable_corrector[i] == step_minus_1)
      return 1;
  }
  return 0;
}

static int ensure_bufs(sched_unipc *s, size_t n) {
  if (s->buf_n == n && s->m_hist[0] && s->last_sample && s->x0_cur)
    return 0;
  for (int i = 0; i < WAN_UNIPC_ORDER_MAX; i++) {
    free(s->m_hist[i]);
    s->m_hist[i] = NULL;
  }
  for (int i = 0; i < s->solver_order; i++) {
    s->m_hist[i] = calloc(n, sizeof(float));
    if (!s->m_hist[i])
      return -1;
  }
  free(s->last_sample);
  free(s->x0_cur);
  s->last_sample = calloc(n, sizeof(float));
  s->x0_cur = calloc(n, sizeof(float));
  if (!s->last_sample || !s->x0_cur)
    return -1;
  s->buf_n = n;
  s->n_hist = 0;
  s->lower_order_nums = 0;
  s->this_order = 1;
  s->has_last = 0;
  return 0;
}

static void push_m(sched_unipc *s, const float *m, size_t n) {
  int cap = s->solver_order;
  if (s->n_hist < cap) {
    memcpy(s->m_hist[s->n_hist], m, n * sizeof(float));
    s->n_hist++;
    return;
  }
  float *oldest = s->m_hist[0];
  for (int i = 0; i < cap - 1; i++)
    s->m_hist[i] = s->m_hist[i + 1];
  s->m_hist[cap - 1] = oldest;
  memcpy(s->m_hist[cap - 1], m, n * sizeof(float));
}

static int parse_order_env(void) {
  const char *e = getenv("WAN_UNIPC_ORDER");
  if (!e || !*e)
    return WAN_UNIPC_ORDER_MAX;
  int v = atoi(e);
  if (v < 1)
    v = 1;
  if (v > WAN_UNIPC_ORDER_MAX)
    v = WAN_UNIPC_ORDER_MAX;
  return v;
}

static void parse_disable_env(sched_unipc *s) {
  s->n_disable = 0;
  const char *e = getenv("WAN_UNIPC_DISABLE_CORRECTOR");
  if (!e || !*e)
    return;
  const char *p = e;
  while (*p && s->n_disable < WAN_UNIPC_DISABLE_MAX) {
    while (*p == ' ' || *p == ',')
      p++;
    if (!*p)
      break;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p)
      break;
    s->disable_corrector[s->n_disable++] = (int)v;
    p = end;
  }
}

sched_unipc *sched_unipc_create(int steps, float shift) {
  if (steps < 1)
    return NULL;
  sched_unipc *s = calloc(1, sizeof(*s));
  if (!s)
    return NULL;
  s->steps = steps;
  s->shift = shift;
  s->solver_order = parse_order_env();
  s->this_order = 1;
  parse_disable_env(s);
  s->n_sigmas = steps + 1;
  s->sigmas = calloc((size_t)s->n_sigmas, sizeof(float));
  if (!s->sigmas) {
    free(s);
    return NULL;
  }
  const float sigma_min = 1.0f / 1000.0f;
  for (int i = 0; i < steps; i++) {
    float t = 1.0f - (1.0f - sigma_min) * (float)i / (float)steps;
    s->sigmas[i] = sched_unipc_warp_sigma(t, shift);
  }
  s->sigmas[steps] = 0.0f;
  {
    static int logged;
    if (!logged) {
      fprintf(stderr,
              "wan-c: Flow UniPC bh2 order≤%d (shift=%.2f steps=%d"
              " disable_corrector=%d)\n",
              s->solver_order, shift, steps, s->n_disable);
      logged = 1;
    }
  }
  return s;
}

void sched_unipc_destroy(sched_unipc *s) {
  if (!s)
    return;
  free(s->sigmas);
  for (int i = 0; i < WAN_UNIPC_ORDER_MAX; i++)
    free(s->m_hist[i]);
  free(s->last_sample);
  free(s->x0_cur);
  free(s);
}

int sched_unipc_num_sigmas(const sched_unipc *s) {
  return s ? s->n_sigmas : 0;
}

const float *sched_unipc_sigmas(const sched_unipc *s) {
  return s ? s->sigmas : NULL;
}

void sched_unipc_cfg_combine(const float *x_uncond, const float *x_cond,
                             float *x_out, size_t n, float cfg_scale) {
  for (size_t i = 0; i < n; i++)
    x_out[i] = x_uncond[i] + cfg_scale * (x_cond[i] - x_uncond[i]);
}

/* UniC: correct sample at σ_t using last_sample at σ_{t-1}. order = prior this_order. */
static void unipc_correct(sched_unipc *s, int step, int order, size_t n,
                          float *sample) {
  if (order < 1)
    order = 1;
  if (order > s->n_hist)
    order = s->n_hist;
  if (order < 1 || !s->has_last)
    return;

  const float *m0 = s->m_hist[s->n_hist - 1];
  float sigma_t = s->sigmas[step];
  float sigma_s0 = s->sigmas[step - 1];
  float alpha_t = 1.0f - sigma_t;
  float lam_t = sigma_lambda(sigma_t);
  float lam_s0 = sigma_lambda(sigma_s0);
  float h = lam_t - lam_s0;
  if (fabsf(h) < 1e-12f)
    h = (h < 0.f) ? -1e-12f : 1e-12f;
  float hh = -h; /* predict_x0 */
  float h_phi_1 = expm1f(hh);
  float B_h = expm1f(hh); /* bh2 */
  float inv_s0 = 1.0f / fmaxf(sigma_s0, 1e-12f);

  float rks[WAN_UNIPC_ORDER_MAX];
  float rhos[WAN_UNIPC_ORDER_MAX];
  int n_d1 = order - 1;

  for (int i = 1; i < order; i++) {
    int si = step - (i + 1);
    if (si < 0 || s->n_hist < (i + 1)) {
      order = i;
      n_d1 = order - 1;
      break;
    }
    const float *mi = s->m_hist[s->n_hist - (i + 1)];
    float lam_si = sigma_lambda(s->sigmas[si]);
    float rk = clamp_rk((lam_si - lam_s0) / h);
    rks[i - 1] = rk;
    (void)mi;
  }
  rks[order - 1] = 1.0f;

  if (order == 1) {
    rhos[0] = 0.5f;
  } else {
    float R[WAN_UNIPC_ORDER_MAX * WAN_UNIPC_ORDER_MAX];
    float b[WAN_UNIPC_ORDER_MAX];
    build_R_b(rks, order, hh, B_h, R, b);
    if (solve_lin(R, b, rhos, order) != 0) {
      rhos[0] = 0.5f;
      for (int i = 1; i < order; i++)
        rhos[i] = 0.0f;
      order = 1;
      n_d1 = 0;
    }
  }

  for (size_t i = 0; i < n; i++) {
    float x_t_ =
        sigma_t * inv_s0 * s->last_sample[i] - alpha_t * h_phi_1 * m0[i];
    float corr = 0.0f;
    for (int k = 0; k < n_d1; k++) {
      const float *mi = s->m_hist[s->n_hist - (k + 2)];
      float D1 = (mi[i] - m0[i]) / rks[k];
      corr += rhos[k] * D1;
    }
    float D1_t = s->x0_cur[i] - m0[i];
    corr += rhos[order - 1] * D1_t;
    sample[i] = x_t_ - alpha_t * B_h * corr;
  }
}

/* UniP: advance sample from σ_s0 → σ_t. */
static void unipc_predict(sched_unipc *s, int step, int order, size_t n,
                          float *sample) {
  if (order < 1)
    order = 1;
  if (order > s->n_hist)
    order = s->n_hist;

  float sigma_s0 = s->sigmas[step];
  float sigma_t = s->sigmas[step + 1];
  float alpha_t = 1.0f - sigma_t;
  float lam_t = sigma_lambda(sigma_t);
  float lam_s0 = sigma_lambda(sigma_s0);
  float h = lam_t - lam_s0;
  if (fabsf(h) < 1e-12f)
    h = (h < 0.f) ? -1e-12f : 1e-12f;
  float hh = -h;
  float h_phi_1 = expm1f(hh);
  float B_h = expm1f(hh);
  float inv_s0 = 1.0f / fmaxf(sigma_s0, 1e-12f);
  const float *m0 = s->m_hist[s->n_hist - 1];

  if (order < 2) {
    for (size_t i = 0; i < n; i++)
      sample[i] = sigma_t * inv_s0 * sample[i] - alpha_t * h_phi_1 * m0[i];
    return;
  }

  float rks[WAN_UNIPC_ORDER_MAX];
  float rhos[WAN_UNIPC_ORDER_MAX];
  int n_d1 = order - 1;
  int ok = 1;
  for (int i = 1; i < order; i++) {
    int si = step - i;
    if (si < 0 || s->n_hist < (i + 1)) {
      ok = 0;
      break;
    }
    float lam_si = sigma_lambda(s->sigmas[si]);
    rks[i - 1] = clamp_rk((lam_si - lam_s0) / h);
  }
  if (!ok) {
    for (size_t i = 0; i < n; i++)
      sample[i] = sigma_t * inv_s0 * sample[i] - alpha_t * h_phi_1 * m0[i];
    return;
  }
  rks[order - 1] = 1.0f;

  if (order == 2) {
    rhos[0] = 0.5f;
  } else {
    float R[WAN_UNIPC_ORDER_MAX * WAN_UNIPC_ORDER_MAX];
    float b[WAN_UNIPC_ORDER_MAX];
    build_R_b(rks, order, hh, B_h, R, b);
    /* rhos_p = solve(R[:-1,:-1], b[:-1]) */
    float Rs[(WAN_UNIPC_ORDER_MAX - 1) * (WAN_UNIPC_ORDER_MAX - 1)];
    float bs[WAN_UNIPC_ORDER_MAX - 1];
    for (int i = 0; i < n_d1; i++) {
      bs[i] = b[i];
      for (int j = 0; j < n_d1; j++)
        Rs[i * n_d1 + j] = R[i * order + j];
    }
    if (solve_lin(Rs, bs, rhos, n_d1) != 0) {
      for (size_t i = 0; i < n; i++)
        sample[i] = sigma_t * inv_s0 * sample[i] - alpha_t * h_phi_1 * m0[i];
      return;
    }
  }

  for (size_t i = 0; i < n; i++) {
    float x_t_ = sigma_t * inv_s0 * sample[i] - alpha_t * h_phi_1 * m0[i];
    float pred = 0.0f;
    for (int k = 0; k < n_d1; k++) {
      const float *mi = s->m_hist[s->n_hist - (k + 2)];
      float D1 = (mi[i] - m0[i]) / rks[k];
      pred += rhos[k] * D1;
    }
    sample[i] = x_t_ - alpha_t * B_h * pred;
  }
}

int sched_unipc_step(sched_unipc *s, int step, const float *model_out,
                     float *sample, size_t n) {
  if (!s || !model_out || !sample || step < 0 || step >= s->steps || n < 1)
    return -1;
  if (ensure_bufs(s, n) != 0)
    return -1;

  float sigma_s0 = s->sigmas[step];

  /* convert_model_output: x0 = sample - σ · v */
  for (size_t i = 0; i < n; i++)
    s->x0_cur[i] = sample[i] - sigma_s0 * model_out[i];

  /* UniC uses this_order from the previous step. */
  if (step > 0 && s->has_last && s->n_hist >= 1 &&
      !corrector_disabled(s, step - 1)) {
    unipc_correct(s, step, s->this_order, n, sample);
  }

  push_m(s, s->x0_cur, n);
  memcpy(s->last_sample, sample, n * sizeof(float));
  s->has_last = 1;

  int max_order = s->solver_order;
  /* lower_order_final */
  if (max_order > s->steps - step)
    max_order = s->steps - step;
  if (max_order < 1)
    max_order = 1;
  s->this_order = s->lower_order_nums + 1;
  if (s->this_order > max_order)
    s->this_order = max_order;

  unipc_predict(s, step, s->this_order, n, sample);

  if (s->lower_order_nums < s->solver_order)
    s->lower_order_nums++;
  return 0;
}
