#include "h3_clipproj_host.h"
#include "safetensors_min.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MLP_MAX_LAYERS = 4 };

typedef struct {
  float *weight; /* [out, in] row-major (torch Linear) */
  float *bias;   /* [out] or NULL */
  int in_f;
  int out_f;
} clip_linear;

struct h3_clipproj {
  int din;
  int dout;
  float *W; /* [din, dout] */
  float *mean_in;
  float *std_in;
  float *mean_out;
  float *std_out;
  float *sink_out; /* optional [dout] */
  clip_linear mlp[MLP_MAX_LAYERS];
  int n_mlp;
};

static void fail(char *error, size_t error_size, const char *msg) {
  if (error && error_size)
    snprintf(error, error_size, "%s", msg);
}

static int load_vec(st_file *sf, const char *name, float **dst, int expect_n,
                    char *error, size_t error_size) {
  const st_tensor_t *t = st_find_tensor(sf, name);
  if (!t) {
    fail(error, error_size, "clipproj: missing tensor");
    return -1;
  }
  size_t n = st_tensor_nelems(t);
  if (expect_n > 0 && (int)n != expect_n) {
    fail(error, error_size, "clipproj: bad vector length");
    return -1;
  }
  float *buf = malloc(n * sizeof(float));
  if (!buf || st_tensor_to_f32(sf, t, buf, n) != 0) {
    free(buf);
    fail(error, error_size, "clipproj: decode failed");
    return -1;
  }
  *dst = buf;
  return (int)n;
}

static int load_matrix_W(st_file *sf, float **dst, int *din, int *dout,
                         char *error, size_t error_size) {
  const st_tensor_t *t = st_find_tensor(sf, "W");
  if (!t || t->ndim != 2) {
    fail(error, error_size, "clipproj: W missing or not 2-D");
    return -1;
  }
  *din = (int)t->shape[0];
  *dout = (int)t->shape[1];
  size_t n = (size_t)(*din) * (size_t)(*dout);
  float *buf = malloc(n * sizeof(float));
  if (!buf || st_tensor_to_f32(sf, t, buf, n) != 0) {
    free(buf);
    fail(error, error_size, "clipproj: W decode failed");
    return -1;
  }
  *dst = buf;
  return 0;
}

static int try_load_mlp_layer(st_file *sf, int index, clip_linear *layer) {
  char wname[64], bname[64];
  snprintf(wname, sizeof(wname), "mlp.%d.weight", index);
  snprintf(bname, sizeof(bname), "mlp.%d.bias", index);
  const st_tensor_t *tw = st_find_tensor(sf, wname);
  if (!tw || tw->ndim != 2)
    return 0;
  layer->out_f = (int)tw->shape[0];
  layer->in_f = (int)tw->shape[1];
  size_t n = (size_t)layer->out_f * (size_t)layer->in_f;
  layer->weight = malloc(n * sizeof(float));
  if (!layer->weight || st_tensor_to_f32(sf, tw, layer->weight, n) != 0) {
    free(layer->weight);
    layer->weight = NULL;
    return -1;
  }
  const st_tensor_t *tb = st_find_tensor(sf, bname);
  if (tb) {
    size_t bn = st_tensor_nelems(tb);
    layer->bias = malloc(bn * sizeof(float));
    if (!layer->bias || st_tensor_to_f32(sf, tb, layer->bias, bn) != 0) {
      free(layer->weight);
      free(layer->bias);
      layer->weight = layer->bias = NULL;
      return -1;
    }
  } else {
    layer->bias = NULL;
  }
  return 1;
}

static void gelu_inplace(float *x, size_t n) {
  /* PyTorch GELU (tanh approx). */
  for (size_t i = 0; i < n; i++) {
    float v = x[i];
    float u = 0.7978845608f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(u));
  }
}

static void linear_rows(float *dst, const float *src, int rows, const clip_linear *L) {
  /* dst[r,:] = src[r,:] @ W^T + b  with W [out,in] */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, L->out_f, L->in_f,
              1.0f, src, L->in_f, L->weight, L->in_f, 0.0f, dst, L->out_f);
  if (L->bias) {
    for (int r = 0; r < rows; r++) {
      float *row = dst + (size_t)r * (size_t)L->out_f;
      for (int c = 0; c < L->out_f; c++)
        row[c] += L->bias[c];
    }
  }
}

h3_clipproj *h3_clipproj_load(const char *path, char *error, size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!path || !path[0]) {
    fail(error, error_size, "clipproj: empty path");
    return NULL;
  }
  st_file *sf = st_open(path);
  if (!sf) {
    fail(error, error_size, "clipproj: cannot open safetensors");
    return NULL;
  }
  h3_clipproj *p = calloc(1, sizeof(*p));
  if (!p) {
    st_close(sf);
    return NULL;
  }
  if (load_matrix_W(sf, &p->W, &p->din, &p->dout, error, error_size) != 0)
    goto fail;
  if (load_vec(sf, "mean_in", &p->mean_in, p->din, error, error_size) < 0)
    goto fail;
  if (load_vec(sf, "std_in", &p->std_in, p->din, error, error_size) < 0)
    goto fail;
  if (load_vec(sf, "mean_out", &p->mean_out, p->dout, error, error_size) < 0)
    goto fail;
  if (load_vec(sf, "std_out", &p->std_out, p->dout, error, error_size) < 0)
    goto fail;
  {
    const st_tensor_t *ts = st_find_tensor(sf, "sink_out");
    if (ts) {
      if (load_vec(sf, "sink_out", &p->sink_out, p->dout, error, error_size) < 0)
        goto fail;
    }
  }
  for (int i = 0; i < 8 && p->n_mlp < MLP_MAX_LAYERS; i++) {
    clip_linear layer = {0};
    int rc = try_load_mlp_layer(sf, i, &layer);
    if (rc < 0)
      goto fail;
    if (rc == 0)
      continue;
    p->mlp[p->n_mlp++] = layer;
  }
  st_close(sf);
  return p;

fail:
  st_close(sf);
  h3_clipproj_free(p);
  return NULL;
}

void h3_clipproj_free(h3_clipproj *proj) {
  if (!proj)
    return;
  free(proj->W);
  free(proj->mean_in);
  free(proj->std_in);
  free(proj->mean_out);
  free(proj->std_out);
  free(proj->sink_out);
  for (int i = 0; i < proj->n_mlp; i++) {
    free(proj->mlp[i].weight);
    free(proj->mlp[i].bias);
  }
  free(proj);
}

int h3_clipproj_din(const h3_clipproj *proj) { return proj ? proj->din : 0; }
int h3_clipproj_dout(const h3_clipproj *proj) { return proj ? proj->dout : 0; }
int h3_clipproj_has_sink(const h3_clipproj *proj) {
  return proj && proj->sink_out ? 1 : 0;
}
int h3_clipproj_has_mlp(const h3_clipproj *proj) {
  return proj && proj->n_mlp > 0 ? 1 : 0;
}

int h3_clipproj_apply_affine(const float *hidden, int seq, int din, int dout,
                             const float *W, const float *mean_in,
                             const float *std_in, const float *mean_out,
                             const float *std_out, const float *sink_out,
                             float *cond_out) {
  if (!hidden || !W || !mean_in || !std_in || !mean_out || !std_out ||
      !cond_out || seq < 1 || din < 1 || dout < 1)
    return -1;
  float *xn = malloc((size_t)seq * (size_t)din * sizeof(float));
  float *yn = malloc((size_t)seq * (size_t)dout * sizeof(float));
  if (!xn || !yn) {
    free(xn);
    free(yn);
    return -1;
  }
  for (int s = 0; s < seq; s++) {
    const float *h = hidden + (size_t)s * (size_t)din;
    float *x = xn + (size_t)s * (size_t)din;
    for (int i = 0; i < din; i++) {
      float den = std_in[i];
      if (fabsf(den) < 1e-12f)
        den = 1e-12f;
      x[i] = (h[i] - mean_in[i]) / den;
    }
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq, dout, din, 1.0f,
              xn, din, W, dout, 0.0f, yn, dout);
  for (int s = 0; s < seq; s++) {
    float *y = yn + (size_t)s * (size_t)dout;
    float *o = cond_out + (size_t)s * (size_t)dout;
    for (int j = 0; j < dout; j++)
      o[j] = y[j] * std_out[j] + mean_out[j];
  }
  if (sink_out && seq > 0)
    memcpy(cond_out, sink_out, (size_t)dout * sizeof(float));
  free(xn);
  free(yn);
  return 0;
}

int h3_clipproj_apply(const h3_clipproj *proj, const float *hidden, int seq,
                      float *cond_out, char *error, size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!proj || !hidden || !cond_out || seq < 1) {
    fail(error, error_size, "clipproj_apply: bad args");
    return -1;
  }
  int din = proj->din, dout = proj->dout;
  float *xn = malloc((size_t)seq * (size_t)din * sizeof(float));
  float *yn = malloc((size_t)seq * (size_t)dout * sizeof(float));
  if (!xn || !yn) {
    free(xn);
    free(yn);
    fail(error, error_size, "clipproj_apply: OOM");
    return -1;
  }
  for (int s = 0; s < seq; s++) {
    const float *h = hidden + (size_t)s * (size_t)din;
    float *x = xn + (size_t)s * (size_t)din;
    for (int i = 0; i < din; i++) {
      float den = proj->std_in[i];
      if (fabsf(den) < 1e-12f)
        den = 1e-12f;
      x[i] = (h[i] - proj->mean_in[i]) / den;
    }
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq, dout, din, 1.0f,
              xn, din, proj->W, dout, 0.0f, yn, dout);

  if (proj->n_mlp > 0) {
    /* Residual in standardized space: yn += mlp(xn) */
    float *act = NULL;
    int act_dim = 0;
    const float *cur = xn;
    int cur_dim = din;
    for (int li = 0; li < proj->n_mlp; li++) {
      const clip_linear *L = &proj->mlp[li];
      if (L->in_f != cur_dim) {
        free(xn);
        free(yn);
        free(act);
        fail(error, error_size, "clipproj_apply: mlp width mismatch");
        return -1;
      }
      float *next = malloc((size_t)seq * (size_t)L->out_f * sizeof(float));
      if (!next) {
        free(xn);
        free(yn);
        free(act);
        fail(error, error_size, "clipproj_apply: OOM mlp");
        return -1;
      }
      linear_rows(next, cur, seq, L);
      if (li + 1 < proj->n_mlp)
        gelu_inplace(next, (size_t)seq * (size_t)L->out_f);
      free(act);
      act = next;
      cur = next;
      cur_dim = L->out_f;
      act_dim = L->out_f;
    }
    if (act_dim != dout) {
      free(xn);
      free(yn);
      free(act);
      fail(error, error_size, "clipproj_apply: mlp out dim mismatch");
      return -1;
    }
    for (size_t i = 0; i < (size_t)seq * (size_t)dout; i++)
      yn[i] += act[i];
    free(act);
  }

  for (int s = 0; s < seq; s++) {
    float *y = yn + (size_t)s * (size_t)dout;
    float *o = cond_out + (size_t)s * (size_t)dout;
    for (int j = 0; j < dout; j++)
      o[j] = y[j] * proj->std_out[j] + proj->mean_out[j];
  }
  if (proj->sink_out && seq > 0)
    memcpy(cond_out, proj->sink_out, (size_t)dout * sizeof(float));
  free(xn);
  free(yn);
  return 0;
}
