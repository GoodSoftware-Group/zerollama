// Laya typed-decision model: ModernBERT-style encoder + RLCD decision head.
// Encoder graph matches modern-bert; head tensors are loaded for CPU post (server).

#include "models.h"

void llama_model_laya::load_arch_hparams(llama_model_loader & ml) {
    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (found_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        uint32_t swa_period = 3;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        hparams.set_swa_pattern(swa_period, true);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    hparams.llm_ffn_op = LLM_FFN_GEGLU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        hparams.llm_ffn_op = llm_ffn_op_type_from_string(hidden_act, LLM_FFN_GEGLU);
    }

    ml.get_key("laya.head_layers", hparams.n_laya_head_layers, false);
    ml.get_key("laya.n_act",       hparams.n_laya_act,         false);
    if (hparams.n_laya_head_layers == 0) {
        hparams.n_laya_head_layers = 2;
    }
    if (hparams.n_laya_act == 0) {
        hparams.n_laya_act = 2;
    }

    switch (hparams.n_layer()) {
        case 12:
            type = LLM_TYPE_47M; break;
        case 22:
            type = LLM_TYPE_149M; break;
        case 28:
            type = LLM_TYPE_395M; break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_laya::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        if (i != 0) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        }

        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd}, 0);
        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, 2 * n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
    }

    // Decision head (CPU post in llama-server /v1/decisions)
    type_embd = create_tensor(tn(LLM_TENSOR_LAYA_TYPE_EMBD, "weight"), {n_embd, 3}, 0);

    const int n_laya_head = (int) hparams.n_laya_head_layers;
    for (int i = 0; i < n_laya_head; ++i) {
        auto & layer = layers[i]; // reuse early layer slots for optional bias storage via wo_b etc.
        // Dedicated Laya head tensors (not encoder layers) — store via get_tensor names;
        // also keep aliases on unused encoder-layer optional fields when i < n_layer.
        (void) layer;
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_ATTN_NORM, "weight", i), {n_embd}, 0);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_ATTN_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_ATTN_QKV,  "weight", i), {n_embd, 3 * n_embd}, 0);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_ATTN_QKV,  "bias",   i), {3 * n_embd}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_ATTN_OUT,  "weight", i), {n_embd, n_embd}, 0);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_ATTN_OUT,  "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_FFN_NORM,  "weight", i), {n_embd}, 0);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_FFN_NORM,  "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_FFN_UP,    "weight", i), {n_embd, 4 * n_embd}, 0);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_FFN_UP,    "bias",   i), {4 * n_embd}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_FFN_DOWN,  "weight", i), {4 * n_embd, n_embd}, 0);
        create_tensor(tn(LLM_TENSOR_LAYA_HEAD_FFN_DOWN,  "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
    }

    create_tensor(tn(LLM_TENSOR_LAYA_SCORER_NORM, "weight"), {n_embd}, 0);
    create_tensor(tn(LLM_TENSOR_LAYA_SCORER_NORM, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);
    create_tensor(tn(LLM_TENSOR_LAYA_SCORER_FC1,  "weight"), {n_embd, n_embd}, 0);
    create_tensor(tn(LLM_TENSOR_LAYA_SCORER_FC1,  "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);
    create_tensor(tn(LLM_TENSOR_LAYA_SCORER_FC2,  "weight"), {n_embd, 1}, 0);
    create_tensor(tn(LLM_TENSOR_LAYA_SCORER_FC2,  "bias"),   {1}, TENSOR_NOT_REQUIRED);

    const int64_t n_act = (int64_t) hparams.n_laya_act;
    create_tensor(tn(LLM_TENSOR_LAYA_ACT_FC1, "weight"), {n_embd + 4, 256}, 0);
    create_tensor(tn(LLM_TENSOR_LAYA_ACT_FC1, "bias"),   {256}, TENSOR_NOT_REQUIRED);
    create_tensor(tn(LLM_TENSOR_LAYA_ACT_FC2, "weight"), {256, n_act}, 0);
    create_tensor(tn(LLM_TENSOR_LAYA_ACT_FC2, "bias"),   {n_act}, TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_laya::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Encoder-only graph (same as modern-bert). Decision head applied in server CPU post.
llama_model_laya::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * inp_pos = build_inp_pos();

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    inpL = build_norm(inpL, model.tok_norm, nullptr, LLM_NORM, 0);
    cb(inpL, "inp_norm", 0);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_no_cache();

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        cur = inpL;

        if (model.layers[il].attn_norm) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);
        }

        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        cb(cur, "kqv_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL,                      NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                hparams.llm_ffn_op,
                LLM_FFN_SEQ, il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM, -1);
    cb(cur, "final_norm_out", -1);

    // Per-token encoder states for CPU decision head (pooling=none).
    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
