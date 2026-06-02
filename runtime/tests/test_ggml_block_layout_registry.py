"""Regression: _GGML_BLOCK_LAYOUT matches ggml-common.h block sizes (QK_K=256)."""

from runtime.gguf_estimate import _GGML_BLOCK_LAYOUT

# (block_size, type_size) — keep in sync with ml/backend/ggml/ggml/src/ggml-common.h
_EXPECTED_FROM_GGML_COMMON: dict[int, tuple[int, int]] = {
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    29: (256, 56),  # IQ1_M
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
}


def test_iq_tq_block_layouts_match_ggml_common():
    for type_id, expected in _EXPECTED_FROM_GGML_COMMON.items():
        assert _GGML_BLOCK_LAYOUT.get(type_id) == expected, f"type {type_id}"
