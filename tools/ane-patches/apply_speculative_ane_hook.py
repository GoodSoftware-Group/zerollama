#!/usr/bin/env python3
"""Apply ANE draft hook call sites to llama.cpp common/ (idempotent, fails loud on drift)."""
from __future__ import annotations

import pathlib
import sys


def patch_once(path: pathlib.Path, needle: str, repl: str, label: str, *, required: bool = True) -> None:
    text = path.read_text()
    if repl.strip() in text or repl in text:
        print(f"  skip {label} (already applied)")
        return
    if needle not in text:
        if not required:
            print(f"  skip {label} (anchor missing — optional)")
            return
        raise SystemExit(f"anchor missing for {label} in {path}")
    path.write_text(text.replace(needle, repl, 1))
    print(f"  patched {label}")


def patch_draft_simple_ctor(spec: pathlib.Path) -> None:
    text = spec.read_text()
    if "common_ane_draft_log_init" in text:
        print("  skip speculative.cpp draft_simple ctor (already applied)")
        return
    needle = (
        '            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");\n'
        "        }\n"
        "    }"
    )
    repl = (
        '            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");\n'
        "        }\n\n"
        "        common_ane_draft_log_init(type, llama_model_n_embd(llama_get_model(ctx_dft)));\n\n"
        "        // Why pre-norm: ANE handoff reads llama_get_embeddings_pre_norm_ith (see ane_draft_hook.cpp).\n"
        "        if (common_ane_draft_enabled()) {\n"
        "            llama_set_embeddings_pre_norm(ctx_dft, true);\n"
        "        }\n"
        "    }"
    )
    patch_once(spec, needle, repl, "speculative.cpp draft_simple ctor")


def patch_draft_simple_draft(spec: pathlib.Path) -> None:
    patch_once(
        spec,
        "    void draft(common_speculative_draft_params_vec & dparams) override {\n        auto & ctx_dft = params.ctx_dft;\n\n        common_batch_clear(batch);",
        "    void draft(common_speculative_draft_params_vec & dparams) override {\n        auto & ctx_dft = params.ctx_dft;\n\n        common_ane_draft_reset_handoff();\n\n        common_batch_clear(batch);",
        "speculative.cpp draft() reset (B7)",
        required=False,
    )

    patch_once(
        spec,
        (
            "        int ret = llama_decode(ctx_dft, batch);\n"
            "        if (ret != 0) {\n"
            '            LOG_WRN("%s: llama_decode returned %d\\n", __func__, ret);\n'
            "            return;\n"
            "        }\n\n"
            "        int i = 0;\n\n"
            "        while (n_drafting > 0) {"
        ),
        (
            "        int ret = llama_decode(ctx_dft, batch);\n"
            "        if (ret != 0) {\n"
            '            LOG_WRN("%s: llama_decode returned %d\\n", __func__, ret);\n'
            "            return;\n"
            "        }\n\n"
            "        // B2/B5: IOSurface handoff after first draft decode (lab; tokens still Metal).\n"
            "        common_ane_draft_handoff_after_decode(ctx_dft, 0);\n\n"
            "        int i = 0;\n\n"
            "        while (n_drafting > 0) {"
        ),
        "speculative.cpp draft() handoff (B2)",
    )

    b7_needle = (
        "                // add drafted token for each sequence\n"
        "                const llama_token id = cur_p->data[0].id;\n\n"
        "                // only collect very high-confidence draft tokens\n"
        "                if (cur_p->data[0].p < params.p_min) {"
    )
    b7_repl = (
        "                llama_token id = cur_p->data[0].id;\n"
        "                float pick_p = cur_p->data[0].p;\n\n"
        "                // B7: optional ANE tied-embed argmax replaces Metal sampler token (lab).\n"
        "                const common_ane_draft_drive_mode drive = common_ane_draft_get_drive_mode();\n"
        "                if (drive != COMMON_ANE_DRAFT_DRIVE_OFF) {\n"
        "                    llama_token ane_id = 0;\n"
        "                    float ane_p = 0.f;\n"
        "                    if (common_ane_draft_try_drive_token(ctx_dft, i_batch - 1, &ane_id, &ane_p)) {\n"
        "                        if (drive == COMMON_ANE_DRAFT_DRIVE_SHADOW) {\n"
        '                            LOG_INF("%s: B7 shadow step=%d seq=%d ane_tok=%d metal_tok=%d match=%d\\n",\n'
        "                                    __func__, i, (int) seq_id, (int) ane_id, (int) id, ane_id == id ? 1 : 0);\n"
        "                        } else {\n"
        "                            id = ane_id;\n"
        "                            pick_p = ane_p;\n"
        '                            LOG_DBG("%s: B7 force seq=%d ane_tok=%d\\n", __func__, (int) seq_id, (int) ane_id);\n'
        "                        }\n"
        "                    }\n"
        "                }\n\n"
        "                // only collect very high-confidence draft tokens\n"
        "                if (pick_p < params.p_min) {"
    )
    text = spec.read_text()
    if "B7 shadow step=" not in text:
        patch_once(spec, b7_needle, b7_repl, "speculative.cpp B7 drive token (shadow/force)")
    elif "pick_p < params.p_min" not in text and "cur_p->data[0].p < params.p_min" in text:
        patch_once(spec, b7_needle, b7_repl, "speculative.cpp B7 drive token (shadow/force)")

    patch_once(
        spec,
        (
            "            // evaluate the drafted tokens on the draft model\n"
            "            ret = llama_decode(ctx_dft, batch);\n"
            "            if (ret != 0) {\n"
            '                LOG_WRN("%s: llama_decode[%d] returned %d\\n", __func__, i, ret);\n'
            "                break;\n"
            "            }\n\n"
            "            ++i;\n"
        ),
        (
            "            // evaluate the drafted tokens on the draft model\n"
            "            ret = llama_decode(ctx_dft, batch);\n"
            "            if (ret != 0) {\n"
            '                LOG_WRN("%s: llama_decode[%d] returned %d\\n", __func__, i, ret);\n'
            "                break;\n"
            "            }\n\n"
            "            // B5: handoff on each speculative draft decode step (stride via env in hook).\n"
            "            common_ane_draft_handoff_after_decode(ctx_dft, 0);\n\n"
            "            ++i;\n"
        ),
        "speculative.cpp inner draft loop handoff (B5)",
        required=False,
    )


def patch_cmake(cmake: pathlib.Path) -> None:
    if "ane_draft_hook.cpp" not in cmake.read_text():
        patch_once(
            cmake,
            "    arg.h\n",
            "    arg.h\n    ane_draft_hook.cpp\n    ane_draft_hook.h\n",
            "CMakeLists.txt hook sources",
        )

    anchor = "target_link_libraries(${TARGET} PUBLIC llama Threads::Threads)"
    block = """
# B1: in-process ANE draft session (Darwin + libane_bridge only).
set(_ZEROLLAMA_ANE_REPO "$ENV{HOME}/Sites/inference/ane")
if (DEFINED ENV{ANE_REPO})
    set(_ZEROLLAMA_ANE_REPO "$ENV{ANE_REPO}")
endif()
set(_ZEROLLAMA_ANE_BRIDGE "${_ZEROLLAMA_ANE_REPO}/bridge/libane_bridge.dylib")

if (APPLE AND EXISTS "${_ZEROLLAMA_ANE_BRIDGE}")
    target_sources(${TARGET} PRIVATE
        ane_draft_session.mm
        ane_draft_session.h
        ane_iosurface_map.h)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_ANE_DRAFT_SESSION=1)
    target_include_directories(${TARGET} PRIVATE "${_ZEROLLAMA_ANE_REPO}/bridge")
    target_link_directories(${TARGET} PRIVATE "${_ZEROLLAMA_ANE_REPO}/bridge")
    target_link_libraries(${TARGET} PRIVATE
        ane_bridge
        "-framework Foundation"
        "-framework IOSurface"
        "-framework Metal")
    message(STATUS "llama-common: ANE draft session enabled (${_ZEROLLAMA_ANE_BRIDGE})")
else()
    target_sources(${TARGET} PRIVATE
        ane_draft_session_stub.cpp
        ane_draft_session.h)
    if (APPLE)
        message(STATUS "llama-common: ANE draft session stub (missing ${_ZEROLLAMA_ANE_BRIDGE})")
    endif()
endif()
"""
    text = cmake.read_text()
    if "ANE draft session" not in text:
        if anchor not in text:
            raise SystemExit(f"CMake anchor not found in {cmake}")
        cmake.write_text(text.replace(anchor, anchor + block, 1))
        print("  patched CMakeLists.txt (ANE session)")


def verify(spec: pathlib.Path) -> None:
    text = spec.read_text()
    missing = []
    for marker in (
        "common_ane_draft_handoff_after_decode",
        "common_ane_draft_try_drive_token",
        "B7 shadow step=",
        "common_ane_draft_reset_handoff",
    ):
        if marker not in text:
            missing.append(marker)
    if missing:
        raise SystemExit(f"speculative.cpp missing ANE markers after apply: {', '.join(missing)}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <llama.cpp-root-or-common-dir>")
    root = pathlib.Path(sys.argv[1])
    common = root / "common" if (root / "common" / "speculative.cpp").exists() else root
    spec = common / "speculative.cpp"
    cmake = common / "CMakeLists.txt"
    if not spec.is_file():
        raise SystemExit(f"missing {spec}")

    patch_once(
        spec,
        '#include "speculative.h"\n',
        '#include "speculative.h"\n\n#include "ane_draft_hook.h"\n',
        "speculative.cpp include",
    )
    patch_draft_simple_ctor(spec)
    patch_draft_simple_draft(spec)
    if cmake.is_file():
        patch_cmake(cmake)
    verify(spec)
    print("  OK: speculative ANE hook markers verified")


if __name__ == "__main__":
    main()
