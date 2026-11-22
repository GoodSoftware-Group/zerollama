# Archived quilt for ggml-org b12119

These 163 format-patches (`0001`–`0163`) apply on tag **b12119** (`6f1e6543704bc5b080c7a98e7b6f707b31f6ed51`). They do not apply on the live pin.

The live series is `llama/patches/` `0001`–`0187`, formatted on **b12557** (`b94959a64d011915d77a5381c7d4846dcda99530`). That series started as this quilt rebased commit-by-commit onto the tag, including the gemma4-assistant draft-stack load (`hparams.n_layer_nextn = hparams.n_layer_all`). Two no-op commits were dropped. This quilt's **0162** (`#30541`) is upstream on b12557 and is not the live **0162** (the prompt-mark grid). Live **0162–0186** are prompt reuse. Upstream mtmd Kokoro and tiny-voca are in the tag.
