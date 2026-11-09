# Checkout housekeeping

This doc holds **local checkout cleanup rules** that do not belong in the forward-looking roadmap.

Use it when you are deciding whether a local leftover should stay untracked, be deleted, or remain in git.

## Leave untracked

These are normal local leftovers unless a change explicitly promotes them into source control:

- Old `vendor/llama-cpp-*` trees that are not the current tracked pin
- Local Python envs such as `venv-training/`
- Downloaded model weights, GGUF scratch outputs, and one-off lab artifacts
- Local binaries, temp logs, and machine-specific scratch files

## Already tracked: leave in git

Do not delete or "clean up" these just because they look large or vendor-like:

- The current tracked `vendor/llama-cpp-*` pin selected by the repo
- Tracked patch series, build scripts, and pin-ladder docs
- Any file that `git ls-files --error-unmatch <path>` reports as tracked

## Working rule

If a path is **tracked**, treat it as repo state.

If a path is **untracked** and looks like a local cache, old vendor checkout, virtualenv, or lab output, it should usually stay **untracked** unless there is an explicit task to promote it.
