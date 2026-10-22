"""T10 GRPO payload validation (no GPU / no trl required)."""

from __future__ import annotations

import unittest

from training_grpo import (
    has_reward_signal,
    normalize_grpo_prompts,
    process_grpo_request,
    resolve_reward_func,
    validate_grpo_payload,
)


class TestTrainingGrpo(unittest.TestCase):
    def test_normalize_prompts_from_training_data(self):
        prompts = normalize_grpo_prompts(
            {
                "training_data": [
                    {"prompt": "hi"},
                    "plain",
                    {"question": "q2"},
                    {"prompt": "  "},
                ]
            }
        )
        self.assertEqual(prompts, ["hi", "plain", "q2"])

    def test_validate_requires_model_prompts_reward(self):
        errs = validate_grpo_payload({})
        self.assertTrue(any("model_name" in e for e in errs))
        self.assertTrue(any("prompts" in e for e in errs))
        self.assertTrue(any("reward" in e for e in errs))

    def test_validate_ok_minimal(self):
        errs = validate_grpo_payload(
            {
                "model_name": "sshleifer/tiny-gpt2",
                "prompts": ["say hi"],
                "reward_builtin": "length",
            }
        )
        self.assertEqual(errs, [])
        self.assertTrue(has_reward_signal({"reward_builtin": "length"}))

    def test_json_reward_funcs_rejected(self):
        errs = validate_grpo_payload(
            {
                "model_name": "x",
                "prompts": ["p"],
                "reward_funcs": "not_a_callable",
            }
        )
        self.assertTrue(any("cannot be sent over JSON" in e for e in errs))

    def test_inprocess_callable_ok(self):
        def r(completions, **kwargs):
            return [1.0] * len(completions)

        func, err = resolve_reward_func({"reward_funcs": r})
        self.assertIsNone(err)
        self.assertTrue(callable(func))

    def test_process_grpo_validation_error(self):
        res = process_grpo_request({"model_name": "x"})
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["kind"], "grpo")
        self.assertIn("prompts", res["error"])

    def test_dry_run_ok_without_trl(self):
        res = process_grpo_request(
            {
                "model_name": "sshleifer/tiny-gpt2",
                "prompts": ["hello"],
                "reward_builtin": "length",
            }
        )
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["dry_run"])
        self.assertFalse(res["trained"])
        self.assertIn("no training performed", res["message"])

    def test_grpo_run_requires_trl(self):
        res = process_grpo_request(
            {
                "model_name": "sshleifer/tiny-gpt2",
                "prompts": ["hello"],
                "reward_builtin": "length",
                "grpo_run": True,
            }
        )
        # Host has no trl — must hard-fail, not dry-run.
        if res.get("trl") is False or "trl" in (res.get("error") or "").lower():
            self.assertEqual(res["status"], "error")
            self.assertFalse(res.get("trained", False))
        else:
            # trl present: either runs or fails inside Trainer — not a dry_run.
            self.assertFalse(res.get("dry_run", False))

    def test_run_alias_ignored(self):
        res = process_grpo_request(
            {
                "model_name": "sshleifer/tiny-gpt2",
                "prompts": ["hello"],
                "reward_builtin": "length",
                "run": True,  # must NOT start Trainer
            }
        )
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["dry_run"])
        self.assertFalse(res["trained"])


if __name__ == "__main__":
    unittest.main()
