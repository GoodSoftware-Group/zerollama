"""T10/T10b GRPO payload validation (no GPU / no trl required)."""

from __future__ import annotations

import http.server
import json
import threading
import unittest

from training_grpo import (
    _shield_sibling_mlx_false_positive,
    describe_reward,
    format_json_reward,
    has_reward_signal,
    length_reward,
    make_exact_match_reward,
    nonempty_reward,
    normalize_grpo_prompts,
    normalize_grpo_references,
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

    def test_normalize_references(self):
        refs = normalize_grpo_references(
            {
                "training_data": [
                    {"prompt": "p", "answer": "a"},
                    {"prompt": "p2", "label": "b"},
                ]
            }
        )
        self.assertEqual(refs, ["a", "b"])
        self.assertEqual(
            normalize_grpo_references({"references": ["x", "y"]}), ["x", "y"]
        )

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

    def test_builtins_length_nonempty_json(self):
        self.assertEqual(length_reward(["ab", ""]), [2.0, 0.0])
        self.assertEqual(nonempty_reward(["x", "  "]), [1.0, 0.0])
        self.assertEqual(format_json_reward(['{"a":1}', "nope"]), [1.0, 0.0])

    def test_exact_match_builtin(self):
        payload = {
            "model_name": "x",
            "prompts": ["p1", "p2"],
            "references": ["yes", "no"],
            "reward_builtin": "exact_match",
        }
        self.assertEqual(validate_grpo_payload(payload), [])
        func, err = resolve_reward_func(payload)
        self.assertIsNone(err)
        self.assertEqual(func(["yes", "maybe"]), [1.0, 0.0])

    def test_exact_match_len_mismatch(self):
        errs = validate_grpo_payload(
            {
                "model_name": "x",
                "prompts": ["p1", "p2"],
                "references": ["only_one"],
                "reward_builtin": "em",
            }
        )
        self.assertTrue(any("len(references)" in e for e in errs))

    def test_reward_hf_dry_run(self):
        res = process_grpo_request(
            {
                "model_name": "sshleifer/tiny-gpt2",
                "prompts": ["hello"],
                "reward_hf": "OpenAssistant/reward-model-deberta-v3-large-v2",
            }
        )
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["dry_run"])
        self.assertIn("hf:", res["reward"])

    def test_reward_url_dry_run_and_call(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                rewards = [float(len(c)) for c in body["completions"]]
                raw = json.dumps({"rewards": rewards}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}/reward"
            payload = {
                "model_name": "x",
                "prompts": ["p"],
                "reward_url": url,
            }
            self.assertEqual(validate_grpo_payload(payload), [])
            self.assertEqual(describe_reward(payload), "url")
            res = process_grpo_request(payload)
            self.assertEqual(res["status"], "ok")
            self.assertTrue(res["dry_run"])
            func, err = resolve_reward_func(payload)
            self.assertIsNone(err)
            self.assertEqual(func(["abc"]), [3.0])
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2)

    def test_reward_url_rejects_file(self):
        errs = validate_grpo_payload(
            {
                "model_name": "x",
                "prompts": ["p"],
                "reward_url": "file:///tmp/x",
            }
        )
        self.assertTrue(any("http(s)" in e for e in errs))

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
        # WHY mock: with trl installed, grpo_run would start Trainer / HF download —
        # host unit must stay offline. Assert the hard-fail path only.
        from unittest.mock import patch

        with patch(
            "training_grpo.trl_grpo_available",
            return_value=(False, "trl not installed (test mock)"),
        ):
            res = process_grpo_request(
                {
                    "model_name": "sshleifer/tiny-gpt2",
                    "prompts": ["hello"],
                    "reward_builtin": "length",
                    "grpo_run": True,
                }
            )
        self.assertEqual(res["status"], "error")
        self.assertFalse(res.get("trained", False))
        self.assertIn("trl", (res.get("error") or "").lower())

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

    def test_make_exact_match_direct(self):
        fn = make_exact_match_reward(["a", "b"])
        self.assertEqual(fn(["a", "x"]), [1.0, 0.0])

    def test_shield_sibling_mlx_is_callable(self):
        # Host unit only — must not raise even when sibling mlx/ exists.
        _shield_sibling_mlx_false_positive()


if __name__ == "__main__":
    unittest.main()
