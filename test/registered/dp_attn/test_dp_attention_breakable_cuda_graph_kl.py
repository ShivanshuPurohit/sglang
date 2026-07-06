import inspect
import random
import unittest

import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    get_input_ids,
)
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=800, stage="base-b", runner_config="2-gpu-large")


MODEL = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
KL_THRESHOLD = 0.0025
NUM_SAMPLES = 48
MAX_PROMPT_TOKENS = 512
MAX_NEW_TOKENS = 128
REQUEST_BATCH_SIZE = 4


def _compute_kl(input_logprobs, output_logprobs):
    kl_divs = []
    for idx, (input_logprob, output_logprob) in enumerate(
        zip(input_logprobs, output_logprobs)
    ):
        input_has_none = any(value is None for value in input_logprob)
        output_has_none = any(value is None for value in output_logprob)
        if input_has_none or output_has_none:
            source = "input" if input_has_none else "output"
            if input_has_none and output_has_none:
                source = "input and output"
            print(f"WARNING: sample {idx}: skipping due to None in {source} logprobs")
            continue

        logr = np.array(input_logprob) - np.array(output_logprob)
        kl_divs.append(float(np.mean((np.exp(logr) - 1) - logr)))

    assert kl_divs, "No valid KL samples"
    avg_kl_div = sum(kl_divs) / len(kl_divs)
    print(f"per-sample KL: {kl_divs}")
    print(f"avg KL: {avg_kl_div:.6f}")
    print(f"KL threshold: {KL_THRESHOLD:.6f}")
    return avg_kl_div


class TestDPAttentionBreakableCudaGraphKL(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--cuda-graph-backend-prefill=breakable",
            ],
        )
        random.seed(42)
        cls.input_ids = get_input_ids(
            cls.model,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            num_samples=NUM_SAMPLES,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _assert_results(self, results, expected_len):
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), expected_len)
        for result in results:
            self.assertIn("meta_info", result)
            self.assertIn("output_ids", result)

    def _generate_in_batches(self, input_ids):
        results = []
        num_requests = (len(input_ids) + REQUEST_BATCH_SIZE - 1) // REQUEST_BATCH_SIZE
        print(
            "Generating decode logprobs: "
            f"samples={len(input_ids)}, requests={num_requests}, "
            f"request_batch_size={REQUEST_BATCH_SIZE}, "
            f"max_new_tokens={MAX_NEW_TOKENS}"
        )
        for request_idx, start in enumerate(
            range(0, len(input_ids), REQUEST_BATCH_SIZE), start=1
        ):
            batch = input_ids[start : start + REQUEST_BATCH_SIZE]
            print(
                f"decode request {request_idx}/{num_requests}: "
                f"batch_size={len(batch)}, prompt_tokens={[len(x) for x in batch]}"
            )
            batch_results = _generate(
                self.base_url,
                batch,
                max_new_tokens=MAX_NEW_TOKENS,
                return_logprob=True,
            )
            self._assert_results(batch_results, len(batch))
            results.extend(batch_results)
        return results

    def _get_input_logprobs_in_batches(self, new_input_ids, output_logprobs):
        input_logprobs = []
        num_requests = (
            len(new_input_ids) + REQUEST_BATCH_SIZE - 1
        ) // REQUEST_BATCH_SIZE
        print(
            "Replaying prefill logprobs: "
            f"samples={len(new_input_ids)}, requests={num_requests}, "
            f"request_batch_size={REQUEST_BATCH_SIZE}"
        )
        for request_idx, start in enumerate(
            range(0, len(new_input_ids), REQUEST_BATCH_SIZE), start=1
        ):
            end = start + REQUEST_BATCH_SIZE
            batch_input_ids = new_input_ids[start:end]
            batch_output_logprobs = output_logprobs[start:end]
            print(
                f"prefill replay request {request_idx}/{num_requests}: "
                f"batch_size={len(batch_input_ids)}, "
                f"replay_tokens={[len(x) for x in batch_input_ids]}"
            )
            input_logprobs.extend(
                _get_input_logprobs(
                    self.base_url,
                    batch_input_ids,
                    batch_output_logprobs,
                )
            )
        return input_logprobs

    def _compare_prefill_and_decode_logprobs(
        self, new_input_ids, output_logprobs, test_name
    ):
        input_logprobs = self._get_input_logprobs_in_batches(
            new_input_ids, output_logprobs
        )
        avg_kl_div = _compute_kl(input_logprobs, output_logprobs)
        self.assertLess(
            avg_kl_div,
            KL_THRESHOLD,
            f"avg_kl_div={avg_kl_div} >= threshold={KL_THRESHOLD} "
            f"for {self.model} {test_name}",
        )

    def test_decode_logprobs_match_prefill(self):
        print(
            "KL test config: "
            f"model={self.model}, samples={len(self.input_ids)}, "
            f"max_prompt_tokens={MAX_PROMPT_TOKENS}, "
            f"max_new_tokens={MAX_NEW_TOKENS}, "
            f"request_batch_size={REQUEST_BATCH_SIZE}, "
            f"threshold={KL_THRESHOLD}"
        )
        _flush_cache(self.base_url)
        results = self._generate_in_batches(self.input_ids)
        self._assert_results(results, len(self.input_ids))

        new_input_ids = []
        output_logprobs = []
        for prompt_ids, result in zip(self.input_ids, results):
            new_input_ids.append(prompt_ids + result["output_ids"])
            output_logprobs.append(_extract_output_logprobs(result))

        self.assertEqual(len(new_input_ids), len(self.input_ids))
        self._compare_prefill_and_decode_logprobs(
            new_input_ids,
            output_logprobs,
            inspect.currentframe().f_code.co_name,
        )


if __name__ == "__main__":
    unittest.main()
