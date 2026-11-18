import json
import os
import random
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import embedding_service
from embedding_service import (
    EmbeddingDailyBudgetExhaustedError,
    EmbeddingDailyBudgetStateError,
    EmbeddingRateLimitError,
    _generate_embedding_batch,
    _get_api_key,
    extract_retry_delay_seconds,
    iter_embeddings,
)
from ingestion_service import ingest_pdf


class FakeGenAIError(Exception):
    def __init__(self, message="", code=None, status=None, details=None, response=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        if details is not None:
            self.details = details
        if response is not None:
            self.response = response


def _emb_response(n):
    return SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[float(i % 7)] * embedding_service.EMBEDDING_DIMENSION)
            for i in range(n)
        ]
    )


class TestEmbeddingRateLimitingAndRetries(unittest.TestCase):
    def setUp(self):
        embedding_service._reset_rate_limiter_for_tests()
        # Isolate from persistent local daily-budget accounting used by production safety guard.
        self._budget_env = patch.dict(
            os.environ,
            {
                "GEMINI_EMBEDDING_DAILY_BUDGET_ENABLED": "false",
            },
            clear=False,
        )
        self._budget_env.start()

    def tearDown(self):
        self._budget_env.stop()

    def test_429_resource_exhausted_handled_separately_and_uses_retryinfo_delay(self):
        client = MagicMock()
        err = FakeGenAIError(
            "429 RESOURCE_EXHAUSTED",
            code=429,
            details=[{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}],
        )
        client.models.embed_content.side_effect = [err, _emb_response(2)]
        sleeps = []

        vectors = _generate_embedding_batch(
            ["a", "b"],
            client=client,
            rpm=1000,  # avoid RPM wait dominating this test
            max_retries=3,
            base_backoff_seconds=1.0,
            max_backoff_seconds=60.0,
            sleep_fn=lambda s: sleeps.append(round(s, 6)),
            now_fn=time.monotonic,
            rng=random.Random(0),
        )

        self.assertEqual(len(vectors), 2)
        self.assertEqual(client.models.embed_content.call_count, 2)
        self.assertTrue(sleeps, "expected retry sleep for 429 RESOURCE_EXHAUSTED")
        # RetryInfo says 27s + jitter in [0,1)
        self.assertGreaterEqual(sleeps[0], 27.0)
        self.assertLessEqual(sleeps[0], 28.0)

    def test_extract_retry_delay_from_retryinfo_shapes(self):
        err1 = FakeGenAIError(details=[{"@type": "google.rpc.RetryInfo", "retryDelay": "27s"}])
        self.assertEqual(extract_retry_delay_seconds(err1), 27.0)
        err2 = FakeGenAIError(details=[{"retry_delay": {"seconds": 5, "nanos": 500000000}}])
        self.assertEqual(extract_retry_delay_seconds(err2), 5.5)

    def test_bounded_exponential_backoff_with_jitter_without_retryinfo(self):
        client = MagicMock()
        # Transient server error (retryable), no RetryInfo
        err = FakeGenAIError("503 Service Unavailable", code=503)
        client.models.embed_content.side_effect = [err, err, _emb_response(1)]
        sleeps = []

        vectors = _generate_embedding_batch(
            ["a"],
            client=client,
            rpm=1000,
            max_retries=3,
            base_backoff_seconds=1.0,
            max_backoff_seconds=2.0,  # cap bounds exponential growth
            sleep_fn=lambda s: sleeps.append(s),
            now_fn=time.monotonic,
            rng=random.Random(2),
        )

        self.assertEqual(len(vectors), 1)
        # sleep_fn also captures RPM slot waits; backoff delays are the >=1s sleeps here.
        backoff_sleeps = [s for s in sleeps if s >= 0.5]
        self.assertEqual(len(backoff_sleeps), 2)
        # attempt 0: base*2^0=1 (+jitter<=1) => <=2 and <=cap; attempt 1: base*2^1=2 (+jitter) capped at 2
        self.assertGreaterEqual(backoff_sleeps[0], 1.0)
        self.assertLessEqual(backoff_sleeps[0], 2.0)
        self.assertGreaterEqual(backoff_sleeps[1], 2.0)
        self.assertLessEqual(backoff_sleeps[1], 2.0)

    def test_non_retryable_error_is_not_retried(self):
        client = MagicMock()
        err = FakeGenAIError("INVALID_ARGUMENT: bad input", code=400)
        client.models.embed_content.side_effect = err
        sleeps = []

        with self.assertRaises(FakeGenAIError):
            _generate_embedding_batch(
                ["a"],
                client=client,
                rpm=1000,
                max_retries=5,
                sleep_fn=lambda s: sleeps.append(s),
                rng=random.Random(0),
            )

        self.assertEqual(client.models.embed_content.call_count, 1)
        self.assertEqual(sleeps, [])

    def test_repeated_429_failures_raise_rate_limit_error_after_bounded_retries(self):
        client = MagicMock()
        err = FakeGenAIError("RESOURCE_EXHAUSTED", code=429, details=[{"retryDelay": "1s"}])
        client.models.embed_content.side_effect = err
        sleeps = []

        with self.assertRaises(EmbeddingRateLimitError):
            _generate_embedding_batch(
                ["a", "b"],
                client=client,
                rpm=1000,
                max_retries=2,
                base_backoff_seconds=0.5,
                max_backoff_seconds=5.0,
                sleep_fn=lambda s: sleeps.append(s),
                rng=random.Random(1),
            )

        self.assertEqual(client.models.embed_content.call_count, 3)  # initial + 2 retries
        # sleep_fn also captures RPM slot waits; RetryInfo-guided backoff sleeps are the >=1s sleeps here.
        backoff_sleeps = [s for s in sleeps if s >= 0.5]
        self.assertEqual(len(backoff_sleeps), 2)

    def test_partial_batch_success_then_429_then_success(self):
        chunks = [
            {"page_number": 1, "chunk_index": 0, "text": "c1"},
            {"page_number": 1, "chunk_index": 1, "text": "c2"},
            {"page_number": 1, "chunk_index": 2, "text": "c3"},
        ]
        client = MagicMock()
        err429 = FakeGenAIError("429 RESOURCE_EXHAUSTED", code=429, details=[{"retryDelay": "0s"}])
        client.models.embed_content.side_effect = [
            _emb_response(2),  # batch 1 success
            err429,            # batch 2 first attempt rate-limited
            _emb_response(1),  # batch 2 retry success
        ]
        sleeps = []

        out = []
        for batch in iter_embeddings(
            chunks,
            batch_size=2,
            rpm=1000,
            max_retries=2,
            client=client,
            sleep_fn=lambda s: sleeps.append(s),
            rng=random.Random(0),
        ):
            out.extend(batch)

        self.assertEqual(len(out), 3)
        self.assertEqual([c["chunk_index"] for c in out], [0, 1, 2])
        self.assertEqual(client.models.embed_content.call_count, 3)
        self.assertTrue(len(sleeps) >= 1)

    def test_configurable_rpm_rate_limiting_enforces_request_spacing(self):
        client = MagicMock()
        client.models.embed_content.side_effect = [_emb_response(1), _emb_response(1)]
        current = {"t": 100.0}
        sleeps = []

        def now():
            return current["t"]

        def sleep(s):
            sleeps.append(round(s, 6))
            current["t"] += s

        out = []
        for batch in iter_embeddings(
            [
                {"page_number": 1, "chunk_index": 0, "text": "a"},
                {"page_number": 1, "chunk_index": 1, "text": "b"},
            ],
            batch_size=1,
            rpm=60,  # 1 request/sec (configurable)
            max_retries=0,
            client=client,
            sleep_fn=sleep,
            now_fn=now,
            rng=random.Random(0),
        ):
            out.extend(batch)

        self.assertEqual(len(out), 2)
        self.assertEqual(client.models.embed_content.call_count, 2)
        # second request should wait ~1s due to RPM limiter
        self.assertTrue(any(s >= 0.99 for s in sleeps), sleeps)


class TestGeminiApiKeyPrecedence(unittest.TestCase):
    def setUp(self):
        self._orig_client = embedding_service._client
        self._orig_client_api_key = embedding_service._client_api_key

    def tearDown(self):
        embedding_service._client = self._orig_client
        embedding_service._client_api_key = self._orig_client_api_key

    def test_resolver_prefers_gemini_api_key_over_google_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-placeholder", "GOOGLE_API_KEY": "google-placeholder"}, clear=False):
            self.assertEqual(_get_api_key(), "gemini-placeholder")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "google-placeholder"}, clear=False):
            self.assertEqual(_get_api_key(), "google-placeholder")

    @patch("embedding_service.genai.Client")
    def test_embedding_client_uses_explicit_resolved_key(self, mock_client_ctor):
        # Ensures our service passes the resolved key explicitly (avoids relying on SDK env precedence/warnings).
        embedding_service._client = None
        embedding_service._client_api_key = None
        sentinel = object()
        mock_client_ctor.return_value = sentinel

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-placeholder", "GOOGLE_API_KEY": "google-placeholder"}, clear=False):
            client = embedding_service._get_client()

        self.assertIs(client, sentinel)
        mock_client_ctor.assert_called_once_with(api_key="gemini-placeholder")


class TestDailyEmbeddingBudgetProtection(unittest.TestCase):
    def setUp(self):
        embedding_service._reset_rate_limiter_for_tests()
        self._tmp = tempfile.TemporaryDirectory()
        self.state_path = os.path.join(self._tmp.name, "daily_budget_state.json")

    def tearDown(self):
        self._tmp.cleanup()

    def _env(self, **overrides):
        base = {
            "GEMINI_EMBEDDING_DAILY_BUDGET_ENABLED": "true",
            "GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "2",
            "GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID": "window-a",
            "GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH": self.state_path,
        }
        base.update({k: str(v) for k, v in overrides.items()})
        return patch.dict(os.environ, base, clear=False)

    def test_budget_exhaustion_stops_before_next_request_and_is_non_retryable(self):
        client = MagicMock()
        client.models.embed_content.side_effect = [_emb_response(1)]
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "1"}):
            # first request consumes budget
            out = _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=2)
            self.assertEqual(len(out), 1)
            self.assertEqual(client.models.embed_content.call_count, 1)

            # next request must be stopped before issuing API call
            client.models.embed_content.reset_mock()
            with self.assertRaises(EmbeddingDailyBudgetExhaustedError) as ctx:
                _generate_embedding_batch(["b"], client=client, rpm=1000, max_retries=5)
            msg = str(ctx.exception).lower()
            self.assertIn("local", msg)
            self.assertIn("authoritative", msg)
            self.assertIn("budget", msg)
            client.models.embed_content.assert_not_called()

            # clear, non-retryable: wrapper should not classify as retryable
            kind, _ = embedding_service._classify_embedding_error(ctx.exception)
            self.assertEqual(kind, "non_retryable")

    def test_persistence_and_restart_recovery_across_process_restart_simulation(self):
        client = MagicMock()
        client.models.embed_content.side_effect = [_emb_response(1), _emb_response(1)]
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "2"}):
            _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=0)
            self.assertTrue(os.path.exists(self.state_path))
            with open(self.state_path, "r", encoding="utf-8") as f:
                state1 = json.load(f)
            self.assertEqual(state1.get("window_id"), "window-a")
            self.assertEqual(state1.get("used_requests"), 1)

            # simulate restart: module-level in-memory state not relied on (file is source of truth)
            embedding_service._client = embedding_service._client
            embedding_service._reset_rate_limiter_for_tests()

            _generate_embedding_batch(["b"], client=client, rpm=1000, max_retries=0)
            with open(self.state_path, "r", encoding="utf-8") as f:
                state2 = json.load(f)
            self.assertEqual(state2.get("used_requests"), 2)

            with self.assertRaises(EmbeddingDailyBudgetExhaustedError):
                _generate_embedding_batch(["c"], client=client, rpm=1000, max_retries=0)
            self.assertEqual(client.models.embed_content.call_count, 2)

    def test_no_automatic_reset_without_operator_window_change(self):
        client = MagicMock()
        client.models.embed_content.side_effect = [_emb_response(1)]
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "1", "GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID": "window-fixed"}):
            _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=0)
            # simulate later time / restart-like reuse of same window; no time-based reset allowed
            with self.assertRaises(EmbeddingDailyBudgetExhaustedError):
                _generate_embedding_batch(["b"], client=client, rpm=1000, max_retries=0)

        # operator-controlled reset mechanism: changing window id starts a fresh local budget window
        client.models.embed_content.side_effect = [_emb_response(1)]
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "1", "GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID": "window-reset-by-operator"}):
            out = _generate_embedding_batch(["c"], client=client, rpm=1000, max_retries=0)
            self.assertEqual(len(out), 1)
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state.get("window_id"), "window-reset-by-operator")
        self.assertEqual(state.get("used_requests"), 1)

    def test_corrupted_json_state_fails_closed_and_blocks_requests(self):
        client = MagicMock()
        client.models.embed_content.return_value = _emb_response(1)
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write("{not-valid-json")
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "5"}):
            with self.assertRaises(EmbeddingDailyBudgetStateError) as ctx:
                _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=0)
        msg = str(ctx.exception).lower()
        self.assertIn("corrupted", msg)
        self.assertIn("operator recovery", msg)
        self.assertIn("gemini_embedding_daily_budget_recovery_mode=reset", msg)
        client.models.embed_content.assert_not_called()

    def test_unreadable_state_file_fails_closed_and_blocks_requests(self):
        client = MagicMock()
        client.models.embed_content.return_value = _emb_response(1)
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"window_id": "window-a", "used_requests": 1}))
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "5"}):
            real_open = open

            def guarded_open(path, mode="r", *args, **kwargs):
                if os.path.abspath(path) == os.path.abspath(self.state_path) and "r" in mode:
                    raise PermissionError("simulated unreadable state file")
                return real_open(path, mode, *args, **kwargs)

            with patch("embedding_service.open", side_effect=guarded_open):
                with self.assertRaises(EmbeddingDailyBudgetStateError) as ctx:
                    _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=0)
        self.assertIn("unreadable", str(ctx.exception).lower())
        client.models.embed_content.assert_not_called()

    def test_interrupted_write_partial_state_file_fails_closed(self):
        client = MagicMock()
        client.models.embed_content.return_value = _emb_response(1)
        # Simulate interrupted/partial write leaving a truncated target state file.
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write('{"window_id":"window-a","used_requests":')
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "5"}):
            with self.assertRaises(EmbeddingDailyBudgetStateError):
                _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=0)
        client.models.embed_content.assert_not_called()

    def test_explicit_recovery_reset_allows_controlled_reinitialization(self):
        client = MagicMock()
        client.models.embed_content.return_value = _emb_response(1)
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write("{corrupt")
        # Without explicit recovery, fail closed.
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "5"}):
            with self.assertRaises(EmbeddingDailyBudgetStateError):
                _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=0)
        client.models.embed_content.assert_not_called()

        # Explicit operator recovery reinitializes local counter for configured window.
        client.models.embed_content.return_value = _emb_response(1)
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "5", "GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE": "reset"}):
            out = _generate_embedding_batch(["a"], client=client, rpm=1000, max_retries=0)
        self.assertEqual(len(out), 1)
        self.assertEqual(client.models.embed_content.call_count, 1)
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state.get("window_id"), "window-a")
        self.assertEqual(state.get("used_requests"), 1)  # recovered to 0, then this request attempt counted
        self.assertTrue(state.get("recovered_from_state_error"))

    def test_retry_attempts_are_counted_and_budget_exhaustion_can_stop_retries(self):
        client = MagicMock()
        transient = FakeGenAIError("503 transient", code=503)
        client.models.embed_content.side_effect = [transient, transient, _emb_response(1)]
        with self._env(**{"GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "2"}):
            with self.assertRaises(EmbeddingDailyBudgetExhaustedError):
                _generate_embedding_batch(
                    ["a"],
                    client=client,
                    rpm=1000,
                    max_retries=5,
                    base_backoff_seconds=0.01,
                    max_backoff_seconds=0.02,
                    sleep_fn=lambda s: None,
                    rng=random.Random(0),
                )
            # attempt 1 (fails transient, counted), attempt 2 (fails transient, counted), attempt 3 would exceed -> stopped before API call
            self.assertEqual(client.models.embed_content.call_count, 2)
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.assertEqual(state.get("used_requests"), 2)

    @patch("ingestion_service.list_document_vector_ids")
    @patch("ingestion_service.upsert_vectors")
    @patch("ingestion_service.extract_text_from_pdf")
    @patch("ingestion_service.create_chunks")
    def test_resume_after_daily_budget_exhaustion_preserves_progress_and_does_not_complete(
        self, mock_chunks, mock_extract, mock_upsert, mock_list_ids
    ):
        # Real iter_embeddings/_generate_embedding_batch path is used (no patch) to exercise budget accounting.
        mock_extract.return_value = [{"page_number": 1, "text": "p1"}]
        mock_chunks.return_value = [
            {"page_number": 1, "chunk_index": 0, "text": "a"},
            {"page_number": 1, "chunk_index": 1, "text": "b"},
        ]
        mock_list_ids.side_effect = [
            [],  # first attempt initial
            ["doc-b-page-1-chunk-0"],  # second attempt initial (after partial upsert)
            ["doc-b-page-1-chunk-0", "doc-b-page-1-chunk-1"],  # second attempt final re-list
        ]
        mock_upsert.return_value = {"status": "success", "total_upserted": 1}

        fake_client = MagicMock()
        # batch_size=1 => two separate embedding API requests needed; budget=1 allows first only.
        fake_client.models.embed_content.side_effect = [_emb_response(1), _emb_response(1)]
        budget_env = {
            "GEMINI_EMBEDDING_DAILY_BUDGET_ENABLED": "true",
            "GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET": "1",
            "GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID": "w1",
            "GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH": self.state_path,
            "GEMINI_EMBEDDING_BATCH_SIZE": "1",
        }

        with patch.dict(os.environ, {**budget_env, "GEMINI_API_KEY": "test-key"}, clear=False), patch.object(embedding_service, "_client", fake_client), patch.object(embedding_service, "_client_api_key", "test-key"):
            with self.assertRaises(RuntimeError) as ctx:
                ingest_pdf(
                    file_bytes=b"pdf",
                    filename="b.pdf",
                    logical_document_id="logical-b",
                    version=1,
                    document_id="doc-b",
                    request_id="req-b",
                )
            self.assertIn("preserved for status verification or reconciliation", str(ctx.exception))
            self.assertEqual(mock_upsert.call_count, 1)
            self.assertEqual(fake_client.models.embed_content.call_count, 1)

            # Operator-controlled reset/new window allows resume; document was never marked completed/active by ingest_pdf.
            budget_env_resume = dict(budget_env)
            budget_env_resume["GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID"] = "w2"
            budget_env_resume["GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET"] = "5"
            budget_env_resume["GEMINI_API_KEY"] = "test-key"
            with patch.dict(os.environ, budget_env_resume, clear=False):
                result = ingest_pdf(
                    file_bytes=b"pdf",
                    filename="b.pdf",
                    logical_document_id="logical-b",
                    version=1,
                    document_id="doc-b",
                    request_id="req-b",
                )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["is_active"])
        self.assertEqual(result["activation_status"], "pending")
        self.assertEqual(result["expected_vectors"], 2)
        self.assertEqual(result["total_vectors"], 2)
        self.assertEqual(result["vectors_already_present"], 1)
        self.assertEqual(result["vectors_newly_upserted"], 1)
        self.assertEqual(fake_client.models.embed_content.call_count, 2)  # only missing chunk embedded on resume


class TestIngestionResumeAndDuplicatePrevention(unittest.TestCase):
    @patch("ingestion_service.list_document_vector_ids")
    @patch("ingestion_service.upsert_vectors")
    @patch("ingestion_service.iter_embeddings")
    @patch("ingestion_service.create_chunks")
    @patch("ingestion_service.extract_text_from_pdf")
    def test_resume_skips_already_upserted_chunks_and_upserts_only_missing(
        self, mock_extract, mock_chunks, mock_iter_embeddings, mock_upsert, mock_list_ids
    ):
        mock_extract.return_value = [{"page_number": 1, "text": "p1"}, {"page_number": 2, "text": "p2"}]
        mock_chunks.return_value = [
            {"page_number": 1, "chunk_index": 0, "text": "a"},
            {"page_number": 1, "chunk_index": 1, "text": "b"},
            {"page_number": 2, "chunk_index": 0, "text": "c"},
        ]
        # First chunk already present in Pinecone (partial previous attempt)
        mock_list_ids.side_effect = [
            ["doc-r-page-1-chunk-0"],
            ["doc-r-page-1-chunk-0", "doc-r-page-1-chunk-1", "doc-r-page-2-chunk-0"],
        ]

        def iter_emb(missing_chunks):
            # Should only receive missing chunks in original order
            self.assertEqual([(c["page_number"], c["chunk_index"]) for c in missing_chunks], [(1, 1), (2, 0)])
            yield [
                {
                    "page_number": c["page_number"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                    "embedding": [0.1] * 768,
                }
                for c in missing_chunks
            ]

        mock_iter_embeddings.side_effect = iter_emb
        mock_upsert.return_value = {"status": "success", "total_upserted": 2}

        result = ingest_pdf(
            file_bytes=b"pdf",
            filename="r.pdf",
            logical_document_id="logical-r",
            version=1,
            document_id="doc-r",
            request_id="req-r",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["expected_vectors"], 3)
        self.assertEqual(result["total_vectors"], 3)
        self.assertEqual(result["vectors_already_present"], 1)
        self.assertEqual(result["vectors_newly_upserted"], 2)
        mock_upsert.assert_called_once()
        upserted_batch = mock_upsert.call_args.kwargs["embedded_chunks"]
        self.assertEqual([(c["page_number"], c["chunk_index"]) for c in upserted_batch], [(1, 1), (2, 0)])

    @patch("ingestion_service.list_document_vector_ids")
    @patch("ingestion_service.upsert_vectors")
    @patch("ingestion_service.iter_embeddings")
    @patch("ingestion_service.create_chunks")
    @patch("ingestion_service.extract_text_from_pdf")
    def test_duplicate_prevention_when_all_vectors_already_present(
        self, mock_extract, mock_chunks, mock_iter_embeddings, mock_upsert, mock_list_ids
    ):
        mock_extract.return_value = [{"page_number": 1, "text": "p1"}]
        mock_chunks.return_value = [
            {"page_number": 1, "chunk_index": 0, "text": "a"},
            {"page_number": 1, "chunk_index": 1, "text": "b"},
        ]
        all_ids = ["doc-d-page-1-chunk-0", "doc-d-page-1-chunk-1"]
        mock_list_ids.side_effect = [all_ids, all_ids]

        result = ingest_pdf(
            file_bytes=b"pdf",
            filename="d.pdf",
            logical_document_id="logical-d",
            version=1,
            document_id="doc-d",
            request_id="req-d",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_vectors"], 2)
        self.assertEqual(result["vectors_newly_upserted"], 0)
        self.assertEqual(result["vectors_already_present"], 2)
        mock_iter_embeddings.assert_not_called()
        mock_upsert.assert_not_called()

    @patch("ingestion_service.list_document_vector_ids")
    @patch("ingestion_service.upsert_vectors")
    @patch("ingestion_service.iter_embeddings")
    @patch("ingestion_service.create_chunks")
    @patch("ingestion_service.extract_text_from_pdf")
    def test_resume_after_partial_embedding_upsert_failure_preserves_progress(
        self, mock_extract, mock_chunks, mock_iter_embeddings, mock_upsert, mock_list_ids
    ):
        mock_extract.return_value = [{"page_number": 1, "text": "p1"}]
        mock_chunks.return_value = [
            {"page_number": 1, "chunk_index": 0, "text": "a"},
            {"page_number": 1, "chunk_index": 1, "text": "b"},
            {"page_number": 1, "chunk_index": 2, "text": "c"},
        ]
        # First attempt: nothing present, first batch embedded+upserted, second embedding batch fails (rate limit)
        # Second attempt (resume): first vector present, remaining two embedded+upserted
        mock_list_ids.side_effect = [
            [],
            ["doc-p-page-1-chunk-0"],
            ["doc-p-page-1-chunk-0", "doc-p-page-1-chunk-1", "doc-p-page-1-chunk-2"],
        ]

        def iter_first(missing_chunks):
            yield [{"page_number": 1, "chunk_index": 0, "text": "a", "embedding": [0.1] * 768}]
            raise EmbeddingRateLimitError("quota exhausted")

        def iter_second(missing_chunks):
            self.assertEqual([c["chunk_index"] for c in missing_chunks], [1, 2])
            yield [
                {"page_number": 1, "chunk_index": 1, "text": "b", "embedding": [0.2] * 768},
                {"page_number": 1, "chunk_index": 2, "text": "c", "embedding": [0.3] * 768},
            ]

        state = {"n": 0}

        def iter_side_effect(missing_chunks):
            state["n"] += 1
            if state["n"] == 1:
                return iter_first(missing_chunks)
            return iter_second(missing_chunks)

        mock_iter_embeddings.side_effect = iter_side_effect
        mock_upsert.side_effect = [
            {"status": "success", "total_upserted": 1},
            {"status": "success", "total_upserted": 2},
        ]

        # Failure must not mark document completed (ingest_pdf raises; activation is separate)
        with self.assertRaisesRegex(RuntimeError, "preserved for status verification or reconciliation"):
            ingest_pdf(
                file_bytes=b"pdf",
                filename="p.pdf",
                logical_document_id="logical-p",
                version=1,
                document_id="doc-p",
                request_id="req-p1",
            )
        self.assertEqual(mock_upsert.call_count, 1)

        # Retry/resume with same document_id (deterministic IDs/idempotent upsert path)
        result = ingest_pdf(
            file_bytes=b"pdf",
            filename="p.pdf",
            logical_document_id="logical-p",
            version=1,
            document_id="doc-p",
            request_id="req-p1",
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_vectors"], 3)
        self.assertEqual(result["vectors_newly_upserted"], 2)
        self.assertEqual(result["vectors_already_present"], 1)
        self.assertEqual(mock_upsert.call_count, 2)
        self.assertEqual(
            [(c["page_number"], c["chunk_index"]) for c in mock_upsert.call_args_list[1].kwargs["embedded_chunks"]],
            [(1, 1), (1, 2)],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
