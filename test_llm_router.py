"""
Comprehensive automated tests for Multi-LLM Provider Fallback Router.
Verifies all 14 required scenarios specified in Section 12:
- Test 1: Groq succeeds -> Groq response returned, no fallback.
- Test 2: Groq returns 429 -> OpenRouter succeeds.
- Test 3: Groq 429 -> OpenRouter 503 -> Gemini succeeds.
- Test 4: Groq fails -> OpenRouter fails -> Gemini fails -> OpenAI succeeds.
- Test 5: All 4 providers fail -> Controlled AllProvidersFailedError.
- Test 6: Groq invalid credentials -> Safe fallback to next provider.
- Test 7: All providers receive identical prepared conversation and RAG context.
- Test 8: Pinecone retrieval executes only once, even if multiple LLM providers fail.
- Test 9: Streaming fallback works when primary provider fails before first token.
- Test 10: Mid-stream provider failure does not duplicate generated content.
- Test 11: Conversation history remains intact after switching providers.
- Test 12: Existing document citations remain correct after fallback.
- Test 13: Existing domain restriction continues to reject unrelated questions.
- Test 14: Existing WordPress chat proxy continues to work.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock, call
from fastapi.testclient import TestClient

from main import app
from conversation_service import conversation_manager
from llm_router import (
    LLMProviderRouter,
    BaseLLMProvider,
    LLMResult,
    LLMProviderError,
    AllProvidersFailedError,
    MidStreamProviderError,
    set_router,
)
from retrieval_service import (
    answer_conversational_question,
    stream_conversational_question,
)


class MockProvider(BaseLLMProvider):
    """Configurable mock provider for deterministic multi-LLM testing."""

    def __init__(
        self,
        name: str,
        fail_with: Exception | None = None,
        return_text: str = "Default answer",
        stream_chunks: list[str] | None = None,
        fail_on_chunk_index: int | None = None,
        is_configured_value: bool = True,
    ):
        super().__init__(model=f"mock-{name}-model", api_key="test-key")
        self.name = name
        self.fail_with = fail_with
        self.return_text = return_text
        self.stream_chunks = stream_chunks or ["Chunk 1 ", "Chunk 2 ", "Chunk 3"]
        self.fail_on_chunk_index = fail_on_chunk_index
        self._configured = is_configured_value
        self.call_history: list[dict] = []

    def is_configured(self) -> bool:
        return self._configured

    def generate(self, messages, temperature=0.2, max_tokens=1500) -> LLMResult:
        self.call_history.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if self.fail_with:
            raise self.fail_with
        return LLMResult(
            content=self.return_text,
            provider=self.name,
            model=self.model,
            finish_reason="stop",
            latency_seconds=0.05,
        )

    def stream(self, messages, temperature=0.2, max_tokens=1500):
        self.call_history.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if self.fail_with and self.fail_on_chunk_index is None:
            raise self.fail_with

        for idx, chunk in enumerate(self.stream_chunks):
            if self.fail_on_chunk_index is not None and idx == self.fail_on_chunk_index:
                raise self.fail_with or RuntimeError(f"Mid-stream failure in {self.name}")
            yield chunk


class TestMultiLLMFallbackRouter(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.chat_token = "test-chat-token"
        self.admin_token = "test-admin-token"
        self.headers = {
            "X-RevOn-Chat-Token": self.chat_token,
            "X-RevOn-Visitor-IP": "203.0.113.1",
        }
        self.env_patcher = patch.dict(
            os.environ,
            {
                "REVON_CHAT_TOKEN": self.chat_token,
                "REVON_ADMIN_TOKEN": self.admin_token,
                "LLM_PROVIDER_PRIORITY": "groq,openrouter,gemini,openai",
            }
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        set_router(None)  # Reset router to default

    # ------------------------------------------------------------------------
    # Test 1: Groq succeeds -> Groq response returned. No fallback.
    # ------------------------------------------------------------------------
    def test_1_groq_succeeds_no_fallback(self):
        p_groq = MockProvider("groq", return_text="Groq answered.")
        p_or = MockProvider("openrouter", return_text="OpenRouter answered.")
        p_gem = MockProvider("gemini", return_text="Gemini answered.")
        p_oa = MockProvider("openai", return_text="OpenAI answered.")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        res = router.execute_generate([{"role": "user", "content": "hi"}])
        self.assertEqual(res.provider, "groq")
        self.assertEqual(res.content, "Groq answered.")
        self.assertEqual(len(p_groq.call_history), 1)
        self.assertEqual(len(p_or.call_history), 0)
        self.assertEqual(len(p_gem.call_history), 0)
        self.assertEqual(len(p_oa.call_history), 0)

    # ------------------------------------------------------------------------
    # Test 2: Groq returns HTTP 429 -> OpenRouter succeeds.
    # ------------------------------------------------------------------------
    def test_2_groq_429_openrouter_succeeds(self):
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "Rate limit exceeded", status_code=429))
        p_or = MockProvider("openrouter", return_text="OpenRouter answered after Groq 429.")
        p_gem = MockProvider("gemini", return_text="Gemini answered.")
        p_oa = MockProvider("openai", return_text="OpenAI answered.")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        res = router.execute_generate([{"role": "user", "content": "hi"}])
        self.assertEqual(res.provider, "openrouter")
        self.assertEqual(res.content, "OpenRouter answered after Groq 429.")
        self.assertEqual(len(p_groq.call_history), 1)
        self.assertEqual(len(p_or.call_history), 1)
        self.assertEqual(len(p_gem.call_history), 0)
        self.assertEqual(len(p_oa.call_history), 0)

    # ------------------------------------------------------------------------
    # Test 3: Groq 429 -> OpenRouter 503 -> Gemini succeeds.
    # ------------------------------------------------------------------------
    def test_3_groq_429_openrouter_503_gemini_succeeds(self):
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "Rate limit", status_code=429))
        p_or = MockProvider("openrouter", fail_with=LLMProviderError("openrouter", "Service Unavailable", status_code=503))
        p_gem = MockProvider("gemini", return_text="Gemini answered after Groq and OpenRouter failed.")
        p_oa = MockProvider("openai", return_text="OpenAI answered.")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        res = router.execute_generate([{"role": "user", "content": "hi"}])
        self.assertEqual(res.provider, "gemini")
        self.assertEqual(res.content, "Gemini answered after Groq and OpenRouter failed.")
        self.assertEqual(len(p_groq.call_history), 1)
        self.assertEqual(len(p_or.call_history), 1)
        self.assertEqual(len(p_gem.call_history), 1)
        self.assertEqual(len(p_oa.call_history), 0)

    # ------------------------------------------------------------------------
    # Test 4: Groq fails -> OpenRouter fails -> Gemini fails -> OpenAI succeeds.
    # ------------------------------------------------------------------------
    def test_4_groq_or_gemini_fail_openai_succeeds(self):
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "Rate limit", status_code=429))
        p_or = MockProvider("openrouter", fail_with=LLMProviderError("openrouter", "Unavailable", status_code=503))
        p_gem = MockProvider("gemini", fail_with=LLMProviderError("gemini", "High demand spike", status_code=503))
        p_oa = MockProvider("openai", return_text="OpenAI generated final successful response.")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        res = router.execute_generate([{"role": "user", "content": "hi"}])
        self.assertEqual(res.provider, "openai")
        self.assertEqual(res.content, "OpenAI generated final successful response.")
        self.assertEqual(len(p_groq.call_history), 1)
        self.assertEqual(len(p_or.call_history), 1)
        self.assertEqual(len(p_gem.call_history), 1)
        self.assertEqual(len(p_oa.call_history), 1)

    # ------------------------------------------------------------------------
    # Test 5: All 4 providers fail -> Controlled error returned.
    # ------------------------------------------------------------------------
    def test_5_all_providers_fail_controlled_error(self):
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "Rate limit", status_code=429))
        p_or = MockProvider("openrouter", fail_with=LLMProviderError("openrouter", "503", status_code=503))
        p_gem = MockProvider("gemini", fail_with=LLMProviderError("gemini", "Timeout", status_code=504))
        p_oa = MockProvider("openai", fail_with=LLMProviderError("openai", "Rate limit", status_code=429))

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        with self.assertRaises(AllProvidersFailedError) as ctx:
            router.execute_generate([{"role": "user", "content": "hi"}])

        self.assertEqual(len(ctx.exception.failure_history), 4)
        self.assertEqual(ctx.exception.failure_history[0]["provider"], "groq")
        self.assertEqual(ctx.exception.failure_history[1]["provider"], "openrouter")
        self.assertEqual(ctx.exception.failure_history[2]["provider"], "gemini")
        self.assertEqual(ctx.exception.failure_history[3]["provider"], "openai")

    # ------------------------------------------------------------------------
    # Test 6: Groq has invalid credentials -> Correct error & safe fallback.
    # ------------------------------------------------------------------------
    def test_6_groq_invalid_credentials_safe_fallback(self):
        # Groq fails with 401 Unauthorized
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "Invalid API Key", status_code=401, is_transient=False))
        p_or = MockProvider("openrouter", return_text="OpenRouter took over after Groq auth failure.")
        p_gem = MockProvider("gemini", return_text="Gemini standby.")
        p_oa = MockProvider("openai", return_text="OpenAI standby.")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        res = router.execute_generate([{"role": "user", "content": "hi"}])
        self.assertEqual(res.provider, "openrouter")
        self.assertEqual(res.content, "OpenRouter took over after Groq auth failure.")

    # ------------------------------------------------------------------------
    # Test 7: All providers receive identical prepared conversation & RAG context.
    # ------------------------------------------------------------------------
    def test_7_all_providers_receive_identical_prepared_context(self):
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "429", status_code=429))
        p_or = MockProvider("openrouter", fail_with=LLMProviderError("openrouter", "503", status_code=503))
        p_gem = MockProvider("gemini", fail_with=LLMProviderError("gemini", "500", status_code=500))
        p_oa = MockProvider("openai", return_text="OpenAI final.")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        messages = [
            {"role": "system", "content": "RevOn Strict Prompt"},
            {"role": "user", "content": "Context: [Source 1] Policy Info\nQuestion: Leave days?"}
        ]
        router.execute_generate(messages)

        # Verify all four providers received the exact same messages payload
        self.assertEqual(p_groq.call_history[0]["messages"], messages)
        self.assertEqual(p_or.call_history[0]["messages"], messages)
        self.assertEqual(p_gem.call_history[0]["messages"], messages)
        self.assertEqual(p_oa.call_history[0]["messages"], messages)

    # ------------------------------------------------------------------------
    # Test 8: Pinecone retrieval executes only once, even if providers fail.
    # ------------------------------------------------------------------------
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_8_pinecone_retrieval_executes_only_once(self, mock_intent, mock_retrieve):
        mock_intent.return_value = {
            "intent": "domain_question",
            "is_relevant": True,
            "requires_retrieval": True,
            "resolved_question": "What is the policy?",
            "search_query": "policy leave days",
            "standalone_query": "policy leave days",
            "reason": "domain inquiry"
        }
        mock_retrieve.return_value = {
            "context": "[Source 1]\nFilename: leave.pdf\nPage: 1\nContent: 20 days annual leave.",
            "sources": [{"filename": "leave.pdf", "page_number": 1, "score": 0.95}],
            "total_matches": 1
        }

        # Groq fails, OpenRouter fails, Gemini succeeds
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "429", status_code=429))
        p_or = MockProvider("openrouter", fail_with=LLMProviderError("openrouter", "503", status_code=503))
        p_gem = MockProvider("gemini", return_text="According to the policy, employees receive 20 days [Source 1].")
        p_oa = MockProvider("openai", return_text="OpenAI")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        res = answer_conversational_question("What is the policy?")
        self.assertIn("20 days", res["answer"])
        self.assertIn("[Source 1]", res["answer"])

        # Retrieval must be called exactly once
        mock_retrieve.assert_called_once_with("policy leave days", top_k=5)

    # ------------------------------------------------------------------------
    # Test 9: Streaming fallback works when primary fails before first token.
    # ------------------------------------------------------------------------
    def test_9_streaming_fallback_before_first_token(self):
        # Groq fails before emitting any token; OpenRouter succeeds with streaming
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "Rate limit 429", status_code=429))
        p_or = MockProvider("openrouter", stream_chunks=["Hello ", "from ", "OpenRouter!"])
        p_gem = MockProvider("gemini")
        p_oa = MockProvider("openai")

        router = LLMProviderRouter(
            priority=["groq", "openrouter", "gemini", "openai"],
            providers={"groq": p_groq, "openrouter": p_or, "gemini": p_gem, "openai": p_oa}
        )
        set_router(router)

        chunks = list(router.generate_chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["Hello ", "from ", "OpenRouter!"])
        self.assertEqual(len(p_groq.call_history), 1)
        self.assertEqual(len(p_or.call_history), 1)
        self.assertEqual(len(p_gem.call_history), 0)

    # ------------------------------------------------------------------------
    # Test 10: Mid-stream provider failure does not duplicate generated content.
    # ------------------------------------------------------------------------
    def test_10_mid_stream_failure_does_not_duplicate_content(self):
        # Groq emits Chunk 0 and Chunk 1, then crashes on Chunk 2
        p_groq = MockProvider(
            "groq",
            stream_chunks=["Part 1 ", "Part 2 ", "Part 3 "],
            fail_on_chunk_index=2,
            fail_with=RuntimeError("Groq socket disconnected abruptly")
        )
        p_or = MockProvider("openrouter", stream_chunks=["OpenRouter duplicate full stream!"])

        router = LLMProviderRouter(
            priority=["groq", "openrouter"],
            providers={"groq": p_groq, "openrouter": p_or}
        )
        set_router(router)

        stream = router.generate_chat_stream([{"role": "user", "content": "hi"}])
        emitted = []
        with self.assertRaises(MidStreamProviderError) as ctx:
            for token in stream:
                emitted.append(token)

        # Verified: Emitted tokens before crash are received
        self.assertEqual(emitted, ["Part 1 ", "Part 2 "])
        # OpenRouter MUST NOT have been called to restart/duplicate output!
        self.assertEqual(len(p_or.call_history), 0)
        self.assertIn("disconnected abruptly", str(ctx.exception))

    # ------------------------------------------------------------------------
    # Test 11: Conversation history remains intact after switching providers.
    # ------------------------------------------------------------------------
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_11_conversation_history_remains_intact_after_fallback(self, mock_intent, mock_retrieve):
        from retrieval_service import IntentAnalysisResult

        # Turn 1: Groq answers (capability/conversational, no retrieval)
        p_groq_1 = MockProvider("groq", return_text="My name is RevOn Assistant.")
        p_or = MockProvider("openrouter", return_text="OpenRouter")
        router1 = LLMProviderRouter(priority=["groq", "openrouter"], providers={"groq": p_groq_1, "openrouter": p_or})
        set_router(router1)
        mock_intent.return_value = IntentAnalysisResult({
            "intent": "capability_question",
            "is_relevant": True,
            "requires_retrieval": False,
            "resolved_question": "Who are you?",
            "search_query": None,
            "standalone_query": None,
            "reason": "test",
        })

        res1 = self.client.post("/api/v1/chat", headers=self.headers, json={"question": "Who are you?"})
        self.assertEqual(res1.status_code, 200)
        cid = res1.json()["conversation_id"]

        # Turn 2: Groq 429 fails -> OpenRouter takes over with conversational memory
        p_groq_2 = MockProvider("groq", fail_with=LLMProviderError("groq", "429", status_code=429))
        p_or_2 = MockProvider("openrouter", return_text="I remember our conversation.")
        router2 = LLMProviderRouter(priority=["groq", "openrouter"], providers={"groq": p_groq_2, "openrouter": p_or_2})
        set_router(router2)
        mock_intent.return_value = IntentAnalysisResult({
            "intent": "conversational",
            "is_relevant": True,
            "requires_retrieval": False,
            "resolved_question": "What did I ask?",
            "search_query": None,
            "standalone_query": None,
            "reason": "test",
        })

        res2 = self.client.post("/api/v1/chat", headers=self.headers, json={"question": "What did I ask?", "conversation_id": cid})
        self.assertEqual(res2.status_code, 200)

        # Inspect memory in OpenRouter's received messages
        received_messages = p_or_2.call_history[0]["messages"]
        self.assertTrue(any("Who are you?" in m.get("content", "") for m in received_messages))
        mock_retrieve.assert_not_called()

    # ------------------------------------------------------------------------
    # Test 12: Existing document citations remain correct after fallback.
    # ------------------------------------------------------------------------
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_12_citations_remain_correct_after_fallback(self, mock_intent, mock_retrieve):
        mock_intent.return_value = {
            "intent": "domain_question",
            "is_relevant": True,
            "requires_retrieval": True,
            "resolved_question": "What is the policy?",
            "search_query": "policy",
            "standalone_query": "policy",
            "reason": "domain query"
        }
        mock_retrieve.return_value = {
            "context": "[Source 1]\nFilename: Handbook.pdf\nPage: 12\nContent: Medical coverage is included.",
            "sources": [{"filename": "Handbook.pdf", "page_number": 12, "score": 0.88}],
            "total_matches": 1
        }

        # Groq fails, OpenRouter generates answer citing [Source 1]
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "429", status_code=429))
        p_or = MockProvider("openrouter", return_text="Medical coverage is provided to all employees [Source 1].")
        router = LLMProviderRouter(priority=["groq", "openrouter"], providers={"groq": p_groq, "openrouter": p_or})
        set_router(router)

        res = answer_conversational_question("What is the policy?")
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "Handbook.pdf")
        self.assertEqual(res["sources"][0]["page_number"], 12)

    # ------------------------------------------------------------------------
    # Test 13: Substantive questions without supporting chunks must not use pretrained knowledge.
    # ------------------------------------------------------------------------
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_13_unrelated_question_without_evidence_returns_document_not_found(self, mock_intent, mock_retrieve):
        mock_intent.return_value = {
            "intent": "document_question",
            "is_relevant": True,
            "requires_retrieval": True,
            "resolved_question": "What is the capital of France?",
            "search_query": "capital of France",
            "standalone_query": "capital of France",
            "reason": "substantive question"
        }
        mock_retrieve.return_value = {"context": "", "sources": [], "total_matches": 0}

        p_groq = MockProvider("groq", return_text="Paris")
        router = LLMProviderRouter(priority=["groq"], providers={"groq": p_groq})
        set_router(router)

        res = self.client.post("/api/v1/chat", headers=self.headers, json={"question": "What is the capital of France?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["answer"], "The uploaded documents do not contain the requested information.")
        self.assertNotIn("paris", data["answer"].lower())
        self.assertEqual(data["sources"], [])
        self.assertFalse(data["is_policy_query"])
        self.assertEqual(len(p_groq.call_history), 0)
        mock_retrieve.assert_called_once()

    # ------------------------------------------------------------------------
    # Test 14: Existing WordPress chat proxy continues to work.
    # ------------------------------------------------------------------------
    def test_14_wordpress_proxy_compatibility(self):
        p_groq = MockProvider("groq", fail_with=LLMProviderError("groq", "429", status_code=429))
        p_or = MockProvider("openrouter", return_text="Welcome to RevOn! How can I help you today?")
        router = LLMProviderRouter(priority=["groq", "openrouter"], providers={"groq": p_groq, "openrouter": p_or})
        set_router(router)

        # WordPress sends X-RevOn-Visitor-IP and X-RevOn-Chat-Token
        wp_headers = {
            "X-RevOn-Chat-Token": self.chat_token,
            "X-RevOn-Visitor-IP": "104.28.19.45",
        }
        res = self.client.post("/api/v1/chat", headers=wp_headers, json={"question": "Hello"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Welcome to RevOn", data["answer"])
        self.assertTrue(bool(data["conversation_id"]))


if __name__ == "__main__":
    unittest.main()
