import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from retrieval_service import (
    DOCUMENT_NOT_FOUND_MESSAGE,
    analyze_query_intent,
    retrieve_context,
)


CHAT_TOKEN = "test-chat-token-general-pdf"
HEADERS = {
    "X-RevOn-Chat-Token": CHAT_TOKEN,
    "X-RevOn-Visitor-IP": "10.20.30.40",
}
ENV = {
    "REVON_CHAT_TOKEN": CHAT_TOKEN,
    "REVON_ADMIN_TOKEN": "test-admin-token-general-pdf",
    "CHAT_RATE_LIMIT_PER_MINUTE": "200",
}


def _intent(intent="document_question", requires_retrieval=True, resolved=None, search=None):
    from retrieval_service import IntentAnalysisResult

    resolved_q = resolved or "resolved question"
    search_q = search if search is not None else (resolved_q if requires_retrieval else None)
    return IntentAnalysisResult(
        {
            "intent": intent,
            "is_relevant": True,
            "requires_retrieval": requires_retrieval,
            "resolved_question": resolved_q,
            "search_query": search_q,
            "standalone_query": search_q,
            "reason": "test",
        }
    )


class TestGeneralPdfRagGrounding(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.env_patcher = patch.dict(os.environ, ENV, clear=False)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_policy_question_with_no_policy_documents_returns_abstention(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(
            intent="document_question",
            resolved="What is the pet insurance benefit?",
            search="pet insurance benefit policy",
        )
        mock_retrieve.return_value = {
            "context": "",
            "sources": [],
            "retrieved_chunks": [],
            "total_matches": 0,
            "kept_chunk_count": 0,
        }
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "What is the pet insurance benefit?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["answer"], DOCUMENT_NOT_FOUND_MESSAGE)
        self.assertEqual(data["sources"], [])
        self.assertFalse(data.get("is_policy_query"))
        mock_retrieve.assert_called_once()
        mock_llm.assert_not_called()

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_profile_question_retrieves_resume_evidence(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(
            intent="document_question",
            resolved="Who is Subrata Haldar?",
            search="Subrata Haldar resume profile Gen AI Developer",
        )
        mock_retrieve.return_value = {
            "context": (
                "[Source 1]\nChunk ID: doc-page-1-chunk-0\nFilename: Subrata_Haldar_Resume_GenAI_Developer.pdf\n"
                "Page: 1\nContent: Subrata Haldar Gen AI Developer | AI Engineer | WordPress Developer"
            ),
            "sources": [
                {
                    "source_index": 1,
                    "chunk_id": "doc-page-1-chunk-0",
                    "filename": "Subrata_Haldar_Resume_GenAI_Developer.pdf",
                    "page_number": 1,
                    "page": 1,
                    "score": 0.91,
                }
            ],
            "retrieved_chunks": [],
            "total_matches": 1,
            "kept_chunk_count": 1,
        }
        mock_llm.return_value = "Subrata Haldar is a Gen AI Developer | AI Engineer | WordPress Developer [Source 1]."
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "who is subrata?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("Subrata Haldar", data["answer"])
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["filename"], "Subrata_Haldar_Resume_GenAI_Developer.pdf")
        mock_retrieve.assert_called_once()

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_technical_query_retrieves_relevant_technical_material(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(
            intent="document_question",
            resolved="What is DNS resolution?",
            search="DNS resolution translating domain names to IP addresses",
        )
        mock_retrieve.return_value = {
            "context": (
                "[Source 1]\nChunk ID: api-page-1-chunk-1\nFilename: Pre-Read_-Basic-Understanding-on-API-1-5983.pdf\n"
                "Page: 1\nContent: 1. DNS Resolution: Translating Domain Names to IP Addresses"
            ),
            "sources": [
                {
                    "source_index": 1,
                    "chunk_id": "api-page-1-chunk-1",
                    "filename": "Pre-Read_-Basic-Understanding-on-API-1-5983.pdf",
                    "page_number": 1,
                    "page": 1,
                    "score": 0.9,
                }
            ],
            "retrieved_chunks": [],
            "total_matches": 1,
            "kept_chunk_count": 1,
        }
        mock_llm.return_value = "DNS resolution is the process of translating domain names to IP addresses [Source 1]."
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "What is DNS resolution?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("DNS", data["answer"])
        self.assertEqual(data["sources"][0]["filename"], "Pre-Read_-Basic-Understanding-on-API-1-5983.pdf")

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_irrelevant_chunks_filtered_by_configured_relevance_gate(self, mock_intent, mock_retrieve, mock_llm):
        # Simulate retrieval layer having applied gating and returning no kept context
        mock_intent.return_value = _intent(
            intent="document_question",
            resolved="What is the maternity leave policy?",
            search="maternity leave policy",
        )
        mock_retrieve.return_value = {
            "context": "",
            "sources": [],
            "retrieved_chunks": [],
            "total_matches": 3,
            "kept_chunk_count": 0,
            "min_score_applied": 0.9,
        }
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "What is the maternity leave policy?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["answer"], DOCUMENT_NOT_FOUND_MESSAGE)
        self.assertEqual(data["sources"], [])
        mock_llm.assert_not_called()

    @patch("retrieval_service.generate_embedding", return_value=[0.1] * 768)
    @patch("retrieval_service.query_vectors")
    def test_retrieve_context_excludes_chunks_below_configured_min_score(self, mock_query, _mock_embed):
        mock_query.return_value = [
            {"id": "a-1", "score": 0.95, "metadata": {"filename": "a.pdf", "page_number": 1, "text": "relevant text"}},
            {"id": "b-1", "score": 0.10, "metadata": {"filename": "b.pdf", "page_number": 2, "text": "unrelated text"}},
        ]
        with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "0.8"}, clear=False):
            result = retrieve_context("query", top_k=5)
        self.assertEqual(result["total_matches"], 2)
        self.assertEqual(result["kept_chunk_count"], 1)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["chunk_id"], "a-1")
        self.assertEqual(result["sources"][0]["filename"], "a.pdf")
        self.assertIn("Chunk ID: a-1", result["context"])

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_missing_evidence_abstention_no_citations(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(intent="document_question", resolved="What is quantum entanglement?", search="quantum entanglement")
        mock_retrieve.return_value = {"context": "", "sources": [], "total_matches": 0, "kept_chunk_count": 0}
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "What is quantum entanglement?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["answer"], DOCUMENT_NOT_FOUND_MESSAGE)
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["total_matches"], 0)
        mock_llm.assert_not_called()

    @patch("retrieval_service.generate_response")
    def test_query_rewriting_preserves_user_intent_on_substantive_question(self, mock_router_llm):
        # Router must not convert substantive question into bypass; resolved/search should preserve intent
        mock_router_llm.return_value = json.dumps(
            {
                "intent": "document_question",
                "requires_retrieval": True,
                "resolved_question": "How does DNS resolution work in the API basics document?",
                "search_query": "DNS resolution translating domain names to IP addresses API basics",
                "standalone_query": "DNS resolution translating domain names to IP addresses API basics",
                "reason": "substantive technical question",
            }
        )
        result = analyze_query_intent("How does DNS resolution work?", history=None)
        self.assertTrue(result.get("requires_retrieval"))
        self.assertIn("DNS", result.get("resolved_question", ""))
        self.assertIn("DNS", result.get("search_query", ""))

    @patch("retrieval_service.generate_embedding", return_value=[0.1] * 768)
    @patch("retrieval_service.query_vectors")
    def test_metadata_filtering_active_only_preserved_in_retrieval(self, mock_query, _mock_embed):
        mock_query.return_value = []
        retrieve_context("test question", top_k=3)
        mock_query.assert_called_once()
        kwargs = mock_query.call_args.kwargs
        self.assertEqual(kwargs.get("top_k"), 3)
        # active-only behavior is default in vector_store; ensure retrieval path does not bypass with explicit overrides
        self.assertNotIn("active_only", kwargs)

    @patch("retrieval_service.query_vectors")
    def test_vector_store_tenant_and_document_filters_supported(self, mock_query_unused):
        # Evidence of tenant-isolation capability at vector-store boundary: query_vectors supports metadata filters
        # via document/logical IDs and is_active; callers must not remove security filters.
        from vector_store import query_vectors
        with patch("vector_store.get_base_url", return_value="https://example"), \
             patch("vector_store.get_headers", return_value={}), \
             patch("httpx.Client") as mock_client:
            resp = mock_client.return_value.__enter__.return_value.post.return_value
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"matches": []}
            query_vectors([0.1] * 768, top_k=5, document_id="doc-1", logical_document_id="logical-1", active_only=True)
            payload = mock_client.return_value.__enter__.return_value.post.call_args.kwargs["json"]
        self.assertEqual(payload["filter"]["is_active"], {"$eq": True})
        self.assertEqual(payload["filter"]["document_id"], {"$eq": "doc-1"})
        self.assertEqual(payload["filter"]["logical_document_id"], {"$eq": "logical-1"})

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_multi_chunk_multi_pdf_evidence_preserved_in_context(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(intent="document_question", resolved="Compare RAG and API basics", search="RAG and API basics overview")
        mock_retrieve.return_value = {
            "context": (
                "[Source 1]\nChunk ID: rag-1\nFilename: Summary_-Introduction-to-Retrieval-Augmented-GenerationRAG-Text-Representation-6066.pdf\n"
                "Page: 1\nContent: Retrieval-Augmented Generation (RAG)\n\n"
                "[Source 2]\nChunk ID: api-1\nFilename: Pre-Read_-Basic-Understanding-on-API-1-5983.pdf\n"
                "Page: 1\nContent: DNS Resolution: Translating Domain Names to IP Addresses"
            ),
            "sources": [
                {"source_index": 1, "chunk_id": "rag-1", "filename": "Summary_-Introduction-to-Retrieval-Augmented-GenerationRAG-Text-Representation-6066.pdf", "page_number": 1, "page": 1, "score": 0.9},
                {"source_index": 2, "chunk_id": "api-1", "filename": "Pre-Read_-Basic-Understanding-on-API-1-5983.pdf", "page_number": 1, "page": 1, "score": 0.88},
            ],
            "retrieved_chunks": [],
            "total_matches": 2,
            "kept_chunk_count": 2,
        }
        mock_llm.return_value = "RAG combines retrieval and generation [Source 1]. DNS resolution translates domain names to IP addresses [Source 2]."
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "Compare RAG and API basics"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["sources"]), 2)
        filenames = {s["filename"] for s in data["sources"]}
        self.assertIn("Summary_-Introduction-to-Retrieval-Augmented-GenerationRAG-Text-Representation-6066.pdf", filenames)
        self.assertIn("Pre-Read_-Basic-Understanding-on-API-1-5983.pdf", filenames)
        passed_messages = mock_llm.call_args.kwargs["messages"]
        user_msg = passed_messages[-1]["content"]
        self.assertIn("Retrieved PDF Context", user_msg)
        self.assertIn("[Source 1]", user_msg)
        self.assertIn("[Source 2]", user_msg)

    @patch("retrieval_service.generate_chat_stream")
    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_streaming_and_non_streaming_grounding_consistent_for_missing_evidence(self, mock_intent, mock_retrieve, mock_chat, mock_stream):
        mock_intent.return_value = _intent(intent="document_question", resolved="pet insurance", search="pet insurance")
        mock_retrieve.return_value = {"context": "", "sources": [], "total_matches": 0, "kept_chunk_count": 0}
        non_stream = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "pet insurance?"})
        stream = self.client.post("/api/v1/chat/stream", headers=HEADERS, json={"question": "pet insurance?"})
        self.assertEqual(non_stream.status_code, 200)
        self.assertEqual(stream.status_code, 200)
        self.assertEqual(non_stream.json()["answer"], DOCUMENT_NOT_FOUND_MESSAGE)
        self.assertIn(DOCUMENT_NOT_FOUND_MESSAGE, stream.text)
        mock_chat.assert_not_called()
        mock_stream.assert_not_called()

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_partial_evidence_answer_is_grounded_and_prompt_requests_missing_info(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(intent="document_question", resolved="What are Subrata's skills and salary?", search="Subrata skills salary")
        mock_retrieve.return_value = {
            "context": (
                "[Source 1]\nChunk ID: resume-1\nFilename: Subrata_Haldar_Resume_GenAI_Developer.pdf\n"
                "Page: 1\nContent: Technical skills include Python, FastAPI, Pinecone, and LangChain."
            ),
            "sources": [
                {"source_index": 1, "chunk_id": "resume-1", "filename": "Subrata_Haldar_Resume_GenAI_Developer.pdf", "page_number": 1, "page": 1, "score": 0.9}
            ],
            "total_matches": 1,
            "kept_chunk_count": 1,
        }
        mock_llm.return_value = "Based on the resume, Subrata's skills include Python, FastAPI, Pinecone, and LangChain [Source 1]. The uploaded PDFs do not contain sufficient information about his salary."
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "What are Subrata's skills and salary?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("Python", data["answer"])
        self.assertIn("sufficient information", data["answer"].lower())
        self.assertEqual(len(data["sources"]), 1)
        user_msg = mock_llm.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("Partially relevant evidence found", user_msg)

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_conflicting_evidence_prompt_instructs_conflict_reporting(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(intent="document_question", resolved="What is the project deadline?", search="project deadline")
        mock_retrieve.return_value = {
            "context": (
                "[Source 1]\nChunk ID: d1\nFilename: plan_a.pdf\nPage: 2\nContent: The deadline is March 10.\n\n"
                "[Source 2]\nChunk ID: d2\nFilename: plan_b.pdf\nPage: 3\nContent: The deadline is March 15."
            ),
            "sources": [
                {"source_index": 1, "chunk_id": "d1", "filename": "plan_a.pdf", "page_number": 2, "page": 2, "score": 0.89},
                {"source_index": 2, "chunk_id": "d2", "filename": "plan_b.pdf", "page_number": 3, "page": 3, "score": 0.88},
            ],
            "total_matches": 2,
            "kept_chunk_count": 2,
        }
        mock_llm.return_value = "The sources conflict: plan_a.pdf states March 10 [Source 1], while plan_b.pdf states March 15 [Source 2]."
        res = self.client.post("/api/v1/chat", headers=HEADERS, json={"question": "What is the project deadline?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["sources"]), 2)
        user_msg = mock_llm.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("Conflicting evidence found", user_msg)

    @patch("retrieval_service.generate_chat_response")
    @patch("retrieval_service.retrieve_context")
    @patch("retrieval_service.analyze_query_intent")
    def test_follow_up_question_uses_history_and_still_retrieves_grounded_evidence(self, mock_intent, mock_retrieve, mock_llm):
        mock_intent.return_value = _intent(
            intent="follow_up_question",
            resolved="What technologies has Subrata Haldar worked with?",
            search="Subrata Haldar technologies skills Pinecone LangChain FastAPI",
        )
        mock_retrieve.return_value = {
            "context": "[Source 1]\nChunk ID: r1\nFilename: Subrata_Haldar_Resume_GenAI_Developer.pdf\nPage: 1\nContent: Retrieval: Pinecone. Agents / Backend: LangChain, FastAPI.",
            "sources": [{"source_index": 1, "chunk_id": "r1", "filename": "Subrata_Haldar_Resume_GenAI_Developer.pdf", "page_number": 1, "page": 1, "score": 0.92}],
            "total_matches": 1,
            "kept_chunk_count": 1,
        }
        mock_llm.return_value = "He has worked with Pinecone, LangChain, and FastAPI [Source 1]."
        payload = {
            "question": "Which technologies has he worked with?",
            "conversation_id": "conv-followup-grounded",
            "history": [
                {"role": "user", "content": "Who is Subrata Haldar?"},
                {"role": "assistant", "content": "Subrata Haldar is a Gen AI Developer [Source 1]."},
            ],
        }
        res = self.client.post("/api/v1/chat", headers=HEADERS, json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("Pinecone", data["answer"])
        self.assertEqual(data["sources"][0]["filename"], "Subrata_Haldar_Resume_GenAI_Developer.pdf")
        # intent analysis receives history for reference resolution
        self.assertEqual(mock_intent.call_args[0][0], "Which technologies has he worked with?")
        self.assertTrue(len(mock_intent.call_args[0][1]) >= 2)
        mock_retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
