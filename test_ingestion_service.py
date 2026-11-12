import unittest
from unittest.mock import patch

from ingestion_service import ingest_pdf


class TestIngestionService(unittest.TestCase):

    @patch("ingestion_service.upsert_vectors")
    @patch("ingestion_service.generate_embeddings")
    @patch("ingestion_service.create_chunks")
    @patch("ingestion_service.extract_text_from_pdf")
    def test_successful_version_ingestion(
        self,
        mock_extract,
        mock_chunks,
        mock_embeddings,
        mock_upsert
    ):
        mock_extract.return_value = [
            {"page_number": 1, "text": "Policy content"}
        ]

        mock_chunks.return_value = [
            {
                "page_number": 1,
                "chunk_index": 0,
                "text": "Policy content"
            }
        ]

        mock_embeddings.return_value = [
            {
                "page_number": 1,
                "chunk_index": 0,
                "text": "Policy content",
                "embedding": [0.1] * 768
            }
        ]

        mock_upsert.return_value = {
            "status": "success",
            "total_upserted": 1
        }

        result = ingest_pdf(
            file_bytes=b"fake-pdf-content",
            filename="leave-policy.pdf",
            logical_document_id="policy-123",
            version=2,
            document_id="doc-123",
            request_id="request-123"
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["document_id"], "doc-123")
        self.assertEqual(result["logical_document_id"], "policy-123")
        self.assertEqual(result["request_id"], "request-123")
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["expected_vectors"], 1)
        self.assertFalse(result["is_active"])
        self.assertEqual(result["activation_status"], "pending")

        mock_upsert.assert_called_once()
        upsert_arguments = mock_upsert.call_args.kwargs

        self.assertEqual(upsert_arguments["document_id"], "doc-123")
        self.assertEqual(upsert_arguments["logical_document_id"], "policy-123")
        self.assertEqual(upsert_arguments["version"], 2)
        self.assertEqual(upsert_arguments["request_id"], "request-123")
        self.assertEqual(upsert_arguments["expected_vectors"], 1)

    def test_empty_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            ingest_pdf(
                file_bytes=b"",
                filename="policy.pdf",
                logical_document_id="policy-123",
                version=1
            )

    def test_invalid_version_is_rejected(self):
        with self.assertRaises(ValueError):
            ingest_pdf(
                file_bytes=b"fake-pdf",
                filename="policy.pdf",
                logical_document_id="policy-123",
                version=0
            )

    def test_blank_request_id_is_rejected(self):
        with self.assertRaises(ValueError):
            ingest_pdf(
                file_bytes=b"fake-pdf",
                filename="policy.pdf",
                logical_document_id="policy-123",
                version=1,
                document_id="doc-123",
                request_id=" "
            )

    @patch("ingestion_service.upsert_vectors")
    @patch("ingestion_service.generate_embeddings")
    @patch("ingestion_service.create_chunks")
    @patch("ingestion_service.extract_text_from_pdf")
    def test_upsert_failure_preserves_vectors_for_reconciliation(
        self,
        mock_extract,
        mock_chunks,
        mock_embeddings,
        mock_upsert
    ):
        mock_extract.return_value = [
            {"page_number": 1, "text": "Policy content"}
        ]

        mock_chunks.return_value = [
            {
                "page_number": 1,
                "chunk_index": 0,
                "text": "Policy content"
            }
        ]

        mock_embeddings.return_value = [
            {
                "page_number": 1,
                "chunk_index": 0,
                "text": "Policy content",
                "embedding": [0.1] * 768
            }
        ]

        mock_upsert.side_effect = RuntimeError("Pinecone upload timed out")

        with self.assertRaisesRegex(
            RuntimeError,
            "preserved for status verification or reconciliation"
        ) as error_context:
            ingest_pdf(
                file_bytes=b"fake-pdf-content",
                filename="policy.pdf",
                logical_document_id="policy-123",
                version=2,
                document_id="doc-123",
                request_id="request-123"
            )

        self.assertIn("doc-123", str(error_context.exception))
        self.assertIn("request-123", str(error_context.exception))
        mock_upsert.assert_called_once()

    @patch("ingestion_service.upsert_vectors")
    @patch("ingestion_service.generate_embeddings")
    @patch("ingestion_service.create_chunks")
    @patch("ingestion_service.extract_text_from_pdf")
    def test_expected_vector_count_is_passed_to_upsert(
        self,
        mock_extract,
        mock_chunks,
        mock_embeddings,
        mock_upsert
    ):
        mock_extract.return_value = [
            {"page_number": 1, "text": "First page"},
            {"page_number": 2, "text": "Second page"}
        ]

        mock_chunks.return_value = [
            {"page_number": 1, "chunk_index": 0, "text": "First page"},
            {"page_number": 2, "chunk_index": 0, "text": "Second page"}
        ]

        mock_embeddings.return_value = [
            {
                "page_number": 1,
                "chunk_index": 0,
                "text": "First page",
                "embedding": [0.1] * 768
            },
            {
                "page_number": 2,
                "chunk_index": 0,
                "text": "Second page",
                "embedding": [0.2] * 768
            }
        ]

        mock_upsert.return_value = {
            "status": "success",
            "total_upserted": 2
        }

        result = ingest_pdf(
            file_bytes=b"fake-pdf-content",
            filename="policy.pdf",
            logical_document_id="policy-123",
            version=1,
            document_id="doc-123",
            request_id="request-123"
        )

        self.assertEqual(result["expected_vectors"], 2)
        self.assertEqual(
            mock_upsert.call_args.kwargs["expected_vectors"],
            2
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
