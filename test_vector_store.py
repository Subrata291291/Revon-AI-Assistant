
import unittest
from unittest.mock import MagicMock, patch

import vector_store


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.embedding = [0.1] * vector_store.DIMENSION

        self.chunk = {
            "page_number": 1,
            "chunk_index": 0,
            "text": "Employee leave policy",
            "embedding": self.embedding
        }

    def make_response(self, data=None):
        response = MagicMock()
        response.json.return_value = data or {}
        response.raise_for_status.return_value = None
        return response

    @patch("vector_store.httpx.Client")
    def test_upsert_vectors_adds_version_metadata(self, mock_client):
        client = MagicMock()
        mock_client.return_value.__enter__.return_value = client

        client.post.return_value = self.make_response({
            "upsertedCount": 1
        })

        result = vector_store.upsert_vectors(
            embedded_chunks=[self.chunk],
            document_id="doc-123",
            logical_document_id="policy-123",
            version=2,
            filename="leave-policy.pdf"
        )

        self.assertEqual(result["total_upserted"], 1)
        self.assertEqual(result["version"], 2)

        payload = client.post.call_args.kwargs["json"]
        metadata = payload["vectors"][0]["metadata"]

        self.assertEqual(metadata["document_id"], "doc-123")
        self.assertEqual(
            metadata["logical_document_id"],
            "policy-123"
        )
        self.assertEqual(metadata["version"], 2)
        self.assertFalse(metadata["is_active"])

    @patch("vector_store.httpx.Client")
    def test_upsert_rejects_count_mismatch(self, mock_client):
        client = MagicMock()
        mock_client.return_value.__enter__.return_value = client

        client.post.return_value = self.make_response({
            "upsertedCount": 0
        })

        with self.assertRaisesRegex(
            RuntimeError,
            "upsert count mismatch"
        ):
            vector_store.upsert_vectors(
                embedded_chunks=[self.chunk],
                document_id="doc-123",
                logical_document_id="policy-123",
                version=2,
                filename="leave-policy.pdf"
            )

    @patch("vector_store.httpx.Client")
    def test_list_document_vector_ids_handles_pagination(
        self, mock_client
    ):
        client = MagicMock()
        mock_client.return_value.__enter__.return_value = client

        client.get.side_effect = [
            self.make_response({
                "vectors": [{"id": "doc-123-page-1-chunk-0"}],
                "pagination": {"next": "token-2"}
            }),
            self.make_response({
                "vectors": [{"id": "doc-123-page-1-chunk-1"}],
                "pagination": {}
            })
        ]

        result = vector_store.list_document_vector_ids("doc-123")

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0],
            "doc-123-page-1-chunk-0"
        )
        self.assertEqual(
            result[1],
            "doc-123-page-1-chunk-1"
        )

        second_call_params = (
            client.get.call_args_list[1].kwargs["params"]
        )

        self.assertEqual(
            second_call_params["paginationToken"],
            "token-2"
        )

    @patch("vector_store._update_vector_active")
    @patch("vector_store.list_document_vector_ids")
    def test_set_document_active_updates_all_vectors(
        self, mock_list_ids, mock_update
    ):
        mock_list_ids.return_value = [
            "doc-123-page-1-chunk-0",
            "doc-123-page-1-chunk-1"
        ]

        result = vector_store.set_document_active(
            document_id="doc-123",
            is_active=True,
            rollback_to=False
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_updated"], 2)
        self.assertTrue(result["is_active"])

        self.assertEqual(mock_update.call_count, 2)

        for call in mock_update.call_args_list:
            self.assertTrue(
                call.kwargs["is_active"]
            )

    @patch("vector_store._update_vector_active")
    @patch("vector_store.list_document_vector_ids")
    def test_partial_update_failure_rolls_back_attempted_vectors(
        self, mock_list_ids, mock_update
    ):
        mock_list_ids.return_value = [
            "doc-123-page-1-chunk-0",
            "doc-123-page-1-chunk-1",
            "doc-123-page-1-chunk-2"
        ]

        def update_side_effect(**kwargs):
            vector_id = kwargs["vector_id"]
            is_active = kwargs["is_active"]

            # Simulate failure on the second activation.
            if (
                vector_id == "doc-123-page-1-chunk-1"
                and is_active is True
            ):
                raise RuntimeError("Simulated Pinecone failure")

        mock_update.side_effect = update_side_effect

        with self.assertRaisesRegex(
            RuntimeError,
            "Document activation update failed"
        ):
            vector_store.set_document_active(
                document_id="doc-123",
                is_active=True,
                rollback_to=False
            )

        # Two activation attempts and two rollback attempts.
        self.assertEqual(mock_update.call_count, 4)

        calls = mock_update.call_args_list

        self.assertTrue(calls[0].kwargs["is_active"])
        self.assertTrue(calls[1].kwargs["is_active"])

        self.assertFalse(calls[2].kwargs["is_active"])
        self.assertFalse(calls[3].kwargs["is_active"])

        # Third vector must not be updated after failure.
        self.assertEqual(
            calls[0].kwargs["vector_id"],
            "doc-123-page-1-chunk-0"
        )
        self.assertEqual(
            calls[1].kwargs["vector_id"],
            "doc-123-page-1-chunk-1"
        )

    @patch("vector_store._update_vector_active")
    @patch("vector_store.list_document_vector_ids")
    def test_rollback_continues_after_one_rollback_fails(
        self, mock_list_ids, mock_update
    ):
        mock_list_ids.return_value = [
            "doc-123-page-1-chunk-0",
            "doc-123-page-1-chunk-1",
            "doc-123-page-1-chunk-2"
        ]

        def update_side_effect(**kwargs):
            vector_id = kwargs["vector_id"]
            is_active = kwargs["is_active"]

            # Fail on second activation.
            if (
                vector_id == "doc-123-page-1-chunk-1"
                and is_active is True
            ):
                raise RuntimeError("Activation failed")

            # Fail while rolling back the first vector.
            if (
                vector_id == "doc-123-page-1-chunk-0"
                and is_active is False
            ):
                raise RuntimeError("Rollback failed")

        mock_update.side_effect = update_side_effect

        with self.assertRaisesRegex(
            RuntimeError,
            "Manual reconciliation required"
        ):
            vector_store.set_document_active(
                document_id="doc-123",
                is_active=True,
                rollback_to=False
            )

        # Both rollback attempts must occur,
        # even though the first rollback fails.
        self.assertEqual(mock_update.call_count, 4)

        rollback_calls = mock_update.call_args_list[2:]

        self.assertEqual(
            rollback_calls[0].kwargs["vector_id"],
            "doc-123-page-1-chunk-0"
        )
        self.assertEqual(
            rollback_calls[1].kwargs["vector_id"],
            "doc-123-page-1-chunk-1"
        )

        self.assertFalse(
            rollback_calls[0].kwargs["is_active"]
        )
        self.assertFalse(
            rollback_calls[1].kwargs["is_active"]
        )

    @patch("vector_store.list_document_vector_ids")
    def test_set_document_active_rejects_empty_vector_list(
        self, mock_list_ids
    ):
        mock_list_ids.return_value = []

        with self.assertRaisesRegex(
            ValueError,
            "No Pinecone vectors found"
        ):
            vector_store.set_document_active(
                document_id="doc-123",
                is_active=True,
                rollback_to=False
            )

    @patch("vector_store.httpx.Client")
    def test_query_vectors_filters_active_documents(self, mock_client):
        client = MagicMock()
        mock_client.return_value.__enter__.return_value = client

        client.post.return_value = self.make_response({
            "matches": [
                {
                    "id": "doc-123-page-1-chunk-0",
                    "score": 0.91,
                    "metadata": {"is_active": True}
                }
            ]
        })

        result = vector_store.query_vectors(
            query_embedding=self.embedding,
            top_k=5
        )

        self.assertEqual(len(result), 1)

        payload = client.post.call_args.kwargs["json"]

        self.assertEqual(
            payload["filter"]["is_active"],
            {"$eq": True}
        )

    @patch("vector_store.httpx.Client")
    def test_delete_vectors_uses_document_filter(self, mock_client):
        client = MagicMock()
        mock_client.return_value.__enter__.return_value = client

        client.post.return_value = self.make_response({})

        result = vector_store.delete_vectors("doc-123")

        payload = client.post.call_args.kwargs["json"]

        self.assertEqual(
            payload["filter"]["document_id"],
            {"$eq": "doc-123"}
        )
        self.assertEqual(result["document_id"], "doc-123")

    def test_invalid_embedding_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            vector_store.query_vectors(
                query_embedding=[0.1, 0.2],
                top_k=5
            )


    @patch("vector_store.get_document_vector_metadata")
    def test_verify_document_active_state_all_active(self, mock_metadata):
        mock_metadata.return_value = {
            "vector-1": {"is_active": True},
            "vector-2": {"is_active": True}
        }

        result = vector_store.verify_document_active_state(
            document_id="doc-123",
            expected_active=True,
            expected_vectors=2
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["actual_vectors"], 2)
        self.assertTrue(result["expected_active"])

    @patch("vector_store.get_document_vector_metadata")
    def test_verify_document_active_state_all_inactive(self, mock_metadata):
        mock_metadata.return_value = {
            "vector-1": {"is_active": False},
            "vector-2": {"is_active": False}
        }

        result = vector_store.verify_document_active_state(
            document_id="doc-123",
            expected_active=False,
            expected_vectors=2
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["actual_vectors"], 2)
        self.assertFalse(result["expected_active"])

    @patch("vector_store.get_document_vector_metadata")
    def test_verify_document_active_state_rejects_mixed_state(
        self, mock_metadata
    ):
        mock_metadata.return_value = {
            "vector-1": {"is_active": True},
            "vector-2": {"is_active": False}
        }

        with self.assertRaisesRegex(
            ValueError,
            "active state mismatch"
        ):
            vector_store.verify_document_active_state(
                document_id="doc-123",
                expected_active=True,
                expected_vectors=2
            )

    @patch("vector_store.get_document_vector_metadata")
    def test_verify_document_active_state_rejects_count_mismatch(
        self, mock_metadata
    ):
        mock_metadata.return_value = {
            "vector-1": {"is_active": True}
        }

        with self.assertRaisesRegex(ValueError, "Vector count mismatch"):
            vector_store.verify_document_active_state(
                document_id="doc-123",
                expected_active=True,
                expected_vectors=2
            )

    @patch("vector_store.get_document_vector_metadata")
    def test_verify_document_active_state_rejects_missing_active_metadata(
        self, mock_metadata
    ):
        mock_metadata.return_value = {
            "vector-1": {"document_id": "doc-123"},
            "vector-2": {"is_active": True}
        }

        with self.assertRaisesRegex(
            ValueError,
            "active state mismatch"
        ):
            vector_store.verify_document_active_state(
                document_id="doc-123",
                expected_active=True,
                expected_vectors=2
            )

    @patch("vector_store.get_document_vector_metadata")
    def test_verify_document_active_state_rejects_non_boolean(
        self, mock_metadata
    ):
        with self.assertRaisesRegex(
            ValueError,
            "expected_active must be a boolean"
        ):
            vector_store.verify_document_active_state(
                document_id="doc-123",
                expected_active="true",
                expected_vectors=2
            )

        mock_metadata.assert_not_called()

if __name__ == "__main__":
    unittest.main(verbosity=2)
