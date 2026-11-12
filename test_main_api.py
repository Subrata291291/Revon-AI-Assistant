import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class TestMainAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.token = "test-admin-token"
        self.env_patcher = patch.dict(
            os.environ,
            {"REVON_ADMIN_TOKEN": self.token}
        )
        self.env_patcher.start()
        self.headers = {"X-RevOn-Admin-Token": self.token}

    def tearDown(self):
        self.env_patcher.stop()

    def upload_form(self, upload_type="new_policy", version=1):
        return {
            "upload_type": upload_type,
            "logical_document_id": "policy-123",
            "document_id": "doc-123",
            "request_id": "request-123",
            "version": str(version)
        }

    def fake_ingest_result(self, **kwargs):
        return {
            "status": "success",
            "document_id": kwargs["document_id"],
            "logical_document_id": kwargs["logical_document_id"],
            "request_id": kwargs["request_id"],
            "version": kwargs["version"],
            "total_pages": 2,
            "total_chunks": 5,
            "total_vectors": 5,
            "expected_vectors": 5,
            "is_active": False,
            "activation_status": "pending"
        }

    @patch("main.ingest_pdf")
    def test_new_policy_upload_uses_wordpress_ids(self, mock_ingest):
        mock_ingest.side_effect = self.fake_ingest_result

        response = self.client.post(
            "/api/v1/documents/upload",
            headers=self.headers,
            data=self.upload_form(),
            files={
                "file": (
                    "policy.pdf",
                    b"%PDF-test-content",
                    "application/pdf"
                )
            }
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["document_id"], "doc-123")
        self.assertEqual(data["logical_document_id"], "policy-123")
        self.assertEqual(data["request_id"], "request-123")
        self.assertEqual(data["version"], 1)
        self.assertFalse(data["is_active"])

        mock_ingest.assert_called_once()

        self.assertEqual(
            mock_ingest.call_args.kwargs["document_id"],
            "doc-123"
        )

    @patch("main.ingest_pdf")
    def test_new_version_upload(self, mock_ingest):
        mock_ingest.side_effect = self.fake_ingest_result

        response = self.client.post(
            "/api/v1/documents/upload",
            headers=self.headers,
            data=self.upload_form("new_version", 2),
            files={
                "file": (
                    "updated.pdf",
                    b"%PDF-test-content",
                    "application/pdf"
                )
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version"], 2)

        self.assertEqual(
            mock_ingest.call_args.kwargs["logical_document_id"],
            "policy-123"
        )

    def test_invalid_upload_type(self):
        data = self.upload_form()
        data["upload_type"] = "invalid"

        response = self.client.post(
            "/api/v1/documents/upload",
            headers=self.headers,
            data=data,
            files={
                "file": (
                    "policy.pdf",
                    b"%PDF-test-content",
                    "application/pdf"
                )
            }
        )

        self.assertEqual(response.status_code, 400)

    def test_new_policy_requires_version_one(self):
        response = self.client.post(
            "/api/v1/documents/upload",
            headers=self.headers,
            data=self.upload_form("new_policy", 2),
            files={
                "file": (
                    "policy.pdf",
                    b"%PDF-test-content",
                    "application/pdf"
                )
            }
        )

        self.assertEqual(response.status_code, 400)

    def test_missing_logical_document_id(self):
        data = self.upload_form()
        data.pop("logical_document_id")

        response = self.client.post(
            "/api/v1/documents/upload",
            headers=self.headers,
            data=data,
            files={
                "file": (
                    "policy.pdf",
                    b"%PDF-test-content",
                    "application/pdf"
                )
            }
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_authentication_token(self):
        response = self.client.post(
            "/api/v1/documents/upload",
            data=self.upload_form(),
            files={
                "file": (
                    "policy.pdf",
                    b"%PDF-test-content",
                    "application/pdf"
                )
            }
        )

        self.assertEqual(response.status_code, 401)

    @patch("main.verify_document_active_state")
    @patch("main.set_document_active")
    @patch("main.verify_document_ownership")
    def test_document_activation(
        self,
        mock_verify_ownership,
        mock_activate,
        mock_verify_active_state
    ):
        mock_verify_ownership.side_effect = [
            {
                "document_id": "doc-456",
                "logical_document_id": "policy-123",
                "actual_vectors": 8,
                "verified": True
            },
            {
                "document_id": "doc-123",
                "logical_document_id": "policy-123",
                "actual_vectors": 5,
                "verified": True
            }
        ]

        mock_activate.side_effect = [
            {
                "status": "success",
                "document_id": "doc-123",
                "is_active": False,
                "total_updated": 5
            },
            {
                "status": "success",
                "document_id": "doc-456",
                "is_active": True,
                "total_updated": 8
            }
        ]

        mock_verify_active_state.return_value = {"verified": True}

        response = self.client.post(
            "/api/v1/documents/activate",
            headers=self.headers,
            data={
                "document_id": "doc-456",
                "logical_document_id": "policy-123",
                "version": "2",
                "expected_vectors": "8",
                "request_id": "request-456",
                "previous_document_id": "doc-123"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_updated"], 8)
        self.assertEqual(mock_activate.call_count, 2)

        self.assertEqual(
            mock_activate.call_args_list[0].kwargs["document_id"],
            "doc-123"
        )

        self.assertEqual(
            mock_activate.call_args_list[1].kwargs["document_id"],
            "doc-456"
        )

        self.assertEqual(mock_verify_active_state.call_count, 2)

        self.assertEqual(
            mock_verify_active_state.call_args_list[0].kwargs,
            {
                "document_id": "doc-456",
                "expected_active": True,
                "expected_vectors": 8
            }
        )

        self.assertEqual(
            mock_verify_active_state.call_args_list[1].kwargs,
            {
                "document_id": "doc-123",
                "expected_active": False,
                "expected_vectors": 5
            }
        )

    @patch("main.verify_document_active_state")
    @patch("main.set_document_active")
    @patch("main.verify_document_ownership")
    def test_document_activation_verifies_both_versions(
        self,
        mock_ownership,
        mock_activate,
        mock_active_state
    ):
        mock_ownership.side_effect = [
            {
                "verified": True,
                "actual_vectors": 8
            },
            {
                "verified": True,
                "actual_vectors": 5
            }
        ]

        mock_activate.side_effect = [
            {
                "status": "success",
                "total_updated": 5
            },
            {
                "status": "success",
                "total_updated": 8
            }
        ]

        mock_active_state.return_value = {"verified": True}

        response = self.client.post(
            "/api/v1/documents/activate",
            headers=self.headers,
            data={
                "document_id": "doc-456",
                "logical_document_id": "policy-123",
                "version": "2",
                "expected_vectors": "8",
                "request_id": "request-456",
                "previous_document_id": "doc-123"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_updated"], 8)
        self.assertEqual(mock_activate.call_count, 2)
        self.assertEqual(mock_active_state.call_count, 2)

        self.assertEqual(
            mock_active_state.call_args_list[0].kwargs["document_id"],
            "doc-456"
        )

        self.assertTrue(
            mock_active_state.call_args_list[0].kwargs["expected_active"]
        )

        self.assertEqual(
            mock_active_state.call_args_list[0].kwargs["expected_vectors"],
            8
        )

        self.assertEqual(
            mock_active_state.call_args_list[1].kwargs["document_id"],
            "doc-123"
        )

        self.assertFalse(
            mock_active_state.call_args_list[1].kwargs["expected_active"]
        )

        self.assertEqual(
            mock_active_state.call_args_list[1].kwargs["expected_vectors"],
            5
        )

    @patch("main.set_document_active")
    @patch("main.verify_document_ownership")
    def test_activation_failure_restores_previous_version(
        self,
        mock_verify,
        mock_activate
    ):
        mock_verify.return_value = {"verified": True}

        mock_activate.side_effect = [
            {"status": "success"},
            RuntimeError("New activation failed"),
            {"status": "success"}
        ]

        response = self.client.post(
            "/api/v1/documents/activate",
            headers=self.headers,
            data={
                "document_id": "doc-456",
                "logical_document_id": "policy-123",
                "version": "2",
                "expected_vectors": "8",
                "request_id": "request-456",
                "previous_document_id": "doc-123"
            }
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(mock_activate.call_count, 3)

        recovery_call = mock_activate.call_args_list[2]

        self.assertEqual(
            recovery_call.kwargs["document_id"],
            "doc-123"
        )

        self.assertTrue(recovery_call.kwargs["is_active"])

    @patch("main.set_document_active")
    @patch("main.verify_document_ownership")
    def test_ownership_mismatch_blocks_activation(
        self,
        mock_verify,
        mock_activate
    ):
        mock_verify.side_effect = ValueError(
            "logical_document_id mismatch"
        )

        response = self.client.post(
            "/api/v1/documents/activate",
            headers=self.headers,
            data={
                "document_id": "doc-456",
                "logical_document_id": "another-policy",
                "version": "2",
                "expected_vectors": "8",
                "request_id": "request-456"
            }
        )

        self.assertEqual(response.status_code, 409)
        mock_activate.assert_not_called()

    def test_activation_rejects_same_document_ids(self):
        response = self.client.post(
            "/api/v1/documents/activate",
            headers=self.headers,
            data={
                "document_id": "doc-123",
                "logical_document_id": "policy-123",
                "version": "2",
                "expected_vectors": "8",
                "request_id": "request-456",
                "previous_document_id": "doc-123"
            }
        )

        self.assertEqual(response.status_code, 400)

    @patch("main.verify_document_vector_count")
    def test_document_status_endpoint(self, mock_verify_count):
        mock_verify_count.return_value = {
            "document_id": "doc-123",
            "expected_vectors": 5,
            "actual_vectors": 5,
            "verified": True,
            "vector_ids": ["v1", "v2", "v3", "v4", "v5"]
        }

        response = self.client.get(
            "/api/v1/documents/doc-123/status?expected_vectors=5",
            headers=self.headers
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "indexed")
        self.assertTrue(response.json()["verified"])

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

    
    @patch("main.verify_document_active_state")
    @patch("main.set_document_active")
    @patch("main.verify_document_ownership")
    def test_activation_verification_failure_triggers_recovery(
        self,
        mock_ownership,
        mock_activate,
        mock_active_state
    ):
        mock_ownership.side_effect = [
            {"verified": True, "actual_vectors": 8},
            {"verified": True, "actual_vectors": 5}
        ]

        mock_activate.return_value = {"status": "success"}

        mock_active_state.side_effect = [
            ValueError("New document active-state verification failed"),
            {"verified": True},
            {"verified": True}
        ]

        response = self.client.post(
            "/api/v1/documents/activate",
            headers=self.headers,
            data={
                "document_id": "doc-456",
                "logical_document_id": "policy-123",
                "version": "2",
                "expected_vectors": "8",
                "request_id": "request-456",
                "previous_document_id": "doc-123"
            }
        )

        self.assertEqual(response.status_code, 500)

        self.assertEqual(mock_activate.call_count, 4)

        recovery_calls = mock_activate.call_args_list[2:]

        self.assertEqual(
            recovery_calls[0].kwargs["document_id"],
            "doc-456"
        )
        self.assertFalse(recovery_calls[0].kwargs["is_active"])

        self.assertEqual(
            recovery_calls[1].kwargs["document_id"],
            "doc-123"
        )
        self.assertTrue(recovery_calls[1].kwargs["is_active"])


if __name__ == "__main__":
    unittest.main(verbosity=2)