import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app


class TestWordPressIntegration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.chat_token = "wp-proxy-secret-token-xyz"
        self.env_patcher = patch.dict(
            os.environ,
            {
                "REVON_CHAT_TOKEN": self.chat_token,
                "CHAT_RATE_LIMIT_PER_MINUTE": "3",  # Low limit to test rate limiting
            }
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_wordpress_proxy_visitor_rate_limiting(self):
        """Simulated WordPress proxy passing X-RevOn-Visitor-IP gets rate limited per visitor IP."""
        visitor_a_headers = {
            "X-RevOn-Chat-Token": self.chat_token,
            "X-RevOn-Visitor-IP": "203.0.113.195",
        }
        visitor_b_headers = {
            "X-RevOn-Chat-Token": self.chat_token,
            "X-RevOn-Visitor-IP": "198.51.100.42",
        }

        with patch("retrieval_service.analyze_query_intent", return_value=(False, None)):
            with patch("retrieval_service.generate_chat_response", return_value="Hello visitor!"):
                # Visitor A makes 3 allowed requests (limit = 3)
                for _ in range(3):
                    res = self.client.post("/api/v1/chat", headers=visitor_a_headers, json={"question": "Hi"})
                    self.assertEqual(res.status_code, 200)

                # 4th request from Visitor A should be rate-limited (429)
                res_a4 = self.client.post("/api/v1/chat", headers=visitor_a_headers, json={"question": "Hi"})
                self.assertEqual(res_a4.status_code, 429)
                self.assertIn("Too many chat requests", res_a4.json()["detail"])
                self.assertIn("Retry-After", res_a4.headers)

                # Visitor B should NOT be rate-limited despite Visitor A being blocked
                res_b1 = self.client.post("/api/v1/chat", headers=visitor_b_headers, json={"question": "Hi"})
                self.assertEqual(res_b1.status_code, 200)

    def test_wordpress_proxy_missing_visitor_ip_falls_back_to_peer(self):
        """When WordPress does not send visitor IP, backend gracefully falls back without crashing."""
        headers = {
            "X-RevOn-Chat-Token": self.chat_token,
        }
        with patch("retrieval_service.analyze_query_intent", return_value=(False, None)):
            with patch("retrieval_service.generate_chat_response", return_value="Hello!"):
                res = self.client.post("/api/v1/chat", headers=headers, json={"question": "Test fallback"})
                self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
