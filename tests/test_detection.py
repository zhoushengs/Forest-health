import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import app, gate


def picture(size=(800, 400)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (80, 110, 65)).save(buffer, "PNG")
    return buffer.getvalue()


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_real_model_and_image(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        health = self.client.get("/api/health").json()
        self.assertEqual(health["classes"], {"0": "notree", "1": "sicktree"})
        response = self.client.post("/api/detect", files={"file": ("forest.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["image_size"], [800, 400])
        self.assertGreater(body["inference_ms"], 0)
        annotated = Image.open(io.BytesIO(base64.b64decode(body["annotated_image"].split(",")[1])))
        self.assertEqual(annotated.size, (800, 400))
        for item in body["detections"]:
            self.assertEqual(item["class_name"], "sicktree")
            x1, y1, x2, y2 = item["box"]
            self.assertTrue(0 <= x1 < x2 <= 800 and 0 <= y1 < y2 <= 400)

    def test_letterbox_coordinate_mapping_nms_and_sicktree_filter(self):
        # 800x400 => 640x320, top padding 160. Target in original: 100,50,300,250.
        predictions = np.array([[[160, 160, 160, 500], [280, 280, 280, 300],
                                 [160, 160, 160, 20], [160, 160, 160, 20],
                                 [.9, .1, .1, .05], [.1, .85, .95, .05]]], dtype=np.float32)
        with patch.object(app.state.detector.session, "run", return_value=[predictions]):
            response = self.client.post("/api/detect", files={"file": ("a.png", picture(), "image/png")})
        body = response.json()
        self.assertEqual(response.status_code, 200, body)
        self.assertEqual(len(body["detections"]), 1)
        self.assertEqual(body["detections"][0]["class_name"], "sicktree")
        self.assertEqual(body["detections"][0]["box"], [100.0, 50.0, 300.0, 250.0])

    def test_empty_detections(self):
        # A high-confidence notree prediction must not appear in results.
        prediction = np.array([[[160], [280], [160], [160], [.99], [.01]]], dtype=np.float32)
        with patch.object(app.state.detector.session, "run", return_value=[prediction]):
            response = self.client.post("/api/detect", files={"file": ("a.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"], [])

    def test_invalid_file_and_parameters(self):
        self.assertEqual(self.client.post("/api/detect", files={"file": ("fake.jpg", b"invalid", "image/jpeg")}).status_code, 400)
        self.assertEqual(self.client.post("/api/detect", files={"file": ("a.png", picture())}, data={"confidence": "1.5"}).status_code, 422)
        self.assertEqual(self.client.post("/api/detect").status_code, 422)

    def test_upload_limit_and_busy_response(self):
        response = self.client.post("/api/detect", files={"file": ("huge.jpg", b"0" * (15 * 1024 * 1024 + 1))})
        self.assertEqual(response.status_code, 413)
        gate.acquire()
        try:
            self.assertEqual(self.client.post("/api/detect", files={"file": ("a.png", picture())}).status_code, 429)
        finally:
            gate.release()


if __name__ == "__main__":
    unittest.main()
