"""UI state tests use a fake detector; real inference is tested separately."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from sports_analyzer.annotations.yolo import Box

APP = Path(__file__).resolve().parents[1] / 'app/streamlit_app.py'


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def predict_image(self, image, imgsz, confidence):
        self.calls += 1
        return self.boxes, {'predict_call_ms': 12.0}


class StreamlitAppTests(unittest.TestCase):
    def upload(self):
        buffer = BytesIO()
        Image.new('RGB', (200, 100), 'green').save(buffer, format='PNG')
        return buffer

    def test_initial_page(self):
        with patch('streamlit.file_uploader', return_value=None):
            app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, 'Sports Analyzer')
        self.assertFalse(app.button)

    def test_analysis_rerun_and_settings_invalidation(self):
        detector = FakeDetector([Box('person', (20, 30, 50, 80), .9)])
        with patch('streamlit.file_uploader', return_value=self.upload()), \
                patch('pathlib.Path.is_file', return_value=True):
            app = AppTest.from_file(str(APP)).run()
            app.session_state['detector'] = detector
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, '1')
            self.assertTrue(app.session_state['result']['png'].startswith(b'\x89PNG'))
            app.run()
            self.assertEqual(detector.calls, 1)
            app.slider[0].set_value(.5).run()
            self.assertNotIn('result', app.session_state)
            self.assertEqual(detector.calls, 1)

    def test_invalid_upload_and_missing_model(self):
        with patch('streamlit.file_uploader', return_value=BytesIO(b'invalid')):
            app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertIn('readable', app.error[0].value)
        app = AppTest.from_file(str(APP))
        with patch('streamlit.file_uploader', return_value=self.upload()), \
                patch('pathlib.Path.is_file', return_value=False):
            app.run()
        self.assertFalse(app.exception)
        self.assertIn('Model weights are missing', app.error[0].value)

    def test_empty_result_and_removing_upload(self):
        with patch('streamlit.file_uploader', return_value=self.upload()), \
                patch('pathlib.Path.is_file', return_value=True):
            app = AppTest.from_file(str(APP)).run()
            app.session_state['detector'] = FakeDetector([])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, '0')
            self.assertIn('result', app.session_state)
        with patch('streamlit.file_uploader', return_value=None):
            app.run()
        self.assertNotIn('result', app.session_state)
        self.assertNotIn('image', app.session_state)


if __name__ == '__main__':
    unittest.main()
