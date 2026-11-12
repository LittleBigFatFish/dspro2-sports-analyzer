"""Integration check using the local assets installed by scripts/setup.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from sports_analyzer.inference import Detector

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/yolo26n.pt"
FRAME = ROOT / "data/raw/Football-Image-Dataset/images/train/20250710_144746_004987.jpg"


class ImageInferenceTests(unittest.TestCase):
    @unittest.skipUnless(MODEL.is_file() and FRAME.is_file(), "requires local baseline assets")
    def test_file_and_decoded_image_detect_same_boxes(self):
        detector = Detector(MODEL)
        with Image.open(FRAME) as frame, TemporaryDirectory() as tmp:
            rgb = frame.convert("RGB")
            # PNG preserves pixels: this compares input handling, not JPEG decoders.
            path = Path(tmp) / "frame.png"
            rgb.save(path)
            expected, _ = detector.predict(path, imgsz=640, confidence=0.25)
            actual, _ = detector.predict_image(rgb, imgsz=640, confidence=0.25)

        self.assertTrue(expected, "Reference frame should contain detections")
        self.assertEqual([b.label for b in actual], [b.label for b in expected])
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a.score, b.score, delta=1e-5)
            for x, y in zip(a.xyxy, b.xyxy):
                self.assertAlmostEqual(x, y, delta=0.01)
