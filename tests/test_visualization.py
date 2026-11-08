import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from sports_analyzer.annotations.yolo import Box
from sports_analyzer.visualization import draw_detections


class VisualizationTests(unittest.TestCase):
    def test_draws_boxes_without_modifying_inputs(self):
        image = Image.new('RGB', (200, 150), 'white')
        original = image.tobytes()
        box = Box('person', (20, 40, 120, 120), .87)
        result = draw_detections(image, [box])
        self.assertEqual(image.tobytes(), original)
        self.assertEqual(box.xyxy, (20, 40, 120, 120))
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((20, 100)), (56, 189, 248))
        self.assertEqual(result.getpixel((80, 90)), (255, 255, 255))
        self.assertIsNot(result, image)

    def test_empty_detections_return_rgb_image(self):
        image = Image.new('L', (80, 60), 128)
        result = draw_detections(image, [])
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(image.mode, 'L')

    def test_captions_include_zero_score_and_support_ground_truth(self):
        image = Image.new('RGB', (200, 150), 'white')
        original_text = ImageDraw.ImageDraw.text
        captions = []

        def capture(draw, xy, text, *args, **kwargs):
            captions.append(text)
            return original_text(draw, xy, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, 'text', capture):
            draw_detections(image, [Box('ball', (10, 30, 20, 40), 0.0),
                                    Box('referee', (50, 50, 80, 120))])
        self.assertEqual(captions, ['ball 0.00', 'referee'])

    def test_boundary_clipping_preserves_annotation(self):
        box = Box('ball', (-.0001, 20, 100.0001, 80))
        image = Image.new('RGB', (100, 100), 'white')
        result = draw_detections(image, [box])
        self.assertEqual(result.getpixel((99, 60)), (250, 204, 21))
        self.assertEqual(box.xyxy, (-.0001, 20, 100.0001, 80))

    def test_invalid_annotations_raise(self):
        image = Image.new('RGB', (100, 100), 'white')
        invalid = [Box('ball', (10, 10, 5, 20)),
                   Box('ball', (0, 0, float('nan'), 20)),
                   Box('ball', (110, 10, 120, 20)),
                   Box('ball', (10, 10, 20, 20), 1.1),
                   Box('', (10, 10, 20, 20))]
        for box in invalid:
            with self.subTest(box=box), self.assertRaises(ValueError):
                draw_detections(image, [box])


if __name__ == '__main__':
    unittest.main()
