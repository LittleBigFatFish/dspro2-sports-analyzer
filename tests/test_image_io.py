from io import BytesIO
import unittest
from unittest.mock import patch

from PIL import Image

from sports_analyzer.image_io import decode_image, encode_png


def image_bytes(image, format='PNG', **kwargs):
    buffer = BytesIO()
    image.save(buffer, format=format, **kwargs)
    return buffer.getvalue()


class ImageIOTests(unittest.TestCase):
    def test_transparency_is_white_and_download_is_png(self):
        result = decode_image(image_bytes(Image.new('RGBA', (10, 20), (255, 0, 0, 0))))
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        with Image.open(BytesIO(encode_png(result))) as downloaded:
            self.assertEqual(downloaded.format, 'PNG')
            self.assertEqual(downloaded.size, (10, 20))
            self.assertEqual(downloaded.tobytes(), result.tobytes())

    def test_exif_orientation_applied_once(self):
        exif = Image.Exif()
        exif[274] = 6
        result = decode_image(image_bytes(Image.new('RGB', (30, 10)), 'JPEG', exif=exif))
        self.assertEqual(result.size, (10, 30))
        with Image.open(BytesIO(encode_png(result))) as downloaded:
            self.assertNotIn(274, downloaded.getexif())

    def test_rejects_invalid_unsupported_and_truncated_content(self):
        png = image_bytes(Image.new('RGB', (10, 10)))
        gif = image_bytes(Image.new('RGB', (10, 10)), 'GIF')
        for data in (b'', b'not an image', png[:40], gif):
            with self.subTest(data=data[:10]), self.assertRaises(ValueError):
                decode_image(data)

    def test_limits_checked(self):
        data = image_bytes(Image.new('RGB', (10, 10)))
        with patch('sports_analyzer.image_io.MAX_UPLOAD_BYTES', 5):
            with self.assertRaisesRegex(ValueError, '10 MiB'):
                decode_image(data)
        with patch('sports_analyzer.image_io.MAX_IMAGE_PIXELS', 50):
            with self.assertRaisesRegex(ValueError, 'megapixels'):
                decode_image(data)


if __name__ == '__main__':
    unittest.main()
