"""Decode uploaded still images and encode results without temporary files."""
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_image(data: bytes) -> Image.Image:
    """Validate JPEG/PNG content, apply EXIF orientation and return RGB pixels."""
    if not data:
        raise ValueError('The file is empty.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('The image must not exceed 10 MiB.')
    try:
        with Image.open(BytesIO(data), formats=('JPEG', 'PNG')) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError('The image must not exceed 20 megapixels.')
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Please upload a still image rather than an animation.')
            source.verify()
        # verify() invalidates the decoder; reopen for the actual pixel decode.
        with Image.open(BytesIO(data), formats=('JPEG', 'PNG')) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            return Image.alpha_composite(background, rgba).convert('RGB')
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('The file is not a readable JPEG or PNG image.') from exc


def encode_png(image: Image.Image) -> bytes:
    """Return a PNG download containing rendered pixels, without source metadata."""
    clean = Image.new('RGB', image.size)
    clean.paste(image)
    buffer = BytesIO()
    clean.save(buffer, format='PNG')
    return buffer.getvalue()
