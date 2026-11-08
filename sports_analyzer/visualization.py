"""Render detection boxes independently of the model and user interface."""
from collections.abc import Iterable
import math

from PIL import Image, ImageDraw, ImageFont

from .annotations.yolo import Box


COLORS = {
    'person': '#38BDF8',
    'player': '#38BDF8',
    'ball': '#FACC15',
    'sports ball': '#FACC15',
    'referee': '#FB923C',
}
DEFAULT_COLOR = '#C084FC'


def draw_detections(image: Image.Image, boxes: Iterable[Box]) -> Image.Image:
    """Return an annotated RGB copy; leave the image and boxes unchanged.

    Coordinates must use xyxy pixels relative to this image. Intersecting boxes
    are clipped for display only. Invalid or entirely outside boxes raise errors.
    Scores are optional, so ground-truth annotations can also be rendered.
    """
    if not isinstance(image, Image.Image):
        raise TypeError('draw_detections expects a decoded PIL image')
    if image.width <= 0 or image.height <= 0:
        raise ValueError('Image dimensions must be positive')

    annotated = image.convert('RGB').copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    line_width = max(2, round(min(width, height) / 400))
    font = ImageFont.load_default(size=max(12, round(min(width, height) / 50)))
    padding = 3

    for index, box in enumerate(boxes):
        if not isinstance(box, Box):
            raise TypeError(f'Box {index}: expected a Box instance')
        if not isinstance(box.label, str) or not box.label.strip():
            raise ValueError(f'Box {index}: label must be non-empty text')
        if len(box.xyxy) != 4 or not all(math.isfinite(v) for v in box.xyxy):
            raise ValueError(f'Box {index}: expected four finite coordinates')
        x1, y1, x2, y2 = box.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'Box {index}: width and height must be positive')
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            raise ValueError(f'Box {index}: box is outside the image')
        if box.score is not None and (
            not math.isfinite(box.score) or not 0 <= box.score <= 1
        ):
            raise ValueError(f'Box {index}: score must be finite and between 0 and 1')

        # Clipping affects drawing only, never the source annotations.
        left = max(0, min(width - 1, x1))
        top = max(0, min(height - 1, y1))
        right = max(0, min(width - 1, x2))
        bottom = max(0, min(height - 1, y2))
        color = COLORS.get(box.label, DEFAULT_COLOR)
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)

        caption = box.label if box.score is None else f'{box.label} {box.score:.2f}'
        text_left, text_top, text_right, text_bottom = draw.textbbox(
            (0, 0), caption, font=font
        )
        label_width = text_right - text_left + 2 * padding
        label_height = text_bottom - text_top + 2 * padding
        label_x = max(0, min(left, width - label_width))
        # Prefer above the box; at the top edge put the caption inside the image.
        label_y = top - label_height if top >= label_height else top
        label_y = max(0, min(label_y, height - label_height))
        draw.rectangle(
            (label_x, label_y, label_x + label_width - 1, label_y + label_height - 1),
            fill=color,
        )
        draw.text(
            (label_x + padding - text_left, label_y + padding - text_top),
            caption, font=font, fill='#111827',
        )

    return annotated
