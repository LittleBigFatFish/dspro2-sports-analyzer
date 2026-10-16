"""Strict YOLO detection annotations. Source coordinates are never repaired."""
from dataclasses import dataclass
from pathlib import Path
import math
import warnings
import yaml

@dataclass(frozen=True)
class Box:
    label: str
    xyxy: tuple[float, float, float, float]
    score: float | None = None


def load_names(path: Path) -> dict[int, str]:
    with path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or 'names' not in config:
        raise ValueError(f'{path}: missing names mapping')
    names = config['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError(f'{path}: invalid names')
    if any(type(k) is not int or not isinstance(v, str) for k, v in names.items()):
        raise ValueError(f'{path}: class IDs must be integers and names strings')
    if set(names) != set(range(len(names))) or len(set(names.values())) != len(names):
        raise ValueError(f'{path}: IDs must be contiguous and names unique')
    if config.get('nc', len(names)) != len(names):
        raise ValueError(f'{path}: nc does not match names')
    return names


def read_labels(path: Path, width: int, height: int, names: dict[int, str]) -> list[Box]:
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    boxes = []
    # Missing file is deliberately an error; an existing empty file is valid.
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        location = f'{path}:{number}'
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'{location}: expected 5 fields, got {len(fields)}')
        try:
            class_id = int(fields[0])
            cx, cy, bw, bh = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f'{location}: invalid numeric field') from error
        if class_id not in names:
            raise ValueError(f'{location}: unknown class {class_id}')
        if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
            raise ValueError(f'{location}: non-finite coordinate')
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f'{location}: invalid normalized centre or size')
        normalized = (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)
        if min(normalized) < -1e-6 or max(normalized) > 1+1e-6:
            raise ValueError(f'{location}: box extends outside image')
        if min(normalized) < 0 or max(normalized) > 1:
            warnings.warn(f'{location}: rounding-sized boundary overflow; unchanged', stacklevel=2)
        if min(bw*width, bh*height) < 5:
            warnings.warn(f'{location}: box side below 5 pixels; inspect, do not delete', stacklevel=2)
        x1, y1, x2, y2 = normalized
        boxes.append(Box(names[class_id], (x1*width, y1*height, x2*width, y2*height)))
    return boxes
