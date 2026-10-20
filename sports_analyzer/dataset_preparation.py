"""Read-only audit of a YOLO detection dataset; no image or label repairs."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import platform

from PIL import Image, __version__ as pillow_version

from sports_analyzer.annotations.yolo import load_names

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_dataset(root, metadata_path, expected_names, tiny_side_pixels=5,
                  near_duplicate_distance=4):
    """Collect file/line errors, valid boxes and duplicate candidates.

    Uses the explicit images/{train,val}, labels/{train,val} layout. Source
    YAML supplies class names only: its relative `path` is not used here.
    Valid boxes from a file with bad rows remain visible for diagnosis.
    """
    root = Path(root).resolve()
    metadata_path = Path(metadata_path).resolve()
    names = load_names(metadata_path)
    if list(names[k] for k in sorted(names)) != list(expected_names):
        raise ValueError(f'Unexpected class mapping: {names}; expected {expected_names}')
    if tiny_side_pixels <= 0 or not 0 <= near_duplicate_distance <= 64:
        raise ValueError('Invalid tiny-box or dHash threshold')
    images, boxes, issues = [], [], []

    def issue(severity, code, path, detail, line=None):
        issues.append(dict(severity=severity, code=code,
                           path=Path(path).relative_to(root).as_posix(),
                           line=line, detail=detail))

    for split in ('train', 'val'):
        image_dir, label_dir = root / 'images' / split, root / 'labels' / split
        for folder in (image_dir, label_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f'Required dataset folder missing: {folder}')
        paths = sorted(p for p in image_dir.rglob('*') if p.is_file()
                       and p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            issue('error', 'empty_split', image_dir, 'No supported images')
        expected_labels, seen_keys = set(), set()
        for path in paths:
            key = path.relative_to(image_dir).with_suffix('')
            # Preserve dots within the filename stem.
            label_path = label_dir / (str(path.relative_to(image_dir).with_suffix('')) + '.txt')
            expected_labels.add(label_path)
            if key in seen_keys:
                issue('error', 'ambiguous_stem', path, 'Multiple images share a label path')
            seen_keys.add(key)
            row = dict(split=split, image=path.relative_to(root).as_posix(),
                       label=label_path.relative_to(root).as_posix(),
                       image_sha256=None, label_sha256=None, pixel_sha256=None,
                       dhash=None, width=None, height=None, mode=None,
                       valid_boxes=0, status='error')
            images.append(row)
            try:
                row['image_sha256'] = sha256(path)
                with Image.open(path) as image:
                    image.load()
                    row.update(width=image.width, height=image.height, mode=image.mode)
                    if image.getexif().get(274, 1) != 1:
                        issue('error', 'exif_orientation', path,
                              'Non-default orientation: reconcile image and label coordinates')
                    rgb = image.convert('RGB')
                    digest = hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode())
                    digest.update(rgb.tobytes())
                    row['pixel_sha256'] = digest.hexdigest()
                    pixels = rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS).tobytes()
                    bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
                    row['dhash'] = f'{sum(int(bit) << i for i, bit in enumerate(bits)):016x}'
            except (OSError, ValueError) as exc:
                issue('error', 'unreadable_image', path, str(exc))
                continue
            if not label_path.is_file():
                issue('error', 'missing_label', path, f'Missing {row["label"]}')
                continue
            try:
                row['label_sha256'] = sha256(label_path)
                lines = label_path.read_text(encoding='utf-8-sig').splitlines()
            except (OSError, UnicodeError) as exc:
                issue('error', 'unreadable_label', label_path, str(exc))
                continue
            seen_boxes = set()
            nonempty = 0
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                nonempty += 1
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f'Expected 5 fields, got {len(fields)}')
                    class_id = int(fields[0])
                    cx, cy, bw, bh = map(float, fields[1:])
                    if class_id not in names:
                        raise ValueError(f'Unknown class ID {class_id}')
                    if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                        raise ValueError('Non-finite coordinates')
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        raise ValueError('Invalid normalized center or dimensions')
                    xyxy = (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)
                    if min(xyxy) < -1e-6 or max(xyxy) > 1+1e-6:
                        raise ValueError('Bounding box extends outside image')
                except ValueError as exc:
                    issue('error', 'invalid_label', label_path, str(exc), number)
                    continue
                identity = (class_id, cx, cy, bw, bh)
                if identity in seen_boxes:
                    issue('warning', 'duplicate_box', label_path, 'Repeated annotation row', number)
                seen_boxes.add(identity)
                if min(xyxy) < 0 or max(xyxy) > 1:
                    issue('warning', 'rounding_overflow', label_path,
                          'Boundary overflow <= 1e-6; coordinates unchanged', number)
                pixel_w, pixel_h = bw*row['width'], bh*row['height']
                if min(pixel_w, pixel_h) < tiny_side_pixels:
                    issue('warning', 'tiny_box', label_path,
                          f'Box {pixel_w:.2f} x {pixel_h:.2f} px; inspect, do not auto-delete', number)
                boxes.append(dict(split=split, image=row['image'], line=number,
                                  class_id=class_id, label=names[class_id],
                                  x1=xyxy[0]*row['width'], y1=xyxy[1]*row['height'],
                                  x2=xyxy[2]*row['width'], y2=xyxy[3]*row['height'],
                                  width_pixels=pixel_w, height_pixels=pixel_h,
                                  area_fraction=bw*bh))
                row['valid_boxes'] += 1
            if nonempty == 0:
                issue('warning', 'empty_label', label_path,
                      'May be a valid negative image; verify visually')
        for path in sorted(label_dir.rglob('*.txt')):
            if path not in expected_labels:
                issue('error', 'orphan_label', path, 'No matching supported image')

    error_paths = {i['path'] for i in issues if i['severity'] == 'error'}
    for row in images:
        if row['image'] not in error_paths and row['label'] not in error_paths and row['label_sha256']:
            row['status'] = 'ok'
    duplicates = []
    readable = [r for r in images if r['dhash'] is not None]
    for i, left in enumerate(readable):
        for right in readable[i+1:]:
            distance = (int(left['dhash'], 16) ^ int(right['dhash'], 16)).bit_count()
            kind = ('byte_identical' if left['image_sha256'] == right['image_sha256'] else
                    'pixel_identical' if left['pixel_sha256'] == right['pixel_sha256'] else
                    'visual_candidate' if distance <= near_duplicate_distance else None)
            if kind:
                duplicates.append(dict(image_a=left['image'], image_b=right['image'],
                                       cross_split=left['split'] != right['split'],
                                       kind=kind, dhash_distance=distance))
    summary = dict(images=len(images), boxes=len(boxes),
                   images_by_split=dict(Counter(r['split'] for r in images)),
                   errors=sum(i['severity'] == 'error' for i in issues),
                   warnings=sum(i['severity'] == 'warning' for i in issues),
                   cross_split_exact_pairs=sum(d['cross_split'] and d['kind'] != 'visual_candidate'
                                               for d in duplicates),
                   cross_split_visual_candidates=sum(d['cross_split'] and d['kind'] == 'visual_candidate'
                                                     for d in duplicates))
    return dict(root=str(root), names=names, images=images, boxes=boxes, issues=issues,
                duplicates=duplicates, summary=summary,
                metadata=dict(path=str(metadata_path), sha256=sha256(metadata_path)),
                settings=dict(tiny_side_pixels=tiny_side_pixels,
                              near_duplicate_distance=near_duplicate_distance))


def save_audit(result, output_root):
    """Always create a new report directory; even an invalid dataset is reportable."""
    target = Path(output_root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    fields = {
        'images': ['split', 'image', 'label', 'image_sha256', 'label_sha256', 'pixel_sha256',
                   'dhash', 'width', 'height', 'mode', 'valid_boxes', 'status'],
        'boxes': ['split', 'image', 'line', 'class_id', 'label', 'x1', 'y1', 'x2', 'y2',
                  'width_pixels', 'height_pixels', 'area_fraction'],
        'issues': ['severity', 'code', 'path', 'line', 'detail'],
        'duplicates': ['image_a', 'image_b', 'cross_split', 'kind', 'dhash_distance'],
    }
    for name, columns in fields.items():
        with (target / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(result[name])
    metadata = {k: v for k, v in result.items() if k not in fields}
    metadata.update(python=platform.python_version(), pillow=pillow_version,
                    audit_code_sha256=sha256(Path(__file__)),
                    created_utc=datetime.now(timezone.utc).isoformat())
    (target / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return target
