"""Read-only audit of a YOLO detection dataset; no image or label repairs."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import math
import platform
import random
import re
import shutil
import subprocess
import zipfile

from PIL import Image, ImageDraw, __version__ as pillow_version

from sports_analyzer.annotations.yolo import load_names

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
# SAHI slices like v12_sahi_160_1077.jpg, group 1 is the slice side.
CROP_PATTERN = r'(?:^|_)sahi_(\d+)_'
# Strongest first.
DUPLICATE_KINDS = ('byte_identical', 'pixel_identical', 'mirror_identical',
                   'visual_candidate', 'mirror_candidate')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dhash(rgb):
    pixels = rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
    return f'{sum(int(bit) << i for i, bit in enumerate(bits)):016x}'


def pixel_sha256(rgb):
    digest = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def image_hashes(image):
    rgb = image.convert('RGB')
    mirror = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return dict(pixel_sha256=pixel_sha256(rgb), dhash=dhash(rgb),
                mirror_pixel_sha256=pixel_sha256(mirror), mirror_dhash=dhash(mirror))


def hamming_matches(queries, hashes, distance):
    # With distance+1 bands, any match agrees exactly on at least one band.
    bounds = [64*k // (distance+1) for k in range(distance+2)]
    bands = [(start, (1 << (end-start)) - 1) for start, end in zip(bounds, bounds[1:])]
    index = defaultdict(list)
    for j, value in enumerate(hashes):
        for band, (shift, mask) in enumerate(bands):
            index[band, value >> shift & mask].append(j)
    for i, value in enumerate(queries):
        seen = set()
        for band, (shift, mask) in enumerate(bands):
            for j in index.get((band, value >> shift & mask), ()):
                if j not in seen:
                    seen.add(j)
                    if (d := (value ^ hashes[j]).bit_count()) <= distance:
                        yield i, j, d


def duplicate_kind(left, right, distance):
    plain = (int(left['dhash'], 16) ^ int(right['dhash'], 16)).bit_count()
    mirrored = min((int(left['dhash'], 16) ^ int(right['mirror_dhash'], 16)).bit_count(),
                   (int(left['mirror_dhash'], 16) ^ int(right['dhash'], 16)).bit_count())
    kind = ('byte_identical' if left['image_sha256'] == right['image_sha256'] else
            'pixel_identical' if left['pixel_sha256'] == right['pixel_sha256'] else
            'mirror_identical' if left['pixel_sha256'] == right['mirror_pixel_sha256'] else
            'visual_candidate' if plain <= distance else
            'mirror_candidate' if mirrored <= distance else None)
    return kind, plain, mirrored


def find_duplicates(rows, distance):
    plain = [int(r['dhash'], 16) for r in rows]
    mirror = [int(r['mirror_dhash'], 16) for r in rows]
    pairs = {(i, j) for i, j, _ in hamming_matches(plain, plain, distance) if i < j}
    pairs |= {(min(i, j), max(i, j)) for i, j, _ in hamming_matches(mirror, plain, distance) if i != j}
    by_pixels = defaultdict(list)
    for i, row in enumerate(rows):
        by_pixels[row['pixel_sha256']].append(i)
    for i, row in enumerate(rows):
        pairs |= {(min(i, j), max(i, j)) for j in by_pixels.get(row['mirror_pixel_sha256'], ()) if i != j}
    for i, j in sorted(pairs):
        kind, plain_distance, mirror_distance = duplicate_kind(rows[i], rows[j], distance)
        if kind:
            yield i, j, kind, plain_distance, mirror_distance


def audit_dataset(root, metadata_path, expected_names, tiny_side_pixels=5,
                  near_duplicate_distance=4, splits=('train', 'val'), crop_pattern=CROP_PATTERN):
    """Collect file/line errors, valid boxes, duplicates and an exclusion list.

    Uses the explicit images/<split>, labels/<split> layout. Of a cross-split
    duplicate, the image in the later split is excluded. Source YAML supplies
    class names only: its relative `path` is not used here.
    Valid boxes from a file with bad rows remain visible for diagnosis.
    """
    root = Path(root).resolve()
    metadata_path = Path(metadata_path).resolve()
    names = load_names(metadata_path)
    if list(names[k] for k in sorted(names)) != list(expected_names):
        raise ValueError(f'Unexpected class mapping: {names}; expected {expected_names}')
    if tiny_side_pixels <= 0 or not 0 <= near_duplicate_distance < 64:
        raise ValueError('Invalid tiny-box or dHash threshold')
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError(f'Splits must be unique and non-empty: {splits}')
    images, boxes, issues = [], [], []

    def issue(severity, code, path, detail, line=None):
        issues.append(dict(severity=severity, code=code,
                           path=Path(path).relative_to(root).as_posix(),
                           line=line, detail=detail))

    for split in splits:
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
                       dhash=None, mirror_pixel_sha256=None, mirror_dhash=None,
                       width=None, height=None, mode=None, crop=None,
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
                    row.update(image_hashes(image))
            except (OSError, ValueError) as exc:
                issue('error', 'unreadable_image', path, str(exc))
                continue
            match = re.search(crop_pattern, path.name) if crop_pattern else None
            row['crop'] = bool(match) and (not match.groups()
                                           or max(row['width'], row['height']) <= int(match.group(1)))
            if match and not row['crop']:
                issue('warning', 'crop_size_mismatch', path,
                      f'Crop name but {row["width"]} x {row["height"]} px exceeds the slice side')
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
                    if not (bw > 0 and bh > 0):
                        raise ValueError('Invalid normalized center or dimensions')
                except ValueError as exc:
                    issue('error', 'invalid_label', label_path, str(exc), number)
                    continue
                xyxy = (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)
                if min(xyxy) < -1e-6 or max(xyxy) > 1+1e-6:
                    issue('error', 'out_of_bounds', label_path, 'Box extends outside image: '
                          + ' '.join(f'{v:.6f}' for v in xyxy), number)
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

    error_codes = defaultdict(set)
    for i in issues:
        if i['severity'] == 'error':
            error_codes[i['path']].add(i['code'])
    exclusions = defaultdict(set)
    for row in images:
        codes = error_codes.get(row['image'], set()) | error_codes.get(row['label'], set())
        if codes:
            exclusions[row['image']] |= codes
        elif row['label_sha256']:
            row['status'] = 'ok'
    duplicates, leaks = [], {}
    readable = [r for r in images if r['dhash'] is not None]
    for i, j, kind, distance, mirror_distance in find_duplicates(readable, near_duplicate_distance):
        left, right = readable[i], readable[j]
        duplicates.append(dict(image_a=left['image'], image_b=right['image'],
                               cross_split=left['split'] != right['split'], kind=kind,
                               dhash_distance=distance, mirror_distance=mirror_distance))
        if left['split'] != right['split']:
            later = max(left, right, key=lambda r: splits.index(r['split']))['image']
            if later not in leaks or DUPLICATE_KINDS.index(kind) < DUPLICATE_KINDS.index(leaks[later]):
                leaks[later] = kind
    for image, kind in leaks.items():
        exclusions[image].add(f'leak_{kind}')
    out_of_bounds = [i for i in issues if i['code'] == 'out_of_bounds']
    summary = dict(images=len(images), boxes=len(boxes),
                   images_by_split=dict(Counter(r['split'] for r in images)),
                   boxes_by_class=dict(Counter(b['label'] for b in boxes)),
                   errors=sum(i['severity'] == 'error' for i in issues),
                   warnings=sum(i['severity'] == 'warning' for i in issues),
                   crops=sum(bool(r['crop']) for r in images),
                   empty_labels=sum(i['code'] == 'empty_label' for i in issues),
                   out_of_bounds_rows=len(out_of_bounds),
                   out_of_bounds_images=len({i['path'] for i in out_of_bounds}),
                   cross_split_exact_pairs=sum(d['cross_split'] and d['kind'] in DUPLICATE_KINDS[:2]
                                               for d in duplicates),
                   cross_split_visual_candidates=sum(d['cross_split'] and d['kind'] == 'visual_candidate'
                                                     for d in duplicates),
                   cross_split_mirror_pairs=sum(d['cross_split'] and d['kind'].startswith('mirror')
                                                for d in duplicates),
                   leaked_images_by_kind=dict(Counter(leaks.values())),
                   excluded_images=len(exclusions))
    return dict(root=str(root), names=names, images=images, boxes=boxes, issues=issues,
                duplicates=duplicates, summary=summary,
                exclusions=[dict(image=k, reasons=';'.join(sorted(v))) for k, v in sorted(exclusions.items())],
                metadata=dict(path=str(metadata_path), sha256=sha256(metadata_path)),
                settings=dict(tiny_side_pixels=tiny_side_pixels, splits=list(splits),
                              near_duplicate_distance=near_duplicate_distance,
                              crop_pattern=crop_pattern))


def git_state(root=Path(__file__).resolve().parents[1]):
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')))
    except (OSError, subprocess.CalledProcessError):
        return None


def save_audit(result, output_root):
    """Always create a new report directory; even an invalid dataset is reportable."""
    target = Path(output_root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    fields = {
        'images': ['split', 'image', 'label', 'image_sha256', 'label_sha256', 'pixel_sha256',
                   'dhash', 'mirror_pixel_sha256', 'mirror_dhash', 'width', 'height', 'mode',
                   'crop', 'valid_boxes', 'status'],
        'boxes': ['split', 'image', 'line', 'class_id', 'label', 'x1', 'y1', 'x2', 'y2',
                  'width_pixels', 'height_pixels', 'area_fraction'],
        'issues': ['severity', 'code', 'path', 'line', 'detail'],
        'duplicates': ['image_a', 'image_b', 'cross_split', 'kind', 'dhash_distance', 'mirror_distance'],
    }
    for name, columns in fields.items():
        with (target / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(result[name])
    (target / 'exclusions.txt').write_text(''.join(f'{r["image"]}\t{r["reasons"]}\n'
                                                   for r in result['exclusions']), encoding='utf-8')
    metadata = {k: v for k, v in result.items() if k not in fields and k != 'exclusions'}
    metadata.update(python=platform.python_version(), pillow=pillow_version,
                    audit_code_sha256=sha256(Path(__file__)), git=git_state(),
                    created_utc=datetime.now(timezone.utc).isoformat())
    (target / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return target


def save_overlays(result, target, per_class=20, seed=42, tile=256):
    root, rng = Path(result['root']), random.Random(seed)
    for class_id, name in sorted(result['names'].items()):
        group = [b for b in result['boxes'] if b['class_id'] == class_id]
        chosen = rng.sample(group, min(per_class, len(group)))
        folder = Path(target) / 'overlays' / name
        folder.mkdir(parents=True, exist_ok=True)
        sheet = Image.new('RGB', (5*tile, max(1, math.ceil(len(chosen)/5))*tile), 'white')
        for k, box in enumerate(chosen):
            side = round(3*max(box['width_pixels'], box['height_pixels'], 16))
            left, top = round((box['x1']+box['x2']-side)/2), round((box['y1']+box['y2']-side)/2)
            with Image.open(root / box['image']) as image:
                view = image.convert('RGB').crop((left, top, left+side, top+side)).resize((tile, tile))
            scale = tile/side
            ImageDraw.Draw(view).rectangle(((box['x1']-left)*scale, (box['y1']-top)*scale,
                                            (box['x2']-left)*scale, (box['y2']-top)*scale),
                                           outline='red', width=2)
            view.save(folder / f'{k:02d}_{Path(box["image"]).stem}_L{box["line"]}.jpg', quality=90)
            sheet.paste(view, (k % 5 * tile, k // 5 * tile))
        sheet.save(Path(target) / 'overlays' / f'{name}.jpg', quality=90)
    return Path(target) / 'overlays'


def extract_zip(archive, target, strip_prefix='', skip_suffixes=()):
    """Check every member before extracting anything; returns skipped members."""
    target = Path(target)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            name = member.filename
            if (name.startswith('/') or '\\' in name or ':' in name or '..' in PurePosixPath(name).parts
                    or not name.startswith(strip_prefix)):
                raise ValueError(f'Unsafe or unexpected archive member: {name}')
        target.mkdir(parents=True, exist_ok=False)
        skipped = []
        for member in members:
            relative = member.filename[len(strip_prefix):]
            if member.is_dir() or not relative:
                continue
            if relative.endswith(tuple(skip_suffixes)):
                skipped.append(member.filename)
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as stream, destination.open('xb') as output:
                shutil.copyfileobj(stream, output)
    return skipped
