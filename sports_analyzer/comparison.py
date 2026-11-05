"""Raw predictions of pretrained models for a fixed image view; scoring is separate."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import json
import platform
import random
import re
import statistics
import subprocess

from .dataset_preparation import IMAGE_SUFFIXES, sha256


def source_group(relative) -> str:
    # Only a hint from the filename, not verified match metadata.
    stem = Path(relative).stem
    recording = re.fullmatch(r'(\d{8}_\d{6})_[0-9a-f]+', stem)
    if recording:
        return recording.group(1)
    clip = re.fullmatch(r'(.+?)[-_]*mp4-\d+', stem) or re.fullmatch(r'(.+?)_\d+', stem)
    return clip.group(1) if clip else stem


def yolo_view(root: Path, splits) -> list[dict]:
    frames = {}
    if (root / 'frames.csv').is_file():
        with (root / 'frames.csv').open(encoding='utf-8', newline='') as stream:
            frames = {row['image']: row for row in csv.DictReader(stream)}
    view = []
    for split in splits:
        folder = root / 'images' / split
        if not folder.is_dir():
            raise FileNotFoundError(f'Required dataset folder missing: {folder}')
        for path in sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            relative = path.relative_to(root).as_posix()
            if frames and relative not in frames:
                raise ValueError(f'{relative} missing from frames.csv')
            record = {'image': relative, 'split': split}
            if frames:
                record.update(game=frames[relative]['game'], source=frames[relative]['clip'],
                              frame=int(frames[relative]['frame']))
            else:
                record['source'] = source_group(relative)
            view.append(record)
    if len({r['image'] for r in view}) != len(view):
        raise ValueError('Duplicate image paths in dataset view')
    return view


def read_exclusions(path: Path) -> set[str]:
    return {line.split('\t')[0] for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()}


def filter_by_audit(view, audit_dir: Path, full_frames_only=False, apply_exclusions=False) -> list[dict]:
    with (audit_dir / 'images.csv').open(encoding='utf-8', newline='') as stream:
        crops = {row['image']: row.get('crop') == 'True' for row in csv.DictReader(stream)}
    excluded = read_exclusions(audit_dir / 'exclusions.txt') if apply_exclusions else set()
    # Images missing from the audit count as crops.
    return [r for r in view if r['image'] not in excluded and not (full_frames_only and crops.get(r['image'], True))]


def sample_view(view, max_images: int, seed: int) -> list[dict]:
    if len(view) <= max_images:
        return view
    return sorted(random.Random(seed).sample(view, max_images), key=lambda r: r['image'])


def group_counts(view) -> list[dict]:
    keys = [k for k in ('split', 'game', 'source') if k in view[0]]
    counts = Counter(tuple(r[k] for k in keys) for r in view)
    return [{**dict(zip(keys, group)), 'images': n} for group, n in sorted(counts.items())]


def view_digest(view) -> str:
    lines = sorted(f"{r['image']} {r['image_sha256']}" for r in view)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def check_against_audit(view, audit_dir: Path):
    with (audit_dir / 'images.csv').open(encoding='utf-8', newline='') as stream:
        audited = {row['image']: row['image_sha256'] for row in csv.DictReader(stream)}
    changed = [r['image'] for r in view if audited.get(r['image']) != r['image_sha256']]
    if changed:
        raise ValueError(f'{len(changed)} images differ from audit {audit_dir.name}, e.g. {changed[0]}')


def build_view(root: Path, settings: dict, splits=None, base=Path('.')) -> list[dict]:
    """One dataset entry of configs/model_comparison.yaml; paths are relative to base."""
    view = yolo_view(root, splits or settings['splits'])
    audit = base / settings['audit_run'] if settings.get('audit_run') else None
    if not audit and (settings.get('full_frames_only') or settings.get('apply_exclusions')):
        raise ValueError('Crop and exclusion filters need an audit_run')
    if audit:
        view = filter_by_audit(view, audit, settings.get('full_frames_only', False),
                               settings.get('apply_exclusions', False))
    excluded = set().union(*(read_exclusions(base / path) for path in settings.get('extra_exclusions', [])))
    groups = set(settings.get('exclude_groups', []))
    view = [r for r in view if r['image'] not in excluded and r['source'] not in groups]
    if settings.get('max_images'):
        view = sample_view(view, settings['max_images'], settings['seed'])
    for record in view:
        record['image_sha256'] = sha256(root / record['image'])
    if audit:
        check_against_audit(view, audit)
    return view


def serialise(record, boxes, speed) -> dict:
    return {**record, 'boxes': [{'label': b.label, 'xyxy': [round(v, 3) for v in b.xyxy],
                                 'score': round(b.score, 6)} for b in boxes],
            'speed_ms': {k: round(v, 3) for k, v in speed.items()}}


def load_predictions(path: Path):
    from .annotations.yolo import Box
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {r['image']: [Box(b['label'], tuple(b['xyxy']), b['score']) for b in r['boxes']]
            for r in data['images']}


def predict_view(detector, root: Path, view, imgsz: int, confidence: float) -> list[dict]:
    return [serialise(r, *detector.predict(root / r['image'], imgsz, confidence)) for r in view]


def time_view(detector, root: Path, view, imgsz: int, confidence: float, warmup: int) -> list[dict]:
    for r in view[:warmup]:
        detector.predict(root / r['image'], imgsz, confidence)
    rows = []
    for r in view:
        start = perf_counter()
        boxes, speed = detector.predict(root / r['image'], imgsz, confidence)
        rows.append({'image': r['image'], 'end_to_end_ms': (perf_counter()-start)*1000,
                     'preprocess_ms': speed['preprocess'], 'inference_ms': speed['inference'],
                     'postprocess_ms': speed['postprocess'], 'boxes': len(boxes)})
    return rows


def summarise_timings(values) -> dict:
    return {'n': len(values), 'median_ms': statistics.median(values),
            'p90_ms': statistics.quantiles(values, n=10, method='inclusive')[8]}


def environment(device) -> dict:
    import torch
    import ultralytics
    from PIL import __version__ as pillow_version
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    gpu = str(device) != 'cpu' and torch.cuda.is_available()
    return {'device': str(device),
            'device_name': torch.cuda.get_device_name(int(device)) if gpu else platform.processor(),
            'torch_threads': torch.get_num_threads(), 'python': platform.python_version(),
            'torch': torch.__version__, 'cuda': torch.version.cuda if gpu else None,
            'ultralytics': ultralytics.__version__, 'pillow': pillow_version,
            'platform': platform.platform(), 'git_commit': commit, 'git_dirty': dirty,
            'code_sha256': sha256(Path(__file__))}


def new_run_dir(output_root: Path) -> Path:
    target = Path(output_root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    return target


def save_predictions(output_root: Path, images, manifest) -> Path:
    target = new_run_dir(output_root)
    (target / 'predictions.json').write_text(json.dumps({'images': images}), encoding='utf-8')
    manifest = {**manifest, 'predictions_sha256': sha256(target / 'predictions.json'),
                'created_utc': datetime.now(timezone.utc).isoformat()}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return target
