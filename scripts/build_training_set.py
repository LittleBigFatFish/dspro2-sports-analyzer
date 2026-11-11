"""Named training sets from configs/training_set.yaml plus a run folder with checks and overlays."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import platform
import random
import sys
import warnings

import yaml
from PIL import Image, ImageDraw, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports_analyzer.annotations.yolo import read_labels
from sports_analyzer.dataset_preparation import git_state, sha256
from sports_analyzer.soccernet import NAMES
from sports_analyzer.training_set import DROPPED, build_set, prepare, verify_set, write_csv

COLORS = {'ball': 'red', 'player': 'deepskyblue', 'referee': 'yellow'}


def save_overlays(set_dir, target, count, seed):
    # Resized frames with their unchanged labels, plus a crop around one ball.
    with (set_dir / 'images.csv').open(encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream) if r['image_transform'].startswith('resized')]
    picks = random.Random(seed).sample(rows, min(count, len(rows)))
    folder = target / 'overlays'
    folder.mkdir(parents=True)
    for number, row in enumerate(picks):
        with Image.open(set_dir / row['image']) as handle:
            frame = handle.convert('RGB')
        label = set_dir / 'labels' / Path(row['image']).relative_to('images').with_suffix('.txt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            boxes = read_labels(label, frame.width, frame.height, NAMES)
        draw = ImageDraw.Draw(frame)
        for box in boxes:
            draw.rectangle(box.xyxy, outline=COLORS[box.label], width=2)
        stem = f'{number:02d}_{Path(row["image"]).stem}'
        frame.save(folder / f'{stem}.jpg', quality=90)
        balls = [b.xyxy for b in boxes if b.label == 'ball']
        if balls:
            x1, y1, x2, y2 = balls[0]
            side = max(160, 6*max(x2-x1, y2-y1))
            left, top = int((x1+x2-side)/2), int((y1+y2-side)/2)
            frame.crop((left, top, left+int(side), top+int(side))).resize((480, 480)).save(
                folder / f'{stem}_ball.jpg', quality=90)
    return [r['image'] for r in picks]


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/training_set.yaml')
    parser.add_argument('--set', action='append', help='entry under sets (default: all)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--overlays', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    names = args.set or list(config['sets'])
    unknown = sorted(set(names) - set(config['sets']))
    if unknown:
        parser.error(f'unknown sets {unknown}')
    prepared = prepare(config, ROOT, args.workers)
    target = ROOT / config['runs'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    write_csv(target / 'near_duplicates.csv', prepared['matches'],
              ['reference', 'reference_image', 'dataset', 'source_file', 'split', 'kind', 'dhash_distance',
               'mirror_distance'])
    write_csv(target / 'dropped.csv', prepared['dropped'], DROPPED)
    info = dict(config_file=args.config.resolve().relative_to(ROOT).as_posix(), config_sha256=sha256(args.config),
                script_sha256=sha256(Path(__file__)), git=git_state(), run=target.relative_to(ROOT).as_posix())
    sets, overlays = {}, {}
    for name in names:
        output = ROOT / config['output'] / name
        manifest = build_set(name, config, prepared, output, info)
        checks = verify_set(output, prepared['report_sha256'])
        sets[name] = dict(folder=output.relative_to(ROOT).as_posix(), images=manifest['images'],
                          images_by_source=manifest['images_by_source'], boxes_by_class=manifest['boxes_by_class'],
                          dropped=manifest['dropped'], ball_width_height=manifest['ball_width_height'],
                          manifest_sha256=sha256(output / 'manifest.json'), checks=checks)
        if args.overlays and any(k.endswith('/resized') for k in manifest['ball_width_height']):
            overlays[name] = save_overlays(output, target / name, args.overlays, args.seed)
    run = dict(sets=sets, overlays=overlays, matches=len(prepared['matches']), dropped=len(prepared['dropped']),
               report_images=len(prepared['report_sha256']), arguments={k: str(v) for k, v in vars(args).items()},
               **info, python=platform.python_version(), pillow=pillow_version, platform=platform.platform(),
               created_utc=datetime.now(timezone.utc).isoformat())
    (target / 'manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=info['run'], sets={k: dict(images=v['images'], images_by_source=v['images_by_source'],
                                                         dropped=v['dropped']) for k, v in sets.items()}), indent=2))


if __name__ == '__main__':
    main()
