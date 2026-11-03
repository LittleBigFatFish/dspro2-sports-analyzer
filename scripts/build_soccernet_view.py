"""Sampled SoccerNet Tracking YOLO view plus a run folder with overlays."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import platform
import random
import subprocess
import sys
import warnings

from PIL import Image, ImageDraw, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports_analyzer.annotations.yolo import read_labels
from sports_analyzer.dataset_preparation import sha256
from sports_analyzer.soccernet import IGNORE, NAMES, assert_disjoint_games, build_view, split_games

COLORS = {'ball': 'red', 'player': 'deepskyblue', 'referee': 'yellow', IGNORE: 'gray'}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def view_boxes(view, frames):
    ignores = {}
    with (view / 'ignore.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            ignores.setdefault(row['image'], []).append(
                (IGNORE, tuple(float(row[k]) for k in ('x1', 'y1', 'x2', 'y2'))))
    boxes = {}
    for row in frames:
        image = view / row['image']
        label = view / 'labels' / Path(row['image']).relative_to('images').with_suffix('.txt')
        with Image.open(image) as handle:
            size = handle.size
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            found = [(b.label, b.xyxy) for b in read_labels(label, *size, NAMES)]
        boxes[row['image']] = found + ignores.get(row['image'], [])
    return boxes


def save_overlays(view, frames, target, per_class, seed):
    boxes = view_boxes(view, frames)
    rng = random.Random(seed)
    written = {}
    for label in [*NAMES.values(), IGNORE]:
        candidates = [(image, xyxy) for image, items in boxes.items() for name, xyxy in items if name == label]
        picks = rng.sample(candidates, min(per_class, len(candidates)))
        folder = target / 'overlays' / label
        folder.mkdir(parents=True)
        for number, (image, (x1, y1, x2, y2)) in enumerate(picks):
            with Image.open(view / image) as handle:
                frame = handle.convert('RGB')
            draw = ImageDraw.Draw(frame)
            for name, xyxy in boxes[image]:
                draw.rectangle(xyxy, outline=COLORS[name], width=1)
            draw.rectangle((x1-3, y1-3, x2+3, y2+3), outline=COLORS[label], width=3)
            side = max(240, 3*max(x2-x1, y2-y1))
            cx, cy = (x1+x2)/2, (y1+y2)/2
            left = int(min(max(cx-side/2, 0), frame.width-side))
            top = int(min(max(cy-side/2, 0), frame.height-side))
            crop = frame.crop((left, top, left+int(side), top+int(side))).resize((480, 480))
            crop.save(folder / f'{number:02d}_{Path(image).stem}.jpg', quality=92)
        written[label] = dict(available=len(candidates), saved=len(picks))
    return written


def main():
    parser = ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='split folder with SNMOT-* clips')
    parser.add_argument('--split', required=True)
    parser.add_argument('--step', type=int, required=True, help='keep every n-th frame from frame 1')
    parser.add_argument('--rate', required=True, help='view name suffix, e.g. 1fps')
    parser.add_argument('--other-split', type=Path, action='append', default=[],
                        help='further split folders whose games must not overlap')
    parser.add_argument('--source-json', type=Path, default=ROOT / 'data/raw/sn_tracking_2023.source.json')
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/processed/soccernet')
    parser.add_argument('--runs', type=Path, default=ROOT / 'outputs/soccernet_view')
    parser.add_argument('--overlays', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    splits = {args.split: set(split_games(args.source).values())}
    splits.update({p.name: set(split_games(p).values()) for p in args.other_split})
    games = assert_disjoint_games(splits)
    view = args.processed / f'{args.split}_{args.rate}'
    manifest = build_view(args.source, view, args.split, args.step)
    target = args.runs / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    with (view / 'frames.csv').open(encoding='utf-8') as stream:
        frames = list(csv.DictReader(stream))
    overlays = save_overlays(view, frames, target, args.overlays, args.seed) if args.overlays else {}
    run = dict(view=str(view.resolve()), view_manifest_sha256=sha256(view / 'manifest.json'),
               source_json=str(args.source_json), source_json_sha256=(
                   sha256(args.source_json) if args.source_json.is_file() else None),
               games_by_split=games, arguments={k: str(v) for k, v in vars(args).items()},
               overlays=overlays, summary=manifest, git_commit=git('rev-parse', 'HEAD'),
               git_dirty=bool(git('status', '--porcelain')), python=platform.python_version(),
               pillow=pillow_version, device='cpu', platform=platform.platform(),
               script_sha256=sha256(Path(__file__)),
               created_utc=datetime.now(timezone.utc).isoformat())
    (target / 'manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=str(target), view=str(view), counts=manifest['counts'],
                          games=manifest['games'], overlays=overlays), indent=2))


if __name__ == '__main__':
    main()
