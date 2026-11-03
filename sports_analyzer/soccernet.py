"""SoccerNet Tracking (MOT) clips to a read-only YOLO test view."""
from collections import Counter
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os

from PIL import Image

from sports_analyzer.dataset_preparation import sha256

NAMES = {0: 'ball', 1: 'player', 2: 'referee'}
IGNORE = 'ignore'


def role_class(role: str):
    # Goalkeepers count as players, `other` tracks become ignore regions.
    role = role.strip().lower()
    if role == 'ball':
        return 0
    if role.startswith(('player', 'goalkeeper')):
        return 1
    if role == 'referee':
        return 2
    if role == 'other':
        return IGNORE
    raise ValueError(f'Unknown SoccerNet role {role!r}')


def read_clip(clip_dir):
    clip_dir = Path(clip_dir)
    parser = ConfigParser(interpolation=None)
    if not parser.read(clip_dir / 'gameinfo.ini', encoding='utf-8'):
        raise FileNotFoundError(f'Missing {clip_dir / "gameinfo.ini"}')
    info = parser['Sequence']
    tracks = {}
    for key, value in info.items():
        if key.startswith('trackletid_'):
            tracks[int(key.split('_', 1)[1])] = value.split(';', 1)[0].strip().lower()
    seq = ConfigParser(interpolation=None)
    seq.read(clip_dir / 'seqinfo.ini', encoding='utf-8')
    meta = seq['Sequence'] if seq.has_section('Sequence') else info
    rows = []
    with (clip_dir / 'gt' / 'gt.txt').open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split(',')
            if len(fields) < 7:
                raise ValueError(f'{clip_dir}/gt/gt.txt:{number}: expected >= 7 fields')
            frame, track = int(fields[0]), int(fields[1])
            if track not in tracks:
                raise ValueError(f'{clip_dir}/gt/gt.txt:{number}: track {track} has no role')
            x, y, w, h, conf = map(float, fields[2:7])
            rows.append(dict(frame=frame, track=track, role=tracks[track],
                             x=x, y=y, w=w, h=h, conf=conf))
    return dict(clip=clip_dir.name, game=info.get('gameid', '').strip(), tracks=tracks, rows=rows,
                image_dir=clip_dir / meta.get('imdir', 'img1').strip(),
                extension=meta.get('imext', '.jpg').strip(), length=int(meta['seqlength']),
                width=int(meta['imwidth']), height=int(meta['imheight']))


def clip_box(x, y, w, h, width, height):
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x+w, width), min(y+h, height)
    if x2 <= x1 or y2 <= y1:
        return None, True
    return (x1, y1, x2, y2), (x1, y1, x2, y2) != (x, y, x+w, y+h)


def yolo_line(class_id, xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    values = ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)
    return f'{class_id} ' + ' '.join(f'{v:.10f}' for v in values)


def clip_dirs(split_dir):
    dirs = sorted(p for p in Path(split_dir).iterdir() if (p / 'gameinfo.ini').is_file())
    if not dirs:
        raise FileNotFoundError(f'No clips with gameinfo.ini under {split_dir}')
    return dirs


def split_games(split_dir):
    return {path.name: read_clip(path)['game'] for path in clip_dirs(split_dir)}


def assert_disjoint_games(games_by_split):
    games = {split: set(values) for split, values in games_by_split.items()}
    splits = sorted(games)
    for i, left in enumerate(splits):
        for right in splits[i+1:]:
            shared = games[left] & games[right]
            if shared:
                raise ValueError(f'Splits {left} and {right} share games {sorted(shared)}')
    return {split: sorted(values, key=lambda g: (len(str(g)), str(g))) for split, values in games.items()}


def build_view(split_dir, output_dir, split, step):
    """Every `step`-th frame from frame 1; images are hardlinks, never edit them in place."""
    if step < 1:
        raise ValueError('Frame step must be positive')
    clips = [read_clip(path) for path in clip_dirs(split_dir)]
    for clip in clips:
        for role in clip['tracks'].values():
            role_class(role)
    output_dir = Path(output_dir)
    image_root, label_root = output_dir / 'images' / split, output_dir / 'labels' / split
    output_dir.mkdir(parents=True, exist_ok=False)
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    counts, roles, source_roles = Counter(), Counter(), Counter()
    frames, ignores, clipped, games = [], [], [], {}
    for clip in clips:
        games[clip['clip']] = clip['game']
        source_roles.update(clip['tracks'].values())
        by_frame = {}
        for row in clip['rows']:
            roles[row['role']] += 1
            counts['source_boxes'] += 1
            counts['source_conf_not_one'] += row['conf'] != 1
            by_frame.setdefault(row['frame'], []).append(row)
        if set(by_frame) - set(range(1, clip['length']+1)):
            raise ValueError(f'{clip["clip"]}: gt frames outside 1..{clip["length"]}')
        counts['source_frames'] += clip['length']
        counts['source_images'] += sum(1 for _ in clip['image_dir'].glob('*' + clip['extension']))
        for frame in range(1, clip['length']+1, step):
            source = clip['image_dir'] / f'{frame:06d}{clip["extension"]}'
            name = f'{clip["clip"]}_{frame:06d}'
            image = f'images/{split}/{name}{clip["extension"]}'
            with Image.open(source) as handle:
                if handle.size != (clip['width'], clip['height']):
                    raise ValueError(f'{source}: size {handle.size} differs from ini')
            os.link(source, output_dir / image)
            lines = []
            for row in by_frame.get(frame, []):
                class_id = role_class(row['role'])
                xyxy, was_clipped = clip_box(row['x'], row['y'], row['w'], row['h'],
                                             clip['width'], clip['height'])
                label = NAMES.get(class_id, IGNORE)
                if was_clipped:
                    counts[f'clipped_{label}' if xyxy else f'outside_{label}'] += 1
                    clipped.append(dict(image=image, track=row['track'], label=label,
                                        x=row['x'], y=row['y'], w=row['w'], h=row['h'],
                                        kept=xyxy is not None))
                if xyxy is None:
                    continue
                if class_id == IGNORE:
                    ignores.append(dict(image=image, x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]))
                else:
                    lines.append(yolo_line(class_id, xyxy, clip['width'], clip['height']))
                counts[f'boxes_{label}'] += 1
            (label_root / f'{name}.txt').write_text(''.join(f'{v}\n' for v in lines), encoding='utf-8')
            counts['frames'] += 1
            counts['empty_frames'] += not lines
            frames.append(dict(image=image, clip=clip['clip'], game=clip['game'], frame=frame))
    tables = {'frames': (['image', 'clip', 'game', 'frame'], frames),
              'ignore': (['image', 'x1', 'y1', 'x2', 'y2'], ignores),
              'clipped': (['image', 'track', 'label', 'x', 'y', 'w', 'h', 'kept'], clipped)}
    for name, (columns, rows) in tables.items():
        with (output_dir / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / 'data.yaml').write_text(
        f'path: .\ntest: images/{split}\nnc: {len(NAMES)}\nnames:\n'
        + ''.join(f'  {k}: {v}\n' for k, v in NAMES.items()), encoding='utf-8')
    manifest = dict(source=str(Path(split_dir).resolve()), split=split, step=step,
                    clips=len(games), games=sorted(set(games.values()), key=lambda g: (len(g), g)), clip_games=games,
                    names=NAMES, counts=dict(sorted(counts.items())),
                    source_boxes_by_role=dict(sorted(roles.items())),
                    source_tracks_by_role=dict(sorted(source_roles.items())),
                    files={f'{name}.csv': sha256(output_dir / f'{name}.csv') for name in tables},
                    code_sha256=sha256(Path(__file__)), images='hardlinks to source frames',
                    created_utc=datetime.now(timezone.utc).isoformat())
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
