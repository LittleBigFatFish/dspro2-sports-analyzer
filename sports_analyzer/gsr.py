"""SoccerNet GSR clips: audit, overlap keys and a YOLO view with pitch positions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os

from PIL import Image

from sports_analyzer.dataset_preparation import sha256
from sports_analyzer.soccernet import IGNORE, NAMES, clip_box, role_class, yolo_line

LABELS = 'Labels-GameState.json'
ROLES = ('player', 'goalkeeper', 'referee', 'ball', 'other')
TEAMS = ('left', 'right')
# Metres from the centre spot of a 105 x 68 pitch.
HALF_PITCH = (52.5, 34.0)
ERRORS = {'unknown_role', 'role_category_mismatch', 'missing_bbox', 'non_positive_size', 'outside_image',
          'unknown_image', 'missing_track', 'invalid_team', 'non_finite_pitch', 'duplicate_track_in_frame'}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def object_row(annotation, categories):
    attributes = annotation.get('attributes') or {}
    box, pitch = annotation.get('bbox_image') or {}, annotation.get('bbox_pitch') or {}
    return dict(id=str(annotation.get('id')), image_id=str(annotation.get('image_id')),
                track=annotation.get('track_id'), category=categories.get(annotation.get('category_id')),
                role=attributes.get('role'), team=attributes.get('team'), jersey=attributes.get('jersey'),
                x=box.get('x'), y=box.get('y'), w=box.get('w'), h=box.get('h'),
                x_center=box.get('x_center'), y_center=box.get('y_center'),
                pitch_x=pitch.get('x_bottom_middle'), pitch_y=pitch.get('y_bottom_middle'))


def read_clip(clip_dir):
    clip_dir = Path(clip_dir)
    data = json.loads((clip_dir / LABELS).read_text(encoding='utf-8'))
    info = data['info']
    categories = {c['id']: c['name'] for c in data['categories']}
    rows, pitch_images, other = [], set(), Counter()
    for annotation in data['annotations']:
        kind = annotation.get('supercategory')
        if kind == 'object':
            rows.append(object_row(annotation, categories))
        elif kind == 'pitch':
            pitch_images.add(str(annotation.get('image_id')))
        else:
            other[kind] += 1
    images = [dict(image_id=str(i['image_id']), file_name=i['file_name'], frame=int(Path(i['file_name']).stem),
                   width=i.get('width'), height=i.get('height'), is_labeled=i.get('is_labeled'),
                   has_labeled_person=i.get('has_labeled_person'),
                   has_labeled_pitch=i.get('has_labeled_pitch'), has_labeled_camera=i.get('has_labeled_camera'),
                   ignore_x=i.get('ignore_regions_x') or [], ignore_y=i.get('ignore_regions_y') or [])
              for i in data['images']]
    return dict(clip=clip_dir.name, info=info, game=str(info.get('game_id', '')), version=info.get('version'),
                action=info.get('action_class', ''), length=int(info.get('seq_length', len(images))),
                image_dir=clip_dir / info.get('im_dir', 'img1'), images=images, rows=rows,
                pitch_images=pitch_images, other_annotations=dict(other))


def clip_dirs(split_dir):
    dirs = sorted(p for p in Path(split_dir).iterdir() if (p / LABELS).is_file())
    if not dirs:
        raise FileNotFoundError(f'No clips with {LABELS} under {split_dir}')
    return dirs


def ignore_boxes(image):
    boxes = []
    for xs, ys in zip(image['ignore_x'], image['ignore_y']):
        if isinstance(xs, list) and isinstance(ys, list) and xs and all(map(number, xs + ys)):
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        else:
            boxes.append(None)
    return boxes


def row_issues(row, width, height):
    codes = []
    if row['role'] not in ROLES:
        codes.append('unknown_role')
    elif row['category'] != row['role']:
        codes.append('role_category_mismatch')
    if row['track'] is None:
        codes.append('missing_track')
    if not all(number(row[k]) for k in ('x', 'y', 'w', 'h')):
        codes.append('missing_bbox')
    elif row['w'] <= 0 or row['h'] <= 0:
        codes.append('non_positive_size')
    else:
        xyxy, clipped = clip_box(row['x'], row['y'], row['w'], row['h'], width, height)
        if xyxy is None:
            codes.append('outside_image')
        elif clipped:
            codes.append('partly_outside')
        if not (number(row['x_center']) and number(row['y_center'])
                and abs(row['x'] + row['w']/2 - row['x_center']) <= 1
                and abs(row['y'] + row['h']/2 - row['y_center']) <= 1):
            codes.append('centre_mismatch')
    if row['pitch_x'] is None or row['pitch_y'] is None:
        codes.append('missing_pitch')
    elif not (number(row['pitch_x']) and number(row['pitch_y'])):
        codes.append('non_finite_pitch')
    elif abs(row['pitch_x']) > HALF_PITCH[0] or abs(row['pitch_y']) > HALF_PITCH[1]:
        codes.append('off_pitch')
    if row['team'] is not None and row['team'] not in TEAMS:
        codes.append('invalid_team')
    elif row['role'] in ('player', 'goalkeeper') and row['team'] is None:
        codes.append('missing_team')
    elif row['role'] in ('referee', 'ball', 'other') and row['team'] is not None:
        codes.append('team_on_non_player')
    if row['jersey'] is not None and not str(row['jersey']).isdigit():
        codes.append('jersey_not_digit')
    return codes


def audit_clip(clip, check_files=True):
    counts, issues, examples = Counter(), Counter(), {}
    by_role = defaultdict(Counter)
    resolutions, track_values = Counter(), defaultdict(lambda: defaultdict(set))

    def issue(code, example):
        issues[code] += 1
        examples.setdefault(code, example)

    images = {}
    for image in clip['images']:
        if image['image_id'] in images:
            issue('duplicate_image_id', image['image_id'])
        images[image['image_id']] = image
        resolutions[f'{image["width"]}x{image["height"]}'] += 1
        for key in ('is_labeled', 'has_labeled_person', 'has_labeled_pitch', 'has_labeled_camera'):
            counts[f'images_{key}'] += image[key] is True
        counts['images_with_ignore_regions'] += bool(image['ignore_x'] or image['ignore_y'])
        counts['ignore_regions'] += len(image['ignore_x'])
        if len(image['ignore_x']) != len(image['ignore_y']) or None in ignore_boxes(image):
            issue('unparsed_ignore_region', image['file_name'])
        if check_files:
            path = clip['image_dir'] / image['file_name']
            if not path.is_file():
                issue('missing_image_file', image['file_name'])
                continue
            with Image.open(path) as handle:
                if handle.size != (image['width'], image['height']):
                    issue('image_size_mismatch', f'{image["file_name"]} {handle.size}')
    counts['images'] = len(images)
    counts['images_with_pitch_lines'] = len(clip['pitch_images'] & set(images))
    if len(images) != clip['length']:
        issue('length_mismatch', f'{len(images)} images, seq_length {clip["length"]}')
    if check_files:
        files = sum(1 for _ in clip['image_dir'].glob('*.jpg'))
        counts['image_files'] = files
        if files != len(images):
            issue('unlisted_image_files', f'{files} files, {len(images)} listed')
    seen = set()
    for row in clip['rows']:
        image = images.get(row['image_id'])
        if image is None:
            issue('unknown_image', row['id'])
            continue
        if (row['image_id'], row['track']) in seen:
            issue('duplicate_track_in_frame', row['id'])
        seen.add((row['image_id'], row['track']))
        role = row['role'] if row['role'] in ROLES else 'unknown'
        stats = by_role[role]
        stats['annotations'] += 1
        stats['pitch'] += number(row['pitch_x']) and number(row['pitch_y'])
        stats[f'team_{row["team"] if row["team"] in TEAMS else "none"}'] += 1
        stats['jersey'] += row['jersey'] is not None
        for code in row_issues(row, image['width'], image['height']):
            issue(code, row['id'])
            stats[f'issue_{code}'] += 1
        values = track_values[row['track']]
        for key in ('role', 'team', 'jersey'):
            if row[key] is not None:
                values[key].add(row[key])
    for track, values in track_values.items():
        role = next(iter(values['role'])) if len(values['role']) == 1 else 'mixed'
        by_role[role if role in ROLES else 'unknown']['tracks'] += 1
        for key in ('role', 'team', 'jersey'):
            if len(values[key]) > 1:
                issue(f'track_{key}_changes', f'track {track}: {sorted(map(str, values[key]))}')
    counts['tracks'] = len(track_values)
    return dict(counts=counts, by_role={k: dict(v) for k, v in by_role.items()},
                resolutions=resolutions, issues=issues, examples=examples)


def audit_split(split_dir, check_files=True):
    clips, issue_rows = [], []
    totals, resolutions, issues, versions, actions = Counter(), Counter(), Counter(), Counter(), Counter()
    by_role = defaultdict(Counter)
    for path in clip_dirs(split_dir):
        clip = read_clip(path)
        result = audit_clip(clip, check_files)
        totals.update(result['counts'])
        resolutions.update(result['resolutions'])
        issues.update(result['issues'])
        versions[str(clip['version'])] += 1
        actions[clip['action']] += 1
        for role, stats in result['by_role'].items():
            by_role[role].update(stats)
        info = clip['info']
        clips.append(dict(clip=clip['clip'], game=clip['game'], action=clip['action'],
                          game_time_start=info.get('game_time_start'), clip_start=info.get('clip_start'),
                          clip_stop=info.get('clip_stop'), action_position=info.get('action_position'),
                          version=clip['version'], images=result['counts']['images'],
                          annotations=len(clip['rows']), tracks=result['counts']['tracks'],
                          resolutions=';'.join(sorted(result['resolutions'])),
                          errors=sum(v for k, v in result['issues'].items() if k in ERRORS),
                          warnings=sum(v for k, v in result['issues'].items() if k not in ERRORS)))
        issue_rows += [dict(clip=clip['clip'], severity='error' if code in ERRORS else 'warning', code=code,
                            count=count, example=result['examples'][code])
                       for code, count in sorted(result['issues'].items())]
    roles = {}
    for role, stats in sorted(by_role.items()):
        n = stats['annotations']
        roles[role] = dict(stats, pitch_share=round(stats['pitch']/n, 6) if n else None,
                           jersey_share=round(stats['jersey']/n, 6) if n else None)
    games = sorted({c['game'] for c in clips}, key=lambda g: (len(g), g))
    summary = dict(clips=len(clips), games=games, frames=totals['images'],
                   annotations=sum(s['annotations'] for s in by_role.values()),
                   tracks=totals['tracks'], counts=dict(sorted(totals.items())), roles=roles,
                   resolutions=dict(resolutions), versions=dict(versions), action_classes=dict(actions.most_common()),
                   errors=sum(v for k, v in issues.items() if k in ERRORS),
                   warnings=sum(v for k, v in issues.items() if k not in ERRORS),
                   issues=dict(sorted(issues.items())))
    return dict(summary=summary, clips=clips, issues=issue_rows)


def clip_key(info):
    """Same broadcast moment: start time, clip window and annotated action."""
    return tuple(str(info.get(k, '')).strip() for k in ('game_time_start', 'clip_start', 'clip_stop',
                                                         'action_position', 'action_class'))


def match_clips(gsr, tracking):
    index = defaultdict(list)
    for clip, key in tracking.items():
        index[key].append(clip)
    return {clip: sorted(index.get(key, [])) for clip, key in sorted(gsr.items())}


def build_view(split_dir, output_dir, split, step):
    """Every `step`-th frame with person labels; images are hardlinks."""
    if step < 1:
        raise ValueError('Frame step must be positive')
    clips = [read_clip(path) for path in clip_dirs(split_dir)]
    for clip in clips:
        for row in clip['rows']:
            role_class(row['role'] or '')
    output_dir = Path(output_dir)
    image_root, label_root = output_dir / 'images' / split, output_dir / 'labels' / split
    output_dir.mkdir(parents=True, exist_ok=False)
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    counts, roles = Counter(), Counter()
    frames, ignores, clipped, tracks, games = [], [], [], [], {}
    for clip in clips:
        games[clip['clip']] = clip['game']
        by_image = defaultdict(list)
        for row in clip['rows']:
            by_image[row['image_id']].append(row)
            roles[row['role']] += 1
        counts['source_boxes'] += len(clip['rows'])
        counts['source_frames'] += len(clip['images'])
        for image in sorted(clip['images'], key=lambda i: i['frame']):
            if (image['frame'] - 1) % step:
                continue
            if not (image['is_labeled'] and image['has_labeled_person']):
                counts['skipped_unlabeled_frames'] += 1
                continue
            width, height = image['width'], image['height']
            source = clip['image_dir'] / image['file_name']
            name = f'{clip["clip"]}_{image["frame"]:06d}'
            relative = f'images/{split}/{name}{source.suffix}'
            with Image.open(source) as handle:
                if handle.size != (width, height):
                    raise ValueError(f'{source}: size {handle.size} differs from {LABELS}')
            os.link(source, output_dir / relative)
            for region in ignore_boxes(image):
                if region is None:
                    raise ValueError(f'{source}: unreadable ignore region')
                ignores.append(dict(image=relative, x1=region[0], y1=region[1], x2=region[2], y2=region[3],
                                    source='ignore_region'))
                counts['ignore_regions'] += 1
            lines = []
            for row in by_image.get(image['image_id'], []):
                class_id = role_class(row['role'])
                label = NAMES.get(class_id, IGNORE)
                xyxy, was_clipped = clip_box(row['x'], row['y'], row['w'], row['h'], width, height)
                if was_clipped:
                    counts[f'clipped_{label}' if xyxy else f'outside_{label}'] += 1
                    clipped.append(dict(image=relative, track=row['track'], label=label, x=row['x'],
                                        y=row['y'], w=row['w'], h=row['h'], kept=xyxy is not None))
                if xyxy is None:
                    continue
                if class_id == IGNORE:
                    ignores.append(dict(image=relative, x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                                        source='other'))
                else:
                    lines.append(yolo_line(class_id, xyxy, width, height))
                counts[f'boxes_{label}'] += 1
                tracks.append(dict(image=relative, clip=clip['clip'], frame=image['frame'],
                                   annotation=row['id'], x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                                   label=label, track=row['track'], role=row['role'],
                                   team=row['team'] or 'none', jersey=row['jersey'] or '',
                                   pitch_x=row['pitch_x'], pitch_y=row['pitch_y']))
            (label_root / f'{name}.txt').write_text(''.join(f'{v}\n' for v in lines), encoding='utf-8')
            counts['frames'] += 1
            counts['empty_frames'] += not lines
            frames.append(dict(image=relative, clip=clip['clip'], game=clip['game'], frame=image['frame'],
                               image_id=image['image_id']))
    tables = {'frames': (['image', 'clip', 'game', 'frame', 'image_id'], frames),
              'ignore': (['image', 'x1', 'y1', 'x2', 'y2', 'source'], ignores),
              'clipped': (['image', 'track', 'label', 'x', 'y', 'w', 'h', 'kept'], clipped),
              'tracks': (['image', 'clip', 'frame', 'annotation', 'x1', 'y1', 'x2', 'y2', 'label', 'track',
                          'role', 'team', 'jersey', 'pitch_x', 'pitch_y'], tracks)}
    for name, (columns, rows) in tables.items():
        with (output_dir / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    key = 'test' if split == 'test' else 'val'
    (output_dir / 'data.yaml').write_text(
        f'path: .\n{key}: images/{split}\nnc: {len(NAMES)}\nnames:\n'
        + ''.join(f'  {k}: {v}\n' for k, v in NAMES.items()), encoding='utf-8')
    manifest = dict(source=str(Path(split_dir).resolve()), split=split, step=step, clips=len(games),
                    games=sorted(set(games.values()), key=lambda g: (len(g), g)), clip_games=games,
                    names=NAMES, counts=dict(sorted(counts.items())),
                    source_boxes_by_role=dict(sorted(roles.items(), key=lambda r: str(r[0]))),
                    files={f'{name}.csv': sha256(output_dir / f'{name}.csv') for name in tables},
                    code_sha256=sha256(Path(__file__)), images='hardlinks to source frames',
                    created_utc=datetime.now(timezone.utc).isoformat())
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
