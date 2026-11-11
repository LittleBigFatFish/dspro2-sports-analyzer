"""Named YOLO training sets from Football, Soccana and SoccerNet; raw files are only read."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import statistics
import warnings

from PIL import Image

from sports_analyzer.annotations.yolo import load_names, read_labels
from sports_analyzer.comparison import check_against_audit, filter_by_audit, read_exclusions, yolo_view
from sports_analyzer.dataset_preparation import duplicate_kind, hamming_matches, image_hashes, sha256
from sports_analyzer.soccernet import NAMES, clip_box, clip_dirs, read_clip, role_parts, yolo_line

IDS = {name: k for k, name in NAMES.items()}
SPLITS = ('train', 'val')
HASHES = ('image_sha256', 'pixel_sha256', 'dhash', 'mirror_pixel_sha256', 'mirror_dhash')
WIDE = 16 / 9
COLUMNS = ['image', 'split', 'dataset', 'source_file', 'group', 'game', 'licence', 'source_sha256',
           'image_sha256', 'width', 'height', 'image_transform', 'label_transform', 'boxes']
DROPPED = ['dataset', 'split', 'source_file', 'reason']


def class_map(classes, names=None) -> dict:
    """Source class (id or role) -> target id; source data.yaml names must agree."""
    mapping = {}
    for source, target in classes.items():
        if target not in IDS:
            raise ValueError(f'Class {source!r} maps to unknown target {target!r}')
        mapping[str(source)] = IDS[target]
    if names is not None:
        unmapped = sorted({str(k) for k in names} - set(mapping))
        if unmapped:
            raise ValueError(f'Unmapped source classes {unmapped}')
        wrong = {k: v for k, v in names.items() if NAMES[mapping[str(k)]] != v.lower()}
        if wrong:
            raise ValueError(f'Config mapping disagrees with data.yaml names {wrong}')
    return mapping


def remap_lines(text, mapping, where) -> list[str]:
    # Coordinates are kept as written, only the class id changes.
    lines = []
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f'{where}:{number}: expected 5 fields, got {len(fields)}')
        if fields[0] not in mapping:
            raise ValueError(f'{where}:{number}: unmapped class {fields[0]}')
        lines.append(' '.join([str(mapping[fields[0]]), *fields[1:]]))
    return lines


def refuse_report(folder, report_folders):
    folder = Path(folder).resolve()
    for report in report_folders:
        if folder == report or folder.is_relative_to(report) or report.is_relative_to(folder):
            raise ValueError(f'{folder} is or holds report set {report}')


def read_audit(audit_dir) -> dict:
    with (Path(audit_dir) / 'images.csv').open(encoding='utf-8', newline='') as stream:
        return {row['image']: row for row in csv.DictReader(stream)}


def report_sets(config, base=Path('.')) -> dict:
    rows, folders = {}, [(base / f).resolve() for f in config.get('report_folders', [])]
    for name, settings in config['report_sets'].items():
        audit = base / settings['audit_run']
        root = Path(json.loads((audit / 'summary.json').read_text(encoding='utf-8'))['root'])
        folders.append((root / 'images' / settings['split']).resolve())
        rows[name] = [r for r in read_audit(audit).values() if r['split'] == settings['split'] and r['dhash']]
        if not rows[name]:
            raise ValueError(f'Report set {name} has no hashed images in {audit}')
    return dict(rows=rows, folders=folders, games={str(g) for g in config.get('report_games', [])})


def target_size(width, height, mode=None):
    # Square frames were squashed from 16:9; restore keeps the height and widens.
    if mode is None or abs(width / height - WIDE) < 0.01:
        return width, height
    if mode not in ('restore', 'drop') or width != height:
        raise ValueError(f'Unexpected frame size {width}x{height} or mode {mode!r}')
    return (round(height * WIDE), height) if mode == 'restore' else None


def rotated(lines, width, height, threshold) -> bool:
    """Median player or referee box width/height, as an upright 16:9 frame would show it, above threshold."""
    ratios = [float(f[3])*width / (float(f[4])*height) for f in map(str.split, lines) if int(f[0]) != IDS['ball']]
    return bool(ratios) and statistics.median(ratios) * WIDE * height / width > threshold


def yolo_source(name, settings, base, report):
    root = base / settings['root']
    split = settings['split']
    refuse_report(root / 'images' / split, report['folders'])
    mapping = class_map(settings['classes'], load_names(root / settings.get('data_yaml', 'data.yaml')))
    audit = base / settings['audit_run']
    extra = set().union(*(read_exclusions(base / p) for p in settings.get('extra_exclusions', [])))
    groups, val_groups = set(settings.get('exclude_groups', [])), set(settings.get('val_groups', []))
    full, excluded = settings.get('full_frames_only', False), settings.get('apply_exclusions', False)
    steps = [('crop', lambda v: filter_by_audit(v, audit, full)),
             ('audit_exclusion', lambda v: filter_by_audit(v, audit, apply_exclusions=excluded)),
             ('overlap_exclusion', lambda v: [r for r in v if r['image'] not in extra]),
             ('excluded_group', lambda v: [r for r in v if r['source'] not in groups])]
    view, dropped = yolo_view(root, [split]), []
    missing = sorted(val_groups - {r['source'] for r in view})
    if missing:
        raise ValueError(f'{name}: unknown val groups {missing}')
    for r in view:
        r['split'] = 'val' if r['source'] in val_groups else 'train'
    for reason, step in steps:
        kept = step(view)
        names = {r['image'] for r in kept}
        dropped += [dict(dataset=name, split=r['split'], source_file=r['image'], reason=reason)
                    for r in view if r['image'] not in names]
        view = kept
    for r in view:
        r['image_sha256'] = sha256(root / r['image'])
    check_against_audit(view, audit)
    audited, records = read_audit(audit), []
    identity = all(str(v) == k for k, v in mapping.items())
    for r in view:
        row = audited[r['image']]
        width, height = int(row['width']), int(row['height'])
        label = root / row['label']
        lines = remap_lines(label.read_text(encoding='utf-8-sig'), mapping, label)
        # Decided on the source size, before any resize.
        if settings.get('drop_rotated') and rotated(lines, width, height, settings['rotated_width_height']):
            dropped.append(dict(dataset=name, split=r['split'], source_file=r['image'], reason='rotated'))
            continue
        size = target_size(width, height, settings.get('squashed_frames'))
        if size is None:
            dropped.append(dict(dataset=name, split=r['split'], source_file=r['image'], reason='squashed'))
            continue
        records.append(dict(dataset=name, split=r['split'], path=root / r['image'], source_file=r['image'],
                            name=Path(r['image']).stem, group=r['source'], game='', width=width, height=height,
                            size=size, outside=0, licence=settings['licence'], lines=lines,
                            label_transform='copied' if identity else 'class ids remapped',
                            **{k: row[k] for k in HASHES}))
    return records, dropped


def soccernet_source(name, settings, base, report):
    root = base / settings['root']
    refuse_report(root, report['folders'])
    mapping, drop_roles = class_map(settings['classes']), set(settings.get('drop_roles', []))
    splits = {str(g): ('train', settings['train_step']) for g in settings['train_games']}
    splits.update({str(g): ('val', settings['val_step']) for g in settings['val_games']})
    if len(splits) != len(settings['train_games']) + len(settings['val_games']):
        raise ValueError(f'{name}: a game is in both train and val')
    if set(splits) & report['games']:
        raise ValueError(f'{name}: report games {sorted(set(splits) & report["games"])} requested')
    records, dropped = [], []
    for path in clip_dirs(root):
        clip = read_clip(path)
        if clip['game'] in report['games'] or clip['game'] not in splits:
            raise ValueError(f'{clip["clip"]}: game {clip["game"]} is a report game or has no split')
        unmapped = {r for r in clip['tracks'].values() if role_parts(r)[0] not in mapping.keys() | drop_roles}
        if unmapped:
            raise ValueError(f'{clip["clip"]}: unmapped roles {sorted(unmapped)}')
        split, step = splits[clip['game']]
        by_frame = defaultdict(list)
        for row in clip['rows']:
            by_frame[row['frame']].append(row)
        for frame in range(1, clip['length']+1, step):
            image = clip['image_dir'] / f'{frame:06d}{clip["extension"]}'
            source_file = image.relative_to(root).as_posix()
            rows = by_frame.get(frame, [])
            if any(role_parts(r['role'])[0] in drop_roles for r in rows):
                dropped.append(dict(dataset=name, split=split, source_file=source_file, reason='other_role'))
                continue
            lines, outside = [], 0
            for row in rows:
                xyxy, _ = clip_box(row['x'], row['y'], row['w'], row['h'], clip['width'], clip['height'])
                if xyxy is None:
                    outside += 1
                    continue
                lines.append(yolo_line(mapping[role_parts(row['role'])[0]], xyxy, clip['width'], clip['height']))
            records.append(dict(dataset=name, split=split, path=image, source_file=source_file,
                                name=f'{clip["clip"]}_{frame:06d}', group=clip['clip'], game=clip['game'],
                                width=clip['width'], height=clip['height'], size=(clip['width'], clip['height']),
                                licence=settings['licence'], lines=lines, outside=outside,
                                label_transform='gt.txt to YOLO, boxes clipped to the image'))
    return records, dropped


def hash_image(path):
    with Image.open(path) as image:
        image.load()
        return dict(image_sha256=sha256(path), size=image.size, **image_hashes(image))


def add_hashes(records, workers=1):
    todo = [r for r in records if not r.get('dhash')]
    paths = [r['path'] for r in todo]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            hashes = list(pool.map(hash_image, paths, chunksize=32))
    else:
        hashes = [hash_image(p) for p in paths]
    for record, found in zip(todo, hashes):
        if tuple(found.pop('size')) != (record['width'], record['height']):
            raise ValueError(f'{record["path"]}: size differs from its annotation')
        record.update(found)


def near_duplicates(queries, rows, distance):
    """(i, j, kind, distance, mirror distance) for query i near row j, plain or mirrored."""
    plain, mirror = [int(r['dhash'], 16) for r in rows], [int(r['mirror_dhash'], 16) for r in rows]
    pairs = set()
    for column in ('dhash', 'mirror_dhash'):
        pairs |= {(i, j) for i, j, _ in hamming_matches([int(q[column], 16) for q in queries], plain, distance)}
    pairs |= {(i, j) for i, j, _ in hamming_matches([int(q['dhash'], 16) for q in queries], mirror, distance)}
    by_pixels = defaultdict(list)
    for j, row in enumerate(rows):
        by_pixels[row['pixel_sha256']].append(j)
    for i, query in enumerate(queries):
        pairs |= {(i, j) for j in by_pixels.get(query['mirror_pixel_sha256'], ())}
    for i, j in sorted(pairs):
        kind, plain_distance, mirror_distance = duplicate_kind(queries[i], rows[j], distance)
        if kind:
            yield i, j, kind, plain_distance, mirror_distance


def drop_leaks(records, reports, distance):
    """Drops any image near a report set, then training images near a validation image."""
    matches, reasons = [], defaultdict(set)

    def check(name, queries, candidates, reference):
        for i, j, kind, plain, mirrored in near_duplicates(queries, candidates, distance):
            found = candidates[j]
            matches.append(dict(reference=name, reference_image=reference(queries[i]), dataset=found['dataset'],
                                source_file=found['source_file'], split=found['split'], kind=kind,
                                dhash_distance=plain, mirror_distance=mirrored))
            reasons[found['dataset'], found['split'], found['source_file']].add(f'near_{name}')

    for name, rows in reports.items():
        check(name, rows, records, lambda q: q['image'])
    records = [r for r in records if (r['dataset'], r['split'], r['source_file']) not in reasons]
    check('validation', [r for r in records if r['split'] == 'val'], [r for r in records if r['split'] == 'train'],
          lambda q: f'{q["dataset"]}/{q["source_file"]}')
    dropped = [dict(zip(DROPPED, (*key, ';'.join(sorted(v))))) for key, v in sorted(reasons.items())]
    return [r for r in records if (r['dataset'], r['split'], r['source_file']) not in reasons], dropped, matches


def prepare(config, base=Path('.'), workers=1) -> dict:
    report = report_sets(config, base)
    builders = {'yolo': yolo_source, 'soccernet': soccernet_source}
    records, dropped = [], []
    for name, settings in config['sources'].items():
        if settings.get('kind') not in builders:
            raise ValueError(f'Unknown source kind {settings.get("kind")!r} for {name}')
        found, lost = builders[settings['kind']](name, settings, base, report)
        records += found
        dropped += lost
    add_hashes(records, workers)
    records, leaks, matches = drop_leaks(records, report['rows'], config['near_duplicate_distance'])
    forbidden = {r['image_sha256'] for rows in report['rows'].values() for r in rows}
    return dict(records=records, dropped=dropped + leaks, matches=matches, report_sha256=forbidden)


def ball_ratios(lines, before, after):
    # Pixel width over height of each ball box at the source and the derived size.
    for line in lines:
        fields = line.split()
        if int(fields[0]) == IDS['ball']:
            w, h = float(fields[3]), float(fields[4])
            yield w*before[0] / (h*before[1]), w*after[0] / (h*after[1])


def median(values):
    return round(statistics.median(values), 4) if values else None


def tally(values) -> dict:
    return dict(sorted(Counter(values).items()))


def write_csv(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_set(name, config, prepared, output_dir, info=None) -> dict:
    """Train images of the set's sources plus the shared validation images; hardlinks unless resized."""
    sources = config['sets'][name]
    unknown = sorted(set(sources) - set(config['sources']))
    if unknown:
        raise ValueError(f'Set {name}: unknown sources {unknown}')
    records = [r for r in prepared['records'] if r['split'] == 'val' or r['dataset'] in sources]
    if {r['split'] for r in records} != set(SPLITS):
        raise ValueError(f'Set {name}: train or val is empty')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for kind in ('images', 'labels'):
        for split in SPLITS:
            (output_dir / kind / split).mkdir(parents=True)
    rows, boxes, ratios = [], Counter(), defaultdict(list)
    for r in sorted(records, key=lambda r: (r['split'], r['dataset'], r['source_file'])):
        stem = f'{r["dataset"]}_{r["name"]}'
        image = f'images/{r["split"]}/{stem}{r["path"].suffix}'
        if r['size'] == (r['width'], r['height']):
            os.link(r['path'], output_dir / image)
            transform, digest = 'hardlink', r['image_sha256']
        else:
            with Image.open(r['path']) as handle:
                handle.convert('RGB').resize(r['size'], Image.Resampling.LANCZOS).save(
                    output_dir / image, quality=config['jpeg_quality'])
            transform = f'resized {r["width"]}x{r["height"]} to {r["size"][0]}x{r["size"][1]}'
            digest = sha256(output_dir / image)
        (output_dir / 'labels' / r['split'] / f'{stem}.txt').write_text(''.join(f'{v}\n' for v in r['lines']),
                                                                       encoding='utf-8')
        boxes.update((r['split'], r['dataset'], NAMES[int(line.split()[0])]) for line in r['lines'])
        kind = 'unchanged' if transform == 'hardlink' else 'resized'
        ratios[f'{r["dataset"]}/{kind}'] += ball_ratios(r['lines'], (r['width'], r['height']), r['size'])
        rows.append(dict(image=image, split=r['split'], dataset=r['dataset'], source_file=r['source_file'],
                         group=r['group'], game=r['game'], licence=r['licence'], source_sha256=r['image_sha256'],
                         image_sha256=digest, width=r['size'][0], height=r['size'][1], image_transform=transform,
                         label_transform=r['label_transform'], boxes=len(r['lines'])))
    datasets = {r['dataset'] for r in records}
    dropped = [d for d in prepared['dropped'] if d['split'] == 'val' or d['dataset'] in sources]
    matches = [m for m in prepared['matches'] if m['split'] == 'val' or m['dataset'] in sources]
    write_csv(output_dir / 'images.csv', rows, COLUMNS)
    write_csv(output_dir / 'dropped.csv', dropped, DROPPED)
    (output_dir / 'data.yaml').write_text(
        '# No path key, so Ultralytics resolves the folders next to this file.\n'
        f'train: images/train\nval: images/val\nnc: {len(NAMES)}\nnames:\n'
        + ''.join(f'  {k}: {v}\n' for k, v in NAMES.items()), encoding='utf-8')
    for key in list(ratios):
        ratios[f'{key.split("/")[0]}/all'] += ratios[key]
    by_class = Counter()
    for (split, _, label), n in boxes.items():
        by_class[f'{split}/{label}'] += n
    manifest = dict(name=name, sources=sources, names=NAMES, images=tally(r['split'] for r in rows),
                    images_by_source=tally(f'{r["split"]}/{r["dataset"]}' for r in rows),
                    boxes_by_class=dict(sorted(by_class.items())),
                    boxes_by_source={'/'.join(k): n for k, n in sorted(boxes.items())},
                    dropped=tally(f'{d["split"]}/{d["dataset"]}/{d["reason"]}' for d in dropped),
                    boxes_outside_image=tally(r['dataset'] for r in records for _ in range(r['outside'])),
                    near_duplicates=tally(f'{m["reference"]}/{m["split"]}/{m["dataset"]}/{m["kind"]}' for m in matches),
                    ball_width_height={k: dict(boxes=len(v), median_before=median([b for b, _ in v]),
                                               median_after=median([a for _, a in v]))
                                       for k, v in sorted(ratios.items())},
                    credits={d: dict(licence=s['licence'], credit=s['credit'])
                             for d, s in config['sources'].items() if d in datasets},
                    config=config, files={f: sha256(output_dir / f) for f in ('images.csv', 'dropped.csv', 'data.yaml')},
                    code_sha256=sha256(Path(__file__)),
                    images_note='hardlinks to source files; resized frames are new JPEG files',
                    **(info or {}), created_utc=datetime.now(timezone.utc).isoformat())
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def verify_set(output_dir, report_sha256=()) -> dict:
    """Recounts the written files; raises on a label, split, count or report-set problem."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / 'manifest.json').read_text(encoding='utf-8'))
    with (output_dir / 'images.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    images, boxes, stems, problems = Counter(), Counter(), {}, []
    for split in SPLITS:
        paths = sorted((output_dir / 'images' / split).iterdir())
        stems[split] = {p.stem for p in paths}
        if stems[split] != {p.stem for p in (output_dir / 'labels' / split).glob('*.txt')}:
            problems.append(f'{split}: images and labels differ')
            continue
        for path in paths:
            with Image.open(path) as handle:
                size = handle.size
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                found = read_labels(output_dir / 'labels' / split / f'{path.stem}.txt', *size, NAMES)
            boxes.update(f'{split}/{b.label}' for b in found)
            images[split] += 1
    on_disk = {f'images/{s}/{p.name}' for s in SPLITS for p in (output_dir / 'images' / s).iterdir()}
    if {r['image'] for r in rows} != on_disk:
        problems.append('images.csv differs from the image folders')
    if stems['train'] & stems['val']:
        problems.append('file names in train and val')
    for key in (('dataset', 'source_file'), ('source_sha256',), ('image_sha256',)):
        found = [{tuple(r[k] for k in key) for r in rows if r['split'] == split} for split in SPLITS]
        if found[0] & found[1]:
            problems.append(f'{"/".join(key)} in train and val')
    leaked = {r[k] for r in rows for k in ('source_sha256', 'image_sha256')} & set(report_sha256)
    if leaked:
        problems.append(f'{len(leaked)} report-set images')
    if dict(sorted(images.items())) != manifest['images']:
        problems.append('image counts differ from the manifest')
    if dict(sorted(boxes.items())) != manifest['boxes_by_class']:
        problems.append('box counts differ from the manifest')
    if problems:
        raise ValueError(f'{output_dir.name}: ' + '; '.join(problems))
    return dict(images=dict(sorted(images.items())), boxes=dict(sorted(boxes.items())))


