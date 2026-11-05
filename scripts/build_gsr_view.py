"""SoccerNet GSR: extract, audit, check overlap with Tracking and build a YOLO view."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import platform
import sys
import time
import zipfile

from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_soccernet_view import save_overlays
from sports_analyzer.dataset_preparation import extract_zip, git_state, hamming_matches, image_hashes, sha256
from sports_analyzer.gsr import audit_split, build_view, clip_dirs, clip_key, match_clips, read_clip

REPOSITORY = 'https://huggingface.co/datasets/SoccerNet/SN-GSR-2025'
REVISION = 'ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9'
SOURCE = dict(
    name='SoccerNet Game State Reconstruction 2025 (SN-GSR-2025)', repository=REPOSITORY, revision=REVISION,
    licence='Hugging Face card tag gpl-3.0 (no licence text in the archives); frames are third-party broadcast '
            'footage, local coursework use only, no redistribution, cite the paper.',
    citation='Somers et al., "SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and '
             'Identification on a Minimap", CVPR Workshops 2024 (arXiv 2404.11335)',
    extraction='Python zipfile into a new empty folder per split; no absolute, drive, backslash or parent members; '
               'no case collisions', splits={})
TRACKING_KEYS = ('gametimestart', 'clipstart', 'clipstop', 'actionposition', 'actionclass')


def utc_folder(root):
    target = Path(root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_info(script_args):
    return dict(arguments={k: str(v) for k, v in vars(script_args).items()}, python=platform.python_version(),
                pillow=pillow_version, device='cpu', platform=platform.platform(),
                script_sha256=sha256(Path(__file__)), gsr_code_sha256=sha256(ROOT / 'sports_analyzer/gsr.py'),
                git=git_state(), created_utc=datetime.now(timezone.utc).isoformat())


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def pairs(specs):
    return {name: Path(folder).resolve() for name, folder in (s.split('=', 1) for s in specs)}


def extract(args):
    target = args.raw / args.split
    with zipfile.ZipFile(args.archive) as source:
        names = [m.filename for m in source.infolist()]
    folded = Counter(n.lower() for n in names)
    if len(folded) != len(names):
        raise ValueError(f'Case collisions: {[n for n, c in folded.items() if c > 1][:5]}')
    digest = sha256(args.archive)
    if args.sha256 and digest != args.sha256:
        raise ValueError(f'{args.archive}: sha256 {digest}, expected {args.sha256}')
    extract_zip(args.archive, target)
    files = [p for p in target.rglob('*') if p.is_file()]
    record = json.loads(args.source_json.read_text(encoding='utf-8')) if args.source_json.is_file() else SOURCE
    record['splits'][args.split] = dict(
        url=f'{REPOSITORY}/resolve/{REVISION}/{args.archive_name}', file=args.archive_name,
        local_archive=args.archive.resolve().relative_to(ROOT).as_posix(), bytes=args.archive.stat().st_size,
        sha256=digest, downloaded_utc=datetime.fromtimestamp(args.archive.stat().st_mtime, timezone.utc)
        .strftime('%Y-%m-%dT%H:%M:%SZ'), extracted_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        extracted_to=target.resolve().relative_to(ROOT).as_posix(), extracted_files=len(files),
        extracted_bytes=sum(p.stat().st_size for p in files), clips=len(clip_dirs(target)))
    args.source_json.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record['splits'][args.split], indent=2))


def audit(args):
    start = time.perf_counter()
    target = utc_folder(args.output)
    summaries, clips, issues = {}, [], []
    for split, folder in pairs(args.split).items():
        result = audit_split(folder)
        summaries[split] = dict(result['summary'], folder=str(folder))
        clips += [dict(split=split, **row) for row in result['clips']]
        issues += [dict(split=split, **row) for row in result['issues']]
        print(split, json.dumps({k: result['summary'][k] for k in ('clips', 'frames', 'annotations', 'errors',
                                                                     'warnings')}), flush=True)
    write_csv(target / 'clips.csv', clips, list(clips[0]))
    write_csv(target / 'issues.csv', issues, ['split', 'clip', 'severity', 'code', 'count', 'example'])
    (target / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    manifest = dict(run_info(args), seconds=round(time.perf_counter() - start, 1),
                    source_json_sha256=sha256(args.source_json) if args.source_json.is_file() else None,
                    files={n: sha256(target / n) for n in ('clips.csv', 'issues.csv', 'summary.json')})
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Report:', target)


def hash_frame(job):
    path, full = job
    row = dict(image_sha256=sha256(path))
    if full:
        with Image.open(path) as image:
            row.update(image_hashes(image))
    return row


def frame_table(name, split_dir, frames_of, sample_step, workers):
    """Pixel SHA-256 and dHash only for every `sample_step`-th frame."""
    rows = []
    for clip, image_dir in frames_of(split_dir):
        for path in sorted(image_dir.glob('*.jpg')):
            frame = int(path.stem)
            rows.append(dict(split=name, clip=clip, frame=frame, path=path,
                             image=path.relative_to(split_dir).as_posix(), sampled=(frame - 1) % sample_step == 0))
    with ProcessPoolExecutor(workers) as pool:
        hashes = list(pool.map(hash_frame, [(r['path'], r['sampled']) for r in rows], chunksize=64))
    for row, values in zip(rows, hashes):
        row.update(values)
    return rows


def gsr_frames(split_dir):
    return [(path.name, path / 'img1') for path in clip_dirs(split_dir)]


def tracking_clips(split_dir):
    clips = {}
    for path in sorted(p for p in Path(split_dir).iterdir() if (p / 'gameinfo.ini').is_file()):
        parser = ConfigParser(interpolation=None)
        parser.read(path / 'gameinfo.ini', encoding='utf-8')
        info = parser['Sequence']
        clips[path.name] = dict(game=info.get('gameid', '').strip(), image_dir=path / 'img1',
                                key=tuple(info.get(k, '').strip() for k in TRACKING_KEYS))
    return clips


def overlap(args):
    start = time.perf_counter()
    gsr_splits, tracking_splits = pairs(args.gsr), pairs(args.tracking)
    tracking, tracking_rows = {}, []
    for name, folder in tracking_splits.items():
        clips = tracking_clips(folder)
        tracking.update({(name, clip): values for clip, values in clips.items()})
        tracking_rows += frame_table(name, folder, lambda _: [(c, v['image_dir']) for c, v in clips.items()],
                                     args.sample_step, args.workers)
        print(f'tracking {name}: {len(clips)} clips, {len(tracking_rows)} frames hashed so far', flush=True)
    by_file, by_pixel, by_frame = defaultdict(list), defaultdict(list), {}
    for j, row in enumerate(tracking_rows):
        by_file[row['image_sha256']].append(j)
        by_frame[row['split'], row['clip'], row['frame']] = j
        if row['sampled']:
            by_pixel[row['pixel_sha256']].append(j)
    sampled = [j for j, r in enumerate(tracking_rows) if r['sampled']]
    clip_rows, frame_rows, summary = [], [], {}
    for name, folder in gsr_splits.items():
        clips = {path.name: read_clip(path) for path in clip_dirs(folder)}
        keys = {clip: clip_key(c['info']) for clip, c in clips.items()}
        matched = match_clips(keys, {f'{s}/{c}': v['key'] for (s, c), v in tracking.items()})
        rows = frame_table(name, folder, gsr_frames, args.sample_step, args.workers)
        stats = defaultdict(Counter)
        visual = defaultdict(list)
        queries = [i for i, r in enumerate(rows) if r['sampled']]
        for qi, tj, distance in hamming_matches([int(rows[i]['dhash'], 16) for i in queries],
                                                [int(tracking_rows[j]['dhash'], 16) for j in sampled],
                                                args.distance):
            visual[queries[qi]].append((sampled[tj], distance))
        for i, row in enumerate(rows):
            clip_stats = stats[row['clip']]
            clip_stats['frames'] += 1
            hits = [(j, 'byte_identical') for j in by_file.get(row['image_sha256'], [])]
            if row['sampled']:
                clip_stats['sampled'] += 1
                hits += [(j, 'pixel_identical') for j in by_pixel.get(row['pixel_sha256'], [])]
                hits += [(j, f'visual_candidate:{d}') for j, d in visual.get(i, [])]
            best = {}
            for j, kind in hits:
                best.setdefault(j, kind)
            for j, kind in best.items():
                other = tracking_rows[j]
                kind, _, distance = kind.partition(':')
                frame_rows.append(dict(gsr_split=name, gsr_image=row['image'], tracking_split=other['split'],
                                       tracking_image=other['image'], kind=kind, dhash_distance=distance,
                                       same_frame=(other['frame'] == row['frame'])))
            kinds = {k.partition(':')[0] for _, k in hits}
            clip_stats['byte_identical'] += 'byte_identical' in kinds
            clip_stats['sample_pixel_identical'] += row['sampled'] and 'pixel_identical' in kinds
            clip_stats['sample_visual_candidate'] += row['sampled'] and 'visual_candidate' in kinds
            clip_stats['exact_match'] += bool(kinds - {'visual_candidate'})
            for target in matched[row['clip']]:
                split, clip = target.split('/')
                j = by_frame.get((split, clip, row['frame']))
                clip_stats['mapped_frames_same_bytes'] += j is not None and \
                    tracking_rows[j]['image_sha256'] == row['image_sha256']
        for clip, c in clips.items():
            s, targets = stats[clip], matched[clip]
            number = clip.split('-')[-1]
            same_number = sorted(f'{t[0]}/{t[1]}' for t in tracking if t[1].split('-')[-1] == number)
            # dHash hits alone are only candidates, broadcast views look alike.
            verdict = ('identical' if s['frames'] and s['byte_identical'] == s['frames'] else
                       'partial' if s['exact_match'] else 'metadata_only' if targets else
                       'visual_candidate' if s['sample_visual_candidate'] else 'none')
            clip_rows.append(dict(gsr_split=name, gsr_clip=clip, gsr_game=c['game'], frames=s['frames'],
                                  metadata_match=';'.join(targets), tracking_game=';'.join(
                                      tracking[tuple(t.split('/'))]['game'] for t in targets),
                                  same_number_clip=';'.join(same_number),
                                  mapped_frames_same_bytes=s['mapped_frames_same_bytes'],
                                  byte_identical_frames=s['byte_identical'], sampled_frames=s['sampled'],
                                  sample_pixel_identical=s['sample_pixel_identical'],
                                  sample_visual_candidates=s['sample_visual_candidate'], verdict=verdict))
        split_rows = [r for r in clip_rows if r['gsr_split'] == name]
        games = defaultdict(set)
        for r in split_rows:
            if r['metadata_match']:
                games[r['gsr_game']] |= set(r['tracking_game'].split(';'))
        summary[name] = dict(clips=len(split_rows), frames=sum(r['frames'] for r in split_rows),
                             verdicts=dict(Counter(r['verdict'] for r in split_rows)),
                             clips_by_metadata={s: sum(any(t.startswith(s + '/') for t in r['metadata_match'].split(';'))
                                                       for r in split_rows) for s in tracking_splits},
                             byte_identical_frames=sum(r['byte_identical_frames'] for r in split_rows),
                             sample_pixel_identical=sum(r['sample_pixel_identical'] for r in split_rows),
                             sample_visual_candidates=sum(r['sample_visual_candidates'] for r in split_rows),
                             gsr_games=sorted({r['gsr_game'] for r in split_rows}, key=lambda g: (len(g), g)),
                             game_map={g: sorted(v) for g, v in sorted(games.items())},
                             matched_tracking_frames=dict(Counter(
                                 f['tracking_split'] for f in frame_rows if f['gsr_split'] == name)))
        print(name, json.dumps(summary[name]), flush=True)
    tracking_games = defaultdict(set)
    for (split, _), values in tracking.items():
        tracking_games[split].add(values['game'])
    target = utc_folder(args.output)
    write_csv(target / 'gsr_vs_sntracking.csv', clip_rows, list(clip_rows[0]))
    write_csv(target / 'gsr_vs_sntracking_frames.csv', frame_rows,
              ['gsr_split', 'gsr_image', 'tracking_split', 'tracking_image', 'kind', 'dhash_distance', 'same_frame'])
    excluded = sorted({(r['gsr_split'], r['gsr_clip']) for r in clip_rows
                       if r['verdict'] in ('identical', 'partial', 'metadata_only')})
    (target / 'exclude_from_training.txt').write_text(''.join(f'{s}/{c}\n' for s, c in excluded), encoding='utf-8')
    result = dict(gsr=summary, tracking={s: dict(folder=str(f), clips=sum(k[0] == s for k in tracking),
                                                 frames=sum(r['split'] == s for r in tracking_rows),
                                                 games=sorted(tracking_games[s], key=lambda g: (len(g), g)))
                                          for s, f in tracking_splits.items()},
                  excluded_gsr_clips=len(excluded), sample_step=args.sample_step, distance=args.distance,
                  seconds=round(time.perf_counter() - start, 1))
    (target / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    manifest = dict(run_info(args), inputs=dict(gsr={k: str(v) for k, v in gsr_splits.items()},
                                                tracking={k: str(v) for k, v in tracking_splits.items()}),
                    files={n: sha256(target / n) for n in ('gsr_vs_sntracking.csv', 'gsr_vs_sntracking_frames.csv',
                                                           'exclude_from_training.txt', 'summary.json')})
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Report:', target)


def view(args):
    path = args.processed / f'{args.split}_{args.rate}'
    manifest = build_view(args.source, path, args.split, args.step)
    target = utc_folder(args.runs)
    with (path / 'frames.csv').open(encoding='utf-8') as stream:
        frames = list(csv.DictReader(stream))
    overlays = save_overlays(path, frames, target, args.overlays, args.seed) if args.overlays else {}
    run = dict(run_info(args), view=str(path.resolve()), view_manifest_sha256=sha256(path / 'manifest.json'),
               source_json=str(args.source_json), source_json_sha256=(
                   sha256(args.source_json) if args.source_json.is_file() else None),
               overlays=overlays, summary=manifest)
    (target / 'manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=str(target), view=str(path), counts=manifest['counts'], games=manifest['games'],
                          overlays=overlays), indent=2))


def main():
    parser = ArgumentParser()
    parser.add_argument('--source-json', type=Path, default=ROOT / 'data/raw/sn_gsr_2025.source.json')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('extract')
    command.add_argument('--archive', type=Path, required=True)
    command.add_argument('--archive-name', required=True, help='file name in the Hugging Face repository')
    command.add_argument('--split', required=True)
    command.add_argument('--sha256')
    command.add_argument('--raw', type=Path, default=ROOT / 'data/raw/sn_gsr_2025')
    command = commands.add_parser('audit')
    command.add_argument('--split', action='append', required=True, metavar='NAME=FOLDER')
    command.add_argument('--output', type=Path, default=ROOT / 'outputs/gsr_audit')
    command = commands.add_parser('overlap')
    command.add_argument('--gsr', action='append', required=True, metavar='NAME=FOLDER')
    command.add_argument('--tracking', action='append', required=True, metavar='NAME=FOLDER')
    command.add_argument('--sample-step', type=int, default=25)
    command.add_argument('--distance', type=int, default=4)
    command.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    command.add_argument('--output', type=Path, default=ROOT / 'outputs/overlap')
    command = commands.add_parser('view')
    command.add_argument('--source', type=Path, required=True, help='split folder with SNGS-* clips')
    command.add_argument('--split', required=True)
    command.add_argument('--step', type=int, default=25, help='keep every n-th frame from frame 1')
    command.add_argument('--rate', default='1fps', help='view name suffix')
    command.add_argument('--processed', type=Path, default=ROOT / 'data/processed/gsr')
    command.add_argument('--runs', type=Path, default=ROOT / 'outputs/gsr_view')
    command.add_argument('--overlays', type=int, default=20)
    command.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    dict(extract=extract, audit=audit, overlap=overlap, view=view)[args.command](args)


if __name__ == '__main__':
    main()
