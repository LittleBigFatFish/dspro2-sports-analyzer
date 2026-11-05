"""MultiSports (SportsAction) football: extract, audit the tubes and build a YOLO view."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import platform
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_soccernet_view import save_overlays
from sports_analyzer.dataset_preparation import git_state, sha256
from sports_analyzer.sportsaction import SPORT, audit_split, build_view, load_annotations, shared_sources

REPOSITORY = 'https://huggingface.co/datasets/MCG-NJU/SportsAction'
REVISION = '01600ce7eabbf42a5ee7c82b82f49a11597b3a5f'
SOURCE = dict(
    name='MultiSports (Hugging Face MCG-NJU/SportsAction), football part', repository=REPOSITORY, revision=REVISION,
    licence='CC BY-NC 4.0 (gated: non-commercial use accepted on Hugging Face); videos are YouTube competition '
            'footage, local coursework use only, no redistribution, cite the paper.',
    citation='Li et al., "MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions", '
             'ICCV 2021 (arXiv 2105.07404)',
    extraction="Python tarfile with filter='data' into a new empty folder per split; football.tar only",
    splits={})


def utc_folder(root):
    target = Path(root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_info(script_args):
    return dict(arguments={k: str(v) for k, v in vars(script_args).items()}, python=platform.python_version(),
                device='cpu', platform=platform.platform(), script_sha256=sha256(Path(__file__)),
                code_sha256=sha256(ROOT / 'sports_analyzer/sportsaction.py'), git=git_state(),
                created_utc=datetime.now(timezone.utc).isoformat())


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def extract(args):
    target = args.raw / args.split
    digest = sha256(args.archive)
    if args.sha256 and digest != args.sha256:
        raise ValueError(f'{args.archive}: sha256 {digest}, expected {args.sha256}')
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive, 'r:') as source:
        members = source.getmembers()
        names = Counter(m.name.lower() for m in members)
        if len(names) != len(members):
            raise ValueError(f'Case collisions: {[n for n, c in names.items() if c > 1][:5]}')
        if any(not m.name.startswith(f'{SPORT}') for m in members):
            raise ValueError(f'{args.archive}: members outside {SPORT}/')
        source.extractall(target, filter='data')
    files = [p for p in target.rglob('*') if p.is_file()]
    record = json.loads(args.source_json.read_text(encoding='utf-8')) if args.source_json.is_file() else SOURCE
    record['splits'][args.split] = dict(
        url=f'{REPOSITORY}/resolve/{REVISION}/{args.archive_name}', file=args.archive_name,
        local_archive=args.archive.resolve().relative_to(ROOT).as_posix(), bytes=args.archive.stat().st_size,
        sha256=digest, downloaded_utc=datetime.fromtimestamp(args.archive.stat().st_mtime, timezone.utc)
        .strftime('%Y-%m-%dT%H:%M:%SZ'), extracted_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        extracted_to=target.resolve().relative_to(ROOT).as_posix(), extracted_files=len(files),
        extracted_bytes=sum(p.stat().st_size for p in files))
    for path in args.annotation:
        record.setdefault('annotations', {})[path.name] = dict(
            url=f'{REPOSITORY}/resolve/{REVISION}/{path.resolve().relative_to(args.downloads).as_posix()}',
            local=path.resolve().relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha256(path))
    args.source_json.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record['splits'][args.split], indent=2))


def audit(args):
    gt = load_annotations(args.annotations)
    target = utc_folder(args.output)
    summaries = {}
    for split in ('train', 'val'):
        result = audit_split(gt, split)
        summaries[split] = result['summary']
        write_csv(target / f'videos_{split}.csv', result['videos'], list(result['videos'][0]))
    shared = shared_sources(gt)
    summary = dict(annotations=str(args.annotations.resolve()), annotations_sha256=sha256(args.annotations),
                   labels=[l for l in gt['labels'] if l.startswith(SPORT)], splits=summaries,
                   shared_source_videos=dict(count=len(shared), examples=shared[:10]), **run_info(args))
    (target / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=str(target), splits={s: dict(videos=v['videos'], sources=v['source_videos'],
                                                            counts=v['counts'], issues=v['issues'])
                                                  for s, v in summaries.items()},
                          shared_source_videos=len(shared)), indent=2))


def view(args):
    gt = load_annotations(args.annotations)
    output = args.processed / f'{args.split}_{args.rate}'
    manifest = build_view(gt, args.videos, output, args.split, args.step)
    target = utc_folder(args.runs)
    with (output / 'frames.csv').open(encoding='utf-8') as stream:
        frames = list(csv.DictReader(stream))
    overlays = save_overlays(output, frames, target, args.overlays, args.seed) if args.overlays else {}
    run = dict(view=str(output.resolve()), view_manifest_sha256=sha256(output / 'manifest.json'),
               annotations_sha256=sha256(args.annotations), overlays=overlays, summary=manifest, **run_info(args))
    (target / 'manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=str(target), view=str(output), counts=manifest['counts'], overlays=overlays), indent=2))


def main():
    parser = ArgumentParser()
    parser.add_argument('--source-json', type=Path, default=ROOT / 'data/raw/sportsaction.source.json')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('extract')
    command.add_argument('--archive', type=Path, required=True)
    command.add_argument('--archive-name', required=True, help='path in the Hugging Face repository')
    command.add_argument('--split', required=True)
    command.add_argument('--sha256')
    command.add_argument('--annotation', type=Path, action='append', default=[], help='pickle to record')
    command.add_argument('--downloads', type=Path, default=ROOT / 'data/raw/_downloads/sportsaction')
    command.add_argument('--raw', type=Path, default=ROOT / 'data/raw/sportsaction')
    command = commands.add_parser('audit')
    command.add_argument('--annotations', type=Path, required=True, help='multisports_GT.pkl')
    command.add_argument('--output', type=Path, default=ROOT / 'outputs/sportsaction_audit')
    command = commands.add_parser('view')
    command.add_argument('--annotations', type=Path, required=True, help='multisports_GT.pkl')
    command.add_argument('--videos', type=Path, required=True, help='folder that holds football/*.mp4')
    command.add_argument('--split', required=True, choices=['train', 'val'])
    command.add_argument('--step', type=int, default=25, help='keep every n-th frame from frame 1')
    command.add_argument('--rate', default='1fps', help='view name suffix')
    command.add_argument('--processed', type=Path, default=ROOT / 'data/processed/sportsaction')
    command.add_argument('--runs', type=Path, default=ROOT / 'outputs/sportsaction_view')
    command.add_argument('--overlays', type=int, default=20)
    command.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    dict(extract=extract, audit=audit, view=view)[args.command](args)


if __name__ == '__main__':
    main()
