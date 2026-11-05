"""Training images that also appear in a test set, exact, near or mirrored."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time

from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports_analyzer.dataset_preparation import (DUPLICATE_KINDS, IMAGE_SUFFIXES, duplicate_kind,
                                                  git_state, hamming_matches, image_hashes, sha256)

SOCCERNET_PATTERNS = [r'snmot', r'soccer[-_ ]?net', r'sn[-_]?(?:tracking|gsr?|mot)\b']


def hash_file(path):
    row = dict(image_sha256=sha256(path), dhash=None)
    try:
        with Image.open(path) as image:
            image.load()
            row.update(image_hashes(image))
    except (OSError, ValueError) as exc:
        row['error'] = str(exc)
    return row


def hash_folder(folder, base, workers):
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(hash_file, paths, chunksize=32))
    for path, row in zip(paths, rows):
        row['image'] = path.relative_to(base).as_posix()
    return rows


def read_audit(audit_dir):
    with (audit_dir / 'images.csv').open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def overlaps(references, sources, distance):
    plain = [int(r['dhash'], 16) for r in sources]
    pairs = set()
    for column in ('dhash', 'mirror_dhash'):
        queries = [int(r[column], 16) for r in references]
        pairs |= {(i, j) for i, j, _ in hamming_matches(queries, plain, distance)}
    mirrors = [int(r['mirror_dhash'], 16) for r in sources]
    pairs |= {(i, j) for i, j, _ in hamming_matches([int(r['dhash'], 16) for r in references], mirrors, distance)}
    by_pixels = defaultdict(list)
    for j, row in enumerate(sources):
        by_pixels[row['pixel_sha256']].append(j)
    for i, row in enumerate(references):
        pairs |= {(i, j) for j in by_pixels.get(row['mirror_pixel_sha256'], ())}
    for i, j in sorted(pairs):
        kind, plain_distance, mirror_distance = duplicate_kind(references[i], sources[j], distance)
        if kind:
            yield i, j, kind, plain_distance, mirror_distance


def name_hits(source, patterns):
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source).as_posix()
        for pattern in compiled:
            if pattern.search(relative):
                yield relative, pattern.pattern
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'data/raw/soccana_v1')
    parser.add_argument('--source-audit', type=Path, help='dataset_preparation run with images.csv hashes')
    parser.add_argument('--reference', action='append', required=True, metavar='NAME=FOLDER',
                        help='test images to protect, e.g. football_val=data/raw/Football-Image-Dataset/images/val')
    parser.add_argument('--distance', type=int, default=4)
    parser.add_argument('--pattern', action='append', help='file/folder name regex (default: SoccerNet names)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/overlap')
    args = parser.parse_args()
    start = time.perf_counter()
    source = args.source.resolve()
    if args.source_audit:
        sources = read_audit(args.source_audit)
        if Path(json.loads((args.source_audit / 'summary.json').read_text(encoding='utf-8'))['root']) != source:
            raise ValueError(f'{args.source_audit} audits a different dataset than {source}')
    else:
        sources = hash_folder(source / 'images', source, args.workers)
    sources = [r for r in sources if r['dhash']]
    matches, exclusions, inputs = [], defaultdict(set), []
    for spec in args.reference:
        name, folder = spec.split('=', 1)
        folder = Path(folder).resolve()
        references = hash_folder(folder, folder, args.workers)
        unreadable = sum(not r['dhash'] for r in references)
        references = [r for r in references if r['dhash']]
        listing = hashlib.sha256(''.join(f'{r["image"]}\t{r["image_sha256"]}\n' for r in references).encode())
        inputs.append(dict(name=name, folder=str(folder), images=len(references), unreadable=unreadable,
                           listing_sha256=listing.hexdigest()))
        for i, j, kind, distance, mirror_distance in overlaps(references, sources, args.distance):
            matches.append(dict(reference=name, reference_image=references[i]['image'],
                                source_image=sources[j]['image'], kind=kind,
                                dhash_distance=distance, mirror_distance=mirror_distance))
            exclusions[sources[j]['image']].add(f'{name}_{kind}')
        print(f'{name}: {len(references)} images, {sum(m["reference"] == name for m in matches)} matches',
              flush=True)
    patterns = args.pattern or SOCCERNET_PATTERNS
    images = {r['image'] for r in sources}
    for relative, pattern in name_hits(source, patterns):
        matches.append(dict(reference='name_scan', reference_image='', source_image=relative,
                            kind=f'name:{pattern}', dhash_distance=None, mirror_distance=None))
        for image in images:
            if image == relative or image.startswith(relative + '/'):
                exclusions[image].add('name_pattern')
    target = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)
    with (target / 'matches.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['reference', 'reference_image', 'source_image', 'kind',
                                                    'dhash_distance', 'mirror_distance'])
        writer.writeheader()
        writer.writerows(matches)
    (target / 'exclusions.txt').write_text(''.join(f'{k}\t{";".join(sorted(v))}\n'
                                                   for k, v in sorted(exclusions.items())), encoding='utf-8')
    counts = Counter((m['reference'], m['kind']) for m in matches)
    manifest = dict(source=str(source), source_images=len(sources),
                    source_audit=str(args.source_audit.resolve()) if args.source_audit else None,
                    source_audit_images_sha256=sha256(args.source_audit / 'images.csv') if args.source_audit else None,
                    references=inputs, distance=args.distance, patterns=patterns, kinds=list(DUPLICATE_KINDS),
                    matches_by_kind={f'{r}/{k}': v for (r, k), v in sorted(counts.items())},
                    matched_reference_images={i['name']: len({m['reference_image'] for m in matches
                                                              if m['reference'] == i['name']}) for i in inputs},
                    excluded_source_images=len(exclusions), seconds=round(time.perf_counter() - start, 1),
                    python=platform.python_version(), pillow=pillow_version, device='cpu',
                    platform=platform.platform(), script_sha256=sha256(Path(__file__)), git=git_state(),
                    created_utc=datetime.now(timezone.utc).isoformat())
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=str(target), matches=manifest['matches_by_kind'],
                          excluded=len(exclusions), seconds=manifest['seconds']), indent=2))


if __name__ == '__main__':
    main()
