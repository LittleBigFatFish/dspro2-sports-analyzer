"""Score saved predictions against the audited labels; never runs a model."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import platform
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports_analyzer.annotations.yolo import Box, load_names
from sports_analyzer.comparison import new_run_dir, view_digest
from sports_analyzer.dataset_preparation import git_state, sha256
from sports_analyzer.evaluation import (GT_MAPPING, PRED_MAPPING, remap, match, ranked_curve,
                                        interpolated_ap, wilson)

LABELS = ('person', 'ball')
FIELDS = ['run_id', 'dataset', 'model', 'variant', 'imgsz', 'group', 'metric', 'value', 'low', 'high', 'n']


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def variant(manifest) -> str:
    # Runs before the NMS-free comparison have no variant field.
    return manifest.get('variant', 'default')


def find_runs(runs_root: Path, names=None):
    """Latest complete run per view, model and size; --limit runs are skipped."""
    folders = [runs_root / n for n in names] if names else sorted(p for p in runs_root.iterdir() if p.is_dir())
    latest, skipped = {}, []
    for folder in folders:
        if not (folder / 'manifest.json').is_file() or not (folder / 'predictions.json').is_file():
            skipped.append({'run_id': folder.name, 'reason': 'incomplete'})
            continue
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('args', {}).get('limit'):
            skipped.append({'run_id': folder.name, 'reason': 'limited quick check'})
            continue
        key = (manifest['dataset']['name'], manifest['dataset']['view_sha256'], manifest['model'], variant(manifest),
               manifest['imgsz'])
        if key in latest:
            skipped.append({'run_id': latest[key][0].name, 'reason': f'superseded by {folder.name}'})
        latest[key] = folder, manifest
    return sorted(latest.values(), key=lambda item: item[0].name), skipped


def load_run(folder: Path, manifest):
    if sha256(folder / 'predictions.json') != manifest['predictions_sha256']:
        raise ValueError(f'{folder.name}: predictions.json differs from its manifest')
    records = json.loads((folder / 'predictions.json').read_text(encoding='utf-8'))['images']
    if len(records) != manifest['dataset']['images'] or view_digest(records) != manifest['dataset']['view_sha256']:
        raise ValueError(f'{folder.name}: predictions do not match the recorded view')
    predictions = {r['image']: remap([Box(b['label'], tuple(b['xyxy']), b['score']) for b in r['boxes']],
                                     PRED_MAPPING, strict=False) for r in records}
    return records, predictions


def load_ground_truth(root: Path, audit: Path, records):
    names = load_names(root / 'data.yaml')
    audited = {r['image']: r for r in read_csv(audit / 'images.csv')}
    truth, sizes = {}, {}
    for record in records:
        row = audited.get(record['image'])
        if row is None or row['status'] != 'ok':
            raise ValueError(f'{record["image"]}: not a clean image in audit {audit.name}')
        if row['image_sha256'] != record['image_sha256']:
            raise ValueError(f'{record["image"]}: image differs from audit {audit.name}')
        if sha256(root / row['label']) != row['label_sha256']:
            raise ValueError(f'{row["label"]}: label changed since audit {audit.name}')
        truth[record['image']], sizes[record['image']] = [], (int(row['width']), int(row['height']))
    for row in read_csv(audit / 'boxes.csv'):
        if row['image'] in truth:
            # Check by name: Soccana orders its classes Player, Ball, Referee.
            if names[int(row['class_id'])] != row['label']:
                raise ValueError(f'{row["image"]}: class {row["class_id"]} is not {row["label"]!r}')
            label = row['label'].lower()
            if label not in GT_MAPPING:
                raise ValueError(f'No evaluation mapping for {row["label"]!r}')
            truth[row['image']].append(Box(label, tuple(float(row[k]) for k in ('x1', 'y1', 'x2', 'y2'))))
    return truth, sizes


def load_ignore(root: Path) -> dict:
    regions = defaultdict(list)
    if (root / 'ignore.csv').is_file():
        for row in read_csv(root / 'ignore.csv'):
            regions[row['image']].append(tuple(float(row[k]) for k in ('x1', 'y1', 'x2', 'y2')))
    return dict(regions)


def short_side(box: Box, size, imgsz: int) -> float:
    # Letterbox scale: the longer image side becomes imgsz.
    return min(box.xyxy[2]-box.xyxy[0], box.xyxy[3]-box.xyxy[1]) * imgsz / max(size)


def size_bin(side: float, edges) -> str:
    low, high = edges
    return f'<{low}' if side < low else f'{low}-{high}' if side <= high else f'>{high}'


def score_images(truth, predictions, sizes, ignore, imgsz, thresholds, ignore_iou, edges):
    # A confidence cut keeps a prefix of the ranking, so one pass serves every cut.
    scored = {}
    for name, boxes in truth.items():
        for label in LABELS:
            gt = [b for b in boxes if GT_MAPPING[b.label] == label]
            pred = [b for b in predictions.get(name, []) if b.label == label]
            order, results = match(gt, pred, thresholds, ignore.get(name, ()), ignore_iou)
            scores = np.array([pred[i].score for i in order], float)
            classes = np.array([b.label for b in gt], str)
            bins = np.array([size_bin(short_side(b, sizes[name], imgsz), edges) for b in gt], str)
            for threshold, (assigned, ignored) in zip(thresholds, results):
                gt_scores = np.full(len(gt), -1.0)
                gt_scores[assigned[assigned >= 0]] = scores[assigned >= 0]
                scored[name, label, threshold] = (scores[~ignored], assigned[~ignored] >= 0, gt_scores, classes, bins)
    return scored


def pooled(scored, names, label, threshold):
    parts = [scored[n, label, threshold] for n in names]
    return [np.concatenate([p[i] for p in parts]) if parts else np.array([]) for i in range(5)]


def ap(scores, matched, total):
    return interpolated_ap(*ranked_curve(scores, matched, total)) if total else None


def proportion(metric, hits, n):
    low, high = wilson(int(hits), int(n))
    return {'metric': metric, 'value': hits/n if n else None, 'low': low, 'high': high, 'n': int(n)}


def group_metrics(scored, names, settings, full=True) -> list[dict]:
    c, t = settings['report_confidence'], settings['match_iou']
    rows = []
    for label in LABELS:
        scores, matched, gt_scores, classes, bins = pooled(scored, names, label, t)
        rows.append({'metric': f'{label}_AP50', 'value': ap(scores, matched, len(gt_scores)),
                     'low': None, 'high': None, 'n': len(gt_scores)})
        rows.append(proportion(f'{label}_recall_c{c}_iou{t}', (gt_scores >= c).sum(), len(gt_scores)))
        if not full:
            continue
        values = [ap(*pooled(scored, names, label, u)[:2], len(gt_scores)) for u in settings['ap_iou']]
        rows.append({'metric': f'{label}_AP50-95', 'value': None if None in values else sum(values)/len(values),
                     'low': None, 'high': None, 'n': len(gt_scores)})
        for u in settings['grid_iou']:
            scores, matched, gt_scores, _, _ = pooled(scored, names, label, u)
            for cut in settings['grid_confidence']:
                kept = scores >= cut
                rows.append(proportion(f'{label}_precision_c{cut}_iou{u}', (matched & kept).sum(), kept.sum()))
                rows.append(proportion(f'{label}_recall_c{cut}_iou{u}', (gt_scores >= cut).sum(), len(gt_scores)))
        scores, matched, gt_scores, classes, bins = pooled(scored, names, label, t)
        for name in sorted(set(classes.tolist())):
            selected = classes == name
            rows.append(proportion(f'class_{name}_recall_c{c}_iou{t}', (gt_scores[selected] >= c).sum(), selected.sum()))
        if label == 'ball':
            low, high = settings['ball_size_edges']
            for bin_name in (f'<{low}', f'{low}-{high}', f'>{high}'):
                selected = bins == bin_name
                rows.append(proportion(f'ball_recall_size{bin_name}_c{c}_iou{t}',
                                       (gt_scores[selected] >= c).sum(), selected.sum()))
    return rows


def cluster_bootstrap_ap(scored, clusters: dict, label, threshold, resamples, seed):
    """Percentile 95% interval of pooled AP, resampling whole clusters."""
    parts = [pooled(scored, names, label, threshold)[:3] for names in clusters.values()]
    picks = np.random.default_rng(seed).integers(0, len(parts), size=(resamples, len(parts)))
    values = []
    for pick in picks:
        scores = np.concatenate([parts[i][0] for i in pick])
        matched = np.concatenate([parts[i][1] for i in pick])
        value = ap(scores, matched, sum(len(parts[i][2]) for i in pick))
        if value is not None:
            values.append(value)
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5])) if values else (None, None)


def describe(records, dataset: dict):
    for record in records:
        if dataset.get('clip_key'):
            record['clip'] = record[dataset['clip_key']]
        if 'video_game_sources' in dataset:
            record['footage'] = 'video_game' if record['source'] in dataset['video_game_sources'] else 'real'
    return records


def build_groups(records, dataset: dict, min_images: int):
    groups = [('all', [r['image'] for r in records], 'all')]
    for level, keys_list in (('group', dataset.get('groups', [])), ('small', dataset.get('small_groups', []))):
        for keys in keys_list:
            members = defaultdict(list)
            for record in records:
                members['|'.join(f'{k}={record[k]}' for k in keys)].append(record['image'])
            groups.extend((name, images, level) for name, images in sorted(members.items())
                          if level == 'group' or len(images) >= min_images)
    return groups


def write_csv(path: Path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()} for row in rows)


def primary_table(rows, notes, settings, small):
    c, t = settings['report_confidence'], settings['match_iou']
    wanted = {'images': 'images', 'person_AP50': 'person_AP50', 'ball_AP50': 'ball_AP50',
              f'person_recall_c{c}_iou{t}': 'person_recall', f'ball_recall_c{c}_iou{t}': 'ball_recall'}
    table = defaultdict(dict)
    for row in rows:
        name = wanted.get(row['metric'])
        if name is None or (row['run_id'], row['group']) in small:
            continue
        values = table[row['dataset'], row['group'], Path(row['model']).stem, row['variant'], row['imgsz'], row['run_id']]
        if name == 'images':
            values['images'] = row['value']
            continue
        values.update({name: row['value'], f'{name}_low': row['low'], f'{name}_high': row['high']})
        if 'recall' in name:
            values[name.replace('recall', 'n')] = row['n']
    return sorted(({'dataset': dataset, 'group': group, 'note': notes.get((dataset, group), ''), 'model': model,
                    'variant': name, 'imgsz': imgsz, 'run_id': run_id, **values}
                   for (dataset, group, model, name, imgsz, run_id), values in table.items()),
                  key=lambda r: (r['dataset'], r['group'] != 'all', r['group'], r['model'], r['variant'], r['imgsz']))


PRIMARY_FIELDS = ['dataset', 'group', 'note', 'model', 'variant', 'imgsz', 'run_id', 'images',
                  'person_AP50', 'person_AP50_low', 'person_AP50_high', 'ball_AP50', 'ball_AP50_low', 'ball_AP50_high',
                  'person_recall', 'person_recall_low', 'person_recall_high', 'person_n',
                  'ball_recall', 'ball_recall_low', 'ball_recall_high', 'ball_n']


def cross_check(rows, settings, base: Path):
    reference = settings['reference']
    c, t = settings['report_confidence'], settings['match_iou']
    scored = {(r['imgsz'], r['metric']): r for r in rows if r['dataset'] == reference['dataset']
              and r['model'] == reference['model'] and r['variant'] == 'default' and r['group'] == reference['group']}
    output = []
    for ref in read_csv(base / reference['path']):
        imgsz, label = int(ref['imgsz']), ref['label']
        recall = scored.get((imgsz, f'{label}_recall_c{c}_iou{t}'))
        precision = scored.get((imgsz, f'{label}_precision_c{c}_iou{t}'))
        if not recall:
            continue
        tp = round(recall['value']*recall['n'])
        mine = {'tp': tp, 'fp': precision['n']-tp, 'fn': recall['n']-tp, 'precision': precision['value'],
                'recall': recall['value'], 'AP50': scored[imgsz, f'{label}_AP50']['value'],
                'AP50-95': scored[imgsz, f'{label}_AP50-95']['value']}
        for metric, value in mine.items():
            output.append({'imgsz': imgsz, 'label': label, 'metric': metric, 'reference': float(ref[metric]),
                           'scored': value, 'difference': value-float(ref[metric])})
    return output


def figures(target: Path, rows, curves, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e34948']
    c, t = settings['report_confidence'], settings['match_iou']
    panels = [(d, g) for d, entry in settings['datasets'].items() for g in entry.get('figure_groups', [])
              if any(r['dataset'] == d and r['group'] == g for r in rows)]
    models = list(dict.fromkeys((r['model'], r['variant']) for r in rows))
    sizes = sorted({r['imgsz'] for r in rows})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8*len(panels), 3.8), sharey=True, layout='constrained',
                             squeeze=False)
    for ax, (dataset, group) in zip(axes[0], panels):
        n = None
        for k, size in enumerate(sizes):
            for x, (model, name) in enumerate(models):
                row = next((r for r in rows if (r['dataset'], r['group'], r['model'], r['variant'], r['imgsz'], r['metric'])
                            == (dataset, group, model, name, size, f'ball_recall_c{c}_iou{t}')), None)
                if row is None or row['value'] is None:
                    continue
                n = row['n']
                ax.errorbar(x + (k-0.5*(len(sizes)-1))*0.2, row['value'],
                            yerr=[[row['value']-row['low']], [row['high']-row['value']]], fmt='os^'[k % 3],
                            color=colours[k], markersize=6, capsize=3, lw=1.5,
                            label=None if f'imgsz {size}' in ax.get_legend_handles_labels()[1] else f'imgsz {size}')
        ax.set_xticks(range(len(models)), [Path(m).stem + ('' if v == 'default' else f'\n{v}') for m, v in models],
                      fontsize=8)
        title = settings['datasets'][dataset].get('title', dataset)
        ax.set_title(f'{title}\n{group} (n={n} balls)', fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.2)
    axes[0][0].set_ylabel(f'Ball recall (conf {c}, IoU {t}), Wilson 95%')
    axes[0][0].legend(fontsize=8, loc='upper left')
    fig.savefig(target / 'ball_recall.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, label in zip(axes, LABELS):
        for k, ((dataset, group), curve) in enumerate(curves.items()):
            recalls, precisions, total = curve[label]
            title = settings['datasets'][dataset].get('title', dataset)
            ax.plot(recalls, precisions, color=colours[k % len(colours)], lw=2, ls=['-', '--', '-.', ':'][k % 4],
                    label=f'{title}, {group} (n={total})')
        ax.set(title=f'{label}: precision-recall at IoU {t}, {Path(settings["figure_model"]).stem} imgsz '
                     f'{settings["figure_imgsz"]}', xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(target / 'pr_curves.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/scoring.yaml')
    parser.add_argument('--runs', nargs='+', help='run ids under the runs folder; default all')
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    thresholds = sorted(set(settings['ap_iou']) | set(settings['grid_iou']) | {settings['match_iou']})
    runs, skipped = find_runs(ROOT / settings['runs'], args.runs)
    rows, notes, curves, used, truths, small = [], {}, {}, [], {}, set()
    for folder, manifest in runs:
        name = manifest['dataset']['name']
        dataset = settings['datasets'].get(name, {})
        root = ROOT / manifest['dataset']['root']
        audit = ROOT / settings['audit_root'] / manifest['dataset']['audit_run']
        records, predictions = load_run(folder, manifest)
        records = describe(records, dataset)
        key = (root, audit, manifest['dataset']['view_sha256'])
        if key not in truths:
            truths[key] = load_ground_truth(root, audit, records)
        truth, sizes = truths[key]
        ignore = load_ignore(root)
        scored = score_images(truth, predictions, sizes, ignore, manifest['imgsz'], thresholds,
                              settings['ignore_iou'], settings['ball_size_edges'])
        groups = build_groups(records, dataset, settings['min_group_images'])
        clusters = defaultdict(list)
        for record in records:
            if dataset.get('bootstrap_by'):
                clusters[record[dataset['bootstrap_by']]].append(record['image'])
        boot = settings['bootstrap']
        for group, names, level in groups:
            notes[name, group] = dataset.get('notes', {}).get(group, '')
            if level == 'small':
                small.add((folder.name, group))
            base = {'run_id': folder.name, 'dataset': name, 'model': manifest['model'], 'variant': variant(manifest),
                    'imgsz': manifest['imgsz'], 'group': group}
            metrics = group_metrics(scored, names, settings, full=level != 'small')
            if level == 'all' and len(clusters) >= boot['min_clusters']:
                for row in metrics:
                    if row['metric'].endswith('_AP50'):
                        row['low'], row['high'] = cluster_bootstrap_ap(
                            scored, clusters, row['metric'].split('_')[0], settings['match_iou'],
                            boot['resamples'], boot['seed'])
            metrics.append({'metric': 'images', 'value': len(names), 'low': None, 'high': None, 'n': len(names)})
            rows.extend({**base, **row} for row in metrics)
        if (manifest['model'], variant(manifest), manifest['imgsz']) == \
                (settings['figure_model'], 'default', settings['figure_imgsz']):
            for group, names, _ in groups:
                if group in dataset.get('figure_groups', []):
                    curves[name, group] = {}
                    for label in LABELS:
                        scores, matched, gt_scores, _, _ = pooled(scored, names, label, settings['match_iou'])
                        curves[name, group][label] = (*ranked_curve(scores, matched, len(gt_scores)), len(gt_scores))
        used.append({'run_id': folder.name, 'dataset': name, 'model': manifest['model'], 'variant': variant(manifest),
                     'model_sha256': manifest['model_sha256'], 'imgsz': manifest['imgsz'],
                     'images': manifest['dataset']['images'], 'view_sha256': manifest['dataset']['view_sha256'],
                     'audit_run': manifest['dataset']['audit_run'], 'predictions_sha256': manifest['predictions_sha256'],
                     'confidence': manifest['confidence'], 'device': manifest['device'],
                     'inference_git_commit': manifest.get('git_commit'),
                     'ignore_regions': sum(len(v) for k, v in ignore.items() if k in truth)})
        print(f'{folder.name} {name} {Path(manifest["model"]).stem} {variant(manifest)} {manifest["imgsz"]}: '
              f'{len(groups)} groups', flush=True)
    target = new_run_dir(ROOT / settings['output'])
    write_csv(target / 'metrics.csv', rows, FIELDS)
    write_csv(target / 'primary.csv', primary_table(rows, notes, settings, small), PRIMARY_FIELDS)
    check = cross_check(rows, settings, ROOT) if (ROOT / settings['reference']['path']).is_file() else []
    if check:
        write_csv(target / 'crosscheck.csv', check, ['imgsz', 'label', 'metric', 'reference', 'scored', 'difference'])
    figures(target, rows, curves, settings)
    import matplotlib
    manifest = {'runs': used, 'skipped_runs': skipped, 'config': settings, 'args': vars(args) | {'config': str(args.config)},
                'notes': {f'{d} {g}': n for (d, g), n in notes.items() if n},
                'interval_methods': {'recall_precision': 'Wilson 95%, n = ground-truth boxes (recall) or kept predictions (precision)',
                                     'AP50': 'cluster bootstrap by clip, percentile 95%, dataset total only, where at least '
                                             f'{settings["bootstrap"]["min_clusters"]} clips'},
                'git': git_state(), 'python': platform.python_version(), 'numpy': np.__version__,
                'matplotlib': matplotlib.__version__,
                'code_sha256': {'script': sha256(Path(__file__)),
                                'evaluation': sha256(ROOT / 'sports_analyzer/evaluation.py')},
                'created_utc': datetime.now(timezone.utc).isoformat(), 'note': args.note}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    print('->', target.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
