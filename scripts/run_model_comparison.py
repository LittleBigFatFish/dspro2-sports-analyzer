"""Raw predictions of pretrained detectors, one run id per model and size."""
from pathlib import Path
import argparse
import os
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Ultralytics must never download fonts, weights or packages.
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'Ultralytics'))
os.environ.setdefault('YOLO_OFFLINE', 'True')
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')

from sports_analyzer.comparison import (build_view, group_counts, view_digest, predict_view, environment,
                                        save_predictions, sha256)
from sports_analyzer.inference import Detector


def shown(path: Path):
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/model_comparison.yaml')
    parser.add_argument('--dataset', help='entry under datasets in the config')
    parser.add_argument('--device', help='cpu or a CUDA index such as 0')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--sizes', nargs='+', type=int)
    parser.add_argument('--nms-free', action='store_true', help='one-to-one head of YOLO26 models')
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--limit', type=int, help='evenly spaced subset, for quick checks')
    parser.add_argument('--output')
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    device = args.device or config['device']
    device = int(device) if str(device).isdigit() else device
    name = args.dataset or config['dataset']
    settings = config['datasets'][name]
    if not settings.get('audit_run'):
        parser.error(f'{name}: set audit_run in the config before running')
    dataset = ROOT / settings['root']
    view = build_view(dataset, settings, args.splits, ROOT)
    if args.limit:
        view = view[::max(1, len(view)//args.limit)][:args.limit]
    env = environment(device)
    for model in args.models or config['models']:
        model_path = ROOT / model
        detector = Detector(model_path, device=device, nms=False if args.nms_free else None)
        for size in args.sizes or config['image_sizes']:
            images = predict_view(detector, dataset, view, size, config['confidence'])
            manifest = {'model': model_path.relative_to(ROOT).as_posix(), 'model_sha256': sha256(model_path),
                        'imgsz': size, 'variant': 'nms_free' if args.nms_free else 'default',
                        'nms': detector.nms, 'confidence': config['confidence'], **env,
                        'dataset': {'name': name, 'root': settings['root'],
                                    'audit_run': Path(settings['audit_run']).name if settings.get('audit_run') else None,
                                    'extra_exclusions': settings.get('extra_exclusions', []),
                                    'exclude_groups': settings.get('exclude_groups', []),
                                    'images': len(view), 'view_sha256': view_digest(view),
                                    'groups': group_counts(view)},
                        'config': {**config, 'datasets': {name: config['datasets'][name]}}, 'args': {k: v for k, v in vars(args).items() if k != 'config'},
                        'note': args.note}
            target = save_predictions(ROOT / (args.output or config['output']), images, manifest)
            print(f'{model_path.name} {manifest["variant"]} imgsz={size}: {len(images)} images ->', shown(target), flush=True)


if __name__ == '__main__':
    main()
