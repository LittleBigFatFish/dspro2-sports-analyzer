"""CPU latency per image through Detector.predict, median and p90 after warm-up."""
from pathlib import Path
import argparse
import csv
import json
import os
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'Ultralytics'))
os.environ.setdefault('YOLO_OFFLINE', 'True')
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')

from sports_analyzer.comparison import (yolo_view, view_digest, time_view, summarise_timings,
                                        environment, new_run_dir, sha256)
from sports_analyzer.inference import Detector


def shown(path: Path):
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/model_comparison.yaml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    settings = config['latency']
    entry = config['datasets'][settings['dataset']]
    dataset = ROOT / entry['root']
    view = yolo_view(dataset, [settings['split']])[:settings['images']]
    for record in view:
        record['image_sha256'] = sha256(dataset / record['image'])
    env = environment(args.device)
    rows, summary, models = [], [], []
    runs = [(m, 'default') for m in config['models']] + [(m, 'nms_free') for m in settings.get('nms_free_models', [])]
    for model, variant in runs:
        model_path = ROOT / model
        detector = Detector(model_path, device=args.device, nms=False if variant == 'nms_free' else None)
        models.append({'model': model, 'variant': variant, 'sha256': sha256(model_path)})
        for size in config['image_sizes']:
            timed = time_view(detector, dataset, view, size, settings['confidence'], settings['warmup'])
            rows.extend({'model': model_path.name, 'variant': variant, 'imgsz': size, **r} for r in timed)
            result = {'model': model_path.name, 'variant': variant, 'imgsz': size,
                      **summarise_timings([r['end_to_end_ms'] for r in timed])}
            summary.append(result)
            print(f"{result['model']} {variant} imgsz={size}: median {result['median_ms']:.1f} ms, "
                  f"p90 {result['p90_ms']:.1f} ms", flush=True)
    target = new_run_dir(ROOT / settings['output'])
    with (target / 'timings.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'models': models, 'image_sizes': config['image_sizes'], **settings, **env,
                'dataset': {'name': settings['dataset'], 'root': entry['root'], 'images': len(view),
                            'view_sha256': view_digest(view)},
                'timing': 'perf_counter around Detector.predict: image read, preprocess, inference, NMS, box conversion',
                'nms_free': 'nms=False in predict selects the one-to-one head; no NMS step',
                'summary': summary, 'note': args.note}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Timings:', shown(target))


if __name__ == '__main__':
    main()
