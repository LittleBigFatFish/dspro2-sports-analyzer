"""Fine-tune the detector on one training set; one run folder per call, resumable from last.pt."""
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

from sports_analyzer.dataset_preparation import sha256
from sports_analyzer.training import StopRequested, fit, keep_local, resume_run, start_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/training.yaml')
    parser.add_argument('--set', help='folder under sets_root in the config')
    parser.add_argument('--device', help='cpu or a CUDA index such as 0')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--stop-after', type=int, help='stop after this many epochs; the run stays resumable')
    parser.add_argument('--resume', help='run id under the output folder to continue from its last.pt')
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    if args.resume and (args.set or args.epochs):
        parser.error('a resumed run keeps its set and epochs')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    device = args.device or config['device']
    device = int(device) if str(device).isdigit() else device
    keep_local(ROOT / Path(config['model']).parent)
    from ultralytics import YOLO
    overrides = {k: v for k, v in (('epochs', args.epochs), ('batch', args.batch)) if v is not None}
    if args.resume:
        target = ROOT / config['output'] / args.resume
        manifest, kwargs = resume_run(target, ROOT, device, overrides)
        model = YOLO(str(target / 'weights/last.pt'))
    else:
        info = {'config_file': args.config.resolve().relative_to(ROOT).as_posix(), 'config_sha256': sha256(args.config),
                'script_sha256': sha256(Path(__file__)), 'args': {k: str(v) for k, v in vars(args).items()},
                'note': args.note}
        target, manifest, kwargs = start_run(config, args.set or config['set'], ROOT, device, overrides, info)
        model = YOLO(str(ROOT / config['model']), task='detect')
    print('->', target.relative_to(ROOT).as_posix(), flush=True)
    try:
        manifest = fit(model, target, manifest, kwargs, args.stop_after)
    except StopRequested as stop:
        print(f'{stop}; continue with --resume {target.name}')
        return
    print(f'finished: {manifest["epochs_completed"]} epochs, best.pt sha256 {manifest["best_sha256"]}')


if __name__ == '__main__':
    main()
