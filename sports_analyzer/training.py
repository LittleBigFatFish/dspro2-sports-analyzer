"""Fine-tuning runs on a built training set; the run folder keeps everything Ultralytics writes."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

from .comparison import environment, new_run_dir
from .dataset_preparation import sha256
from .evaluation import GT_MAPPING

WEIGHTS = ('best.pt', 'last.pt')


class StopRequested(Exception):
    """Raised once a checkpoint is saved, so weights/last.pt can resume."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unsaved_cache(prefix, path, cache, version):
    # Ultralytics reads the version key back from the cache it would have saved.
    cache['version'] = version


def keep_local(weights_dir: Path):
    """AMP check weights from weights_dir instead of a download; no label caches beside the data."""
    import ultralytics.utils
    from ultralytics.data import dataset
    ultralytics.utils.WEIGHTS_DIR = Path(weights_dir)
    dataset.save_dataset_cache_file = unsaved_cache


def set_summary(folder: Path, base: Path) -> dict:
    """Hashes and counts of a built set; refuses a set changed since it was built."""
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    changed = [f for f, digest in manifest['files'].items() if sha256(folder / f) != digest]
    if changed:
        raise ValueError(f'{folder.name}: {changed} differ from the set manifest')
    if set(manifest['names'].values()) != set(GT_MAPPING):
        raise ValueError(f'{folder.name}: classes {manifest["names"]} are not {sorted(GT_MAPPING)}')
    return {'name': manifest['name'], 'folder': folder.relative_to(base).as_posix(),
            'manifest_sha256': sha256(folder / 'manifest.json'), 'built_by': manifest.get('run'),
            'names': manifest['names'], 'images': manifest['images'], 'images_by_source': manifest['images_by_source'],
            'boxes_by_class': manifest['boxes_by_class']}


def base_weights(path: Path, models_manifest: Path, base: Path) -> dict:
    digest = sha256(path)
    entries = {e['file']: e for e in json.loads(models_manifest.read_text(encoding='utf-8'))['files']}
    if entries.get(path.name, {}).get('sha256') != digest:
        raise ValueError(f'{path.name}: missing from or differs from {models_manifest.name}')
    return {'path': path.relative_to(base).as_posix(), 'sha256': digest, 'source_url': entries[path.name]['source_url']}


def write_manifest(target: Path, manifest):
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')


def start_run(config, set_name, base: Path, device, overrides=None, info=None):
    """New run folder with a manifest marked running; returns it with the Ultralytics train arguments."""
    folder = base / config['sets_root'] / set_name
    summary = set_summary(folder, base)
    model = base_weights(base / config['model'], base / config['models_manifest'], base)
    target = new_run_dir(base / config['output'])
    train_args = {**config['train'], **(overrides or {})}
    manifest = {'status': 'running', 'set': summary, 'base_model': model,
                'train_args': {**train_args, 'data': f'{summary["folder"]}/data.yaml', 'device': str(device)},
                'config': config, **environment(device), 'code_sha256': sha256(Path(__file__)), **(info or {}),
                'started_utc': now(), 'ended_utc': None}
    write_manifest(target, manifest)
    kwargs = {**train_args, 'data': str(folder / 'data.yaml'), 'device': device, 'project': str(target.parent),
              'name': target.name, 'exist_ok': True}
    return target, manifest, kwargs


def resume_run(target: Path, base: Path, device, overrides=None):
    """Marks an unfinished run as running again; a killed run still says running and resumes the same way."""
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] == 'finished':
        raise ValueError(f'{target.name} is finished')
    if not (target / 'weights/last.pt').is_file():
        raise FileNotFoundError(f'{target.name}: no weights/last.pt to resume from')
    if set_summary(base / manifest['set']['folder'], base)['manifest_sha256'] != manifest['set']['manifest_sha256']:
        raise ValueError(f'{target.name}: training set changed since the run started')
    env = environment(device)
    manifest.setdefault('resumes', []).append(
        {'started_utc': now(), 'after_epoch': len(read_results(target / 'results.csv')), 'overrides': overrides or {},
         **{k: env[k] for k in ('device', 'device_name', 'torch', 'ultralytics', 'git_commit', 'git_dirty')}})
    manifest.update(status='running', ended_utc=None, error=None)
    write_manifest(target, manifest)
    return manifest, {**(overrides or {}), 'resume': True, 'device': device}


def epoch_logger(log: list, stop_after=None):
    """on_model_save callback: batch and GPU memory per epoch, optional stop once last.pt is written."""
    def record(trainer):
        import torch
        gpu = trainer.device.type == 'cuda'
        log.append({'epoch': trainer.epoch + 1, 'batch': trainer.batch_size, 'utc': now(),
                    'gpu_max_reserved_gib': round(torch.cuda.max_memory_reserved(trainer.device)/2**30, 3) if gpu else None})
        if stop_after and trainer.epoch + 1 >= stop_after and trainer.epoch + 1 < trainer.epochs:
            raise StopRequested(f'stopped after epoch {trainer.epoch + 1} of {trainer.epochs} as requested')
    return record


def read_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]


def epoch_seconds(rows, restarts) -> list[float]:
    # The time column counts from each start, so the first epoch after a resume counts from zero.
    times, seconds = [float(r['time']) for r in rows], []
    for k, row in enumerate(rows):
        first = k == 0 or int(float(row['epoch'])) - 1 in restarts
        seconds.append(round(times[k] if first else times[k] - times[k-1], 1))
    return seconds


def finish(target: Path, manifest, status, error=None) -> dict:
    rows = read_results(target / 'results.csv')
    seconds = epoch_seconds(rows, {r['after_epoch'] for r in manifest.get('resumes', [])})
    files = {p.relative_to(target).as_posix(): sha256(p) for p in sorted(target.rglob('*'))
             if p.is_file() and p.name != 'manifest.json'}
    manifest.update(status=status, ended_utc=now(), error=error, epochs_completed=len(rows), epoch_seconds=seconds,
                    best_sha256=files.get('weights/best.pt'), last_sha256=files.get('weights/last.pt'),
                    resumable=status != 'finished' and 'weights/last.pt' in files, files=files)
    write_manifest(target, manifest)
    return manifest


def fit(model, target: Path, manifest, kwargs, stop_after=None) -> dict:
    """Runs model.train; the manifest records the outcome, also when training stops or fails."""
    model.add_callback('on_model_save', epoch_logger(manifest.setdefault('epochs_log', []), stop_after))
    try:
        model.train(**kwargs)
    except BaseException as error:
        finish(target, manifest, 'stopped' if isinstance(error, StopRequested) else 'interrupted',
               f'{type(error).__name__}: {error}')
        raise
    return finish(target, manifest, 'finished')
