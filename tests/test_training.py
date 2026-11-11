from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from sports_analyzer.dataset_preparation import sha256
from sports_analyzer.training import StopRequested, fit, resume_run, start_run

CONFIG = {'sets_root': 'data/processed/training', 'set': 'tiny', 'model': 'models/yolo26n.pt',
          'models_manifest': 'models/MANIFEST.json', 'device': 'cpu', 'output': 'outputs/training',
          'train': {'imgsz': 1280, 'batch': 8, 'epochs': 2, 'seed': 0}}


def fake_base(root: Path, names=('ball', 'player', 'referee')):
    (root/'models').mkdir()
    (root/'models/yolo26n.pt').write_bytes(b'weights')
    (root/'models/MANIFEST.json').write_text(json.dumps({'files': [
        {'file': 'yolo26n.pt', 'sha256': sha256(root/'models/yolo26n.pt'), 'source_url': 'local'}]}))
    folder = root/'data/processed/training/tiny'
    folder.mkdir(parents=True)
    for name in ('images.csv', 'dropped.csv', 'data.yaml'):
        (folder/name).write_text(name)
    (folder/'manifest.json').write_text(json.dumps({
        'name': 'tiny', 'names': dict(enumerate(names)), 'images': {'train': 3, 'val': 1},
        'images_by_source': {'train/football': 3, 'val/football': 1}, 'boxes_by_class': {'train/ball': 3},
        'run': 'outputs/training_set/x', 'files': {n: sha256(folder/n) for n in ('images.csv', 'dropped.csv', 'data.yaml')}}))
    return folder


class FakeModel:
    """Writes what Ultralytics writes; the time column restarts on resume."""
    def __init__(self, times, interrupt_at=None):
        self.times, self.interrupt_at, self.callbacks, self.calls = times, interrupt_at, [], []

    def add_callback(self, event, callback):
        self.callbacks.append(callback)

    def train(self, **kwargs):
        self.calls.append(kwargs)
        target = Path(kwargs['project'])/kwargs['name'] if 'project' in kwargs else self.target
        self.target = target
        (target/'weights').mkdir(exist_ok=True)
        csv = target/'results.csv'
        done = len(csv.read_text().splitlines()) - 1 if csv.is_file() else 0
        if not done:
            csv.write_text('epoch,time,metrics/recall(B)\n')
        for epoch in range(done, len(self.times)):
            if epoch == self.interrupt_at:
                raise KeyboardInterrupt
            with csv.open('a') as stream:
                stream.write(f'{epoch+1},{self.times[epoch]},0.5\n')
            for name in ('best.pt', 'last.pt'):
                (target/'weights'/name).write_bytes(f'epoch {epoch}'.encode())
            trainer = SimpleNamespace(epoch=epoch, epochs=len(self.times), batch_size=8, device=SimpleNamespace(type='cpu'))
            for callback in self.callbacks:
                callback(trainer)
        (target/'results.png').write_bytes(b'png')


class TrainingTests(unittest.TestCase):
    def test_manifest_of_a_finished_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fake_base(base)
            target, manifest, kwargs = start_run(CONFIG, 'tiny', base, 'cpu', {'epochs': 2}, {'note': 'test'})
            written = json.loads((target/'manifest.json').read_text())
            self.assertEqual((written['status'], written['ended_utc']), ('running', None))
            self.assertEqual(written['set']['images'], {'train': 3, 'val': 1})
            self.assertEqual(written['set']['manifest_sha256'], sha256(base/'data/processed/training/tiny/manifest.json'))
            self.assertEqual(written['base_model']['sha256'], sha256(base/'models/yolo26n.pt'))
            self.assertEqual(written['train_args']['data'], 'data/processed/training/tiny/data.yaml')
            for key in ('torch', 'ultralytics', 'python', 'git_commit', 'git_dirty', 'device', 'started_utc', 'code_sha256'):
                self.assertIn(key, written)
            self.assertEqual((kwargs['name'], kwargs['exist_ok'], kwargs['imgsz']), (target.name, True, 1280))
            manifest = fit(FakeModel([10.0, 25.5]), target, manifest, kwargs)
            written = json.loads((target/'manifest.json').read_text())
            self.assertEqual(written, json.loads(json.dumps(manifest)))
            self.assertEqual((written['status'], written['epochs_completed'], written['resumable']), ('finished', 2, False))
            self.assertEqual(written['epoch_seconds'], [10.0, 15.5])
            self.assertEqual(written['best_sha256'], sha256(target/'weights/best.pt'))
            self.assertEqual(sorted(written['files']), ['results.csv', 'results.png', 'weights/best.pt', 'weights/last.pt'])
            self.assertEqual([e['epoch'] for e in written['epochs_log']], [1, 2])
            self.assertIsNotNone(written['ended_utc'])

    def test_stop_interrupt_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fake_base(base)
            target, manifest, kwargs = start_run(CONFIG, 'tiny', base, 'cpu')
            model = FakeModel([10.0, 20.0, 30.0])
            with self.assertRaises(StopRequested):
                fit(model, target, manifest, kwargs, stop_after=1)
            written = json.loads((target/'manifest.json').read_text())
            self.assertEqual((written['status'], written['epochs_completed'], written['resumable']), ('stopped', 1, True))
            manifest, kwargs = resume_run(target, base, 'cpu', {'batch': 4})
            self.assertEqual(kwargs, {'batch': 4, 'resume': True, 'device': 'cpu'})
            self.assertEqual(json.loads((target/'manifest.json').read_text())['status'], 'running')
            model = FakeModel([10.0, 12.0, 30.0], interrupt_at=2)
            model.target = target
            with self.assertRaises(KeyboardInterrupt):
                fit(model, target, manifest, kwargs)
            written = json.loads((target/'manifest.json').read_text())
            self.assertEqual((written['status'], written['epochs_completed']), ('interrupted', 2))
            self.assertEqual(written['error'], 'KeyboardInterrupt: ')
            self.assertEqual(written['epoch_seconds'], [10.0, 12.0])
            self.assertEqual([r['after_epoch'] for r in written['resumes']], [1])
            manifest, kwargs = resume_run(target, base, 'cpu')
            model = FakeModel([10.0, 12.0, 30.0])
            model.target = target
            written = fit(model, target, manifest, kwargs)
            self.assertEqual((written['status'], written['epoch_seconds']), ('finished', [10.0, 12.0, 30.0]))
            self.assertEqual([e['epoch'] for e in written['epochs_log']], [1, 2, 3])
            with self.assertRaises(ValueError):
                resume_run(target, base, 'cpu')

    def test_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            folder = fake_base(base)
            (folder/'data.yaml').write_text('changed')
            with self.assertRaises(ValueError):
                start_run(CONFIG, 'tiny', base, 'cpu')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fake_base(base, names=('person', 'sports ball', 'referee'))
            with self.assertRaises(ValueError):
                start_run(CONFIG, 'tiny', base, 'cpu')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fake_base(base)
            (base/'models/yolo26n.pt').write_bytes(b'other')
            with self.assertRaises(ValueError):
                start_run(CONFIG, 'tiny', base, 'cpu')
            self.assertFalse((base/'outputs').exists())


if __name__ == '__main__':
    unittest.main()
