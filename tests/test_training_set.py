from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import csv
import json
import random
import unittest

from PIL import Image

from sports_analyzer.dataset_preparation import audit_dataset, save_audit
from sports_analyzer.training_set import (build_set, class_map, prepare, remap_lines, rotated, target_size,
                                          verify_set)

ROLES = {1: 'ball;1', 2: 'player team left;A', 3: 'goalkeepers team right;1', 4: 'referee;main', 5: 'other;coach'}


def pattern(size, seed):
    # Smooth random blobs give every image its own dHash.
    rng = random.Random(seed)
    small = Image.frombytes('RGB', (8, 6), bytes(rng.randrange(256) for _ in range(8*6*3)))
    return small.resize(size, Image.Resampling.BICUBIC)


def write_yolo(root, names, files):
    root.mkdir(parents=True)
    (root / 'data.yaml').write_text(f'names: {names}\n')
    for relative, (size, seed, label) in files.items():
        for kind in ('images', 'labels'):
            (root / kind / Path(relative).parent).mkdir(parents=True, exist_ok=True)
        pattern(size, seed).save(root / 'images' / relative, quality=95)
        (root / 'labels' / relative).with_suffix('.txt').write_text(label)


def write_clip(root, name, game, gt, seed, length=10):
    clip = root / name
    (clip / 'img1').mkdir(parents=True)
    (clip / 'gt').mkdir()
    for frame in range(1, length+1):
        pattern((64, 36), seed+frame).save(clip / 'img1' / f'{frame:06d}.jpg', quality=95)
    tracks = ''.join(f'trackletID_{k}= {v}\n' for k, v in ROLES.items())
    (clip / 'gameinfo.ini').write_text(
        f'[Sequence]\nname={name}\nimDir=img1\nframeRate=25\nseqLength={length}\n'
        f'imWidth=64\nimHeight=36\nimExt=.jpg\ngameID={game}\nactionClass=Goal\n{tracks}')
    (clip / 'gt' / 'gt.txt').write_text(''.join(f'{f},{t},{x},{y},{w},{h},1,-1,-1,-1\n' for f, t, x, y, w, h in gt))


class TrainingSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = base = Path(self.temp.name)
        football, soccana = base / 'football', base / 'soccana'
        write_yolo(football, '[ball, player, referee]', {
            'train/20250710_144746_aa.jpg': ((160, 90), 1, '0 0.5 0.5 0.05 0.1\n1 0.2 0.5 0.1 0.4\n'),
            'train/20250710_144746_bb.jpg': ((160, 90), 2, '2 0.5 0.5 0.1 0.4\n'),
            # Copy of a validation frame and of a report frame.
            'train/20250710_145737_cc.jpg': ((160, 90), 3, '1 0.5 0.5 0.1 0.4\n'),
            'train/20250710_145737_dd.jpg': ((160, 90), 9, '1 0.5 0.5 0.1 0.4\n'),
            'train/20250812_101950_ee.jpg': ((160, 90), 3, '1 0.3 0.5 0.1 0.4\n'),
            'val/report_1.jpg': ((160, 90), 9, '1 0.5 0.5 0.1 0.4\n')})
        write_yolo(soccana, '{0: Player, 1: Ball, 2: Referee}', {
            'train/v3_1.jpg': ((90, 90), 20, '1 0.5 0.5 0.05625 0.1\n0 0.25 0.5 0.1 0.3\n'),
            'train/v3_2.jpg': ((160, 90), 21, '1 0.4 0.4 0.05 0.1\n'),
            # Rotated by 90 degrees: players lie on their side.
            'train/v3_3.jpg': ((90, 90), 23, '0 0.5 0.3 0.3 0.05\n0 0.5 0.6 0.25 0.05\n1 0.2 0.2 0.1 0.06\n'),
            'train/v7_1.jpg': ((160, 90), 22, '0 0.5 0.5 0.1 0.3\n')})
        audits = {}
        for name, root, names, splits in (('football', football, ['ball', 'player', 'referee'], ('train', 'val')),
                                          ('soccana', soccana, ['Player', 'Ball', 'Referee'], ('train',))):
            result = audit_dataset(root, root / 'data.yaml', names, splits=splits)
            audits[name] = save_audit(result, base / 'audits' / name).relative_to(base).as_posix()
        tracking = base / 'tracking' / 'train'
        write_clip(tracking, 'SNMOT-001', 4, [(1, 1, 10, 10, 4, 4), (1, 2, 20, 5, 6, 20), (1, 5, 40, 5, 6, 20),
                                              (6, 2, 20, 5, 6, 20), (6, 3, 50, 5, 20, 20)], 40)
        write_clip(tracking, 'SNMOT-002', 9, [(1, 1, 30, 10, 4, 4), (1, 4, 5, 5, 6, 20)], 60)
        self.config = dict(
            output='processed', near_duplicate_distance=4, jpeg_quality=95,
            sources=dict(
                football=dict(kind='yolo', root='football', split='train', audit_run=audits['football'],
                              classes={0: 'ball', 1: 'player', 2: 'referee'}, val_groups=['20250812_101950'],
                              licence='CC BY 4.0', credit='football'),
                soccana=dict(kind='yolo', root='soccana', split='train', audit_run=audits['soccana'],
                             full_frames_only=True, apply_exclusions=True, exclude_groups=['v7'],
                             classes={0: 'player', 1: 'ball', 2: 'referee'}, squashed_frames='restore',
                             drop_rotated=True, rotated_width_height=3.0,
                             licence='upstream', credit='soccana'),
                soccernet=dict(kind='soccernet', root='tracking/train', drop_roles=['other'],
                               classes={'ball': 'ball', 'player': 'player', 'goalkeeper': 'player',
                                        'referee': 'referee'},
                               train_games=[4], train_step=5, val_games=[9], val_step=5,
                               licence='research', credit='soccernet')),
            report_sets=dict(football_val=dict(audit_run=audits['football'], split='val')),
            report_folders=['tracking/test'], report_games=[7, 8, 11],
            sets=dict(football_only=['football'], combined=['football', 'soccernet', 'soccana']))

    def build(self, name, config=None):
        config = config or self.config
        prepared = prepare(config, self.base)
        output = self.base / 'processed' / name
        manifest = build_set(name, config, prepared, output)
        with (output / 'images.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        return prepared, manifest, rows, output

    def test_class_remapping_with_soccana_order(self):
        mapping = class_map({0: 'player', 1: 'ball', 2: 'referee'}, {0: 'Player', 1: 'Ball', 2: 'Referee'})
        self.assertEqual(mapping, {'0': 1, '1': 0, '2': 2})
        self.assertEqual(remap_lines('1 0.5 0.4 0.05 0.1\n\n0 0.2 0.3 0.1 0.3\n', mapping, 'x'),
                         ['0 0.5 0.4 0.05 0.1', '1 0.2 0.3 0.1 0.3'])
        _, _, rows, output = self.build('combined')
        soccana = next(r for r in rows if r['source_file'] == 'images/train/v3_2.jpg')
        self.assertEqual((output / 'labels/train/soccana_v3_2.txt').read_text(), '0 0.4 0.4 0.05 0.1\n')
        self.assertEqual(soccana['label_transform'], 'class ids remapped')
        keeper = (output / 'labels/train/soccernet_SNMOT-001_000006.txt').read_text().split('\n')
        self.assertEqual([line.split()[0] for line in keeper if line], ['1', '1'])

    def test_unmapped_class_and_unknown_source_fail(self):
        with self.assertRaises(ValueError):
            remap_lines('3 0.5 0.5 0.1 0.1\n', {'0': 0, '1': 1, '2': 2}, 'x')
        with self.assertRaises(ValueError):
            class_map({0: 'ball', 1: 'player'}, {0: 'ball', 1: 'player', 2: 'referee'})
        with self.assertRaises(ValueError):
            class_map({0: 'player', 1: 'ball', 2: 'referee'}, {0: 'ball', 1: 'player', 2: 'referee'})
        with self.assertRaises(ValueError):
            class_map({'ball': 'goal'})
        config = copy.deepcopy(self.config)
        del config['sources']['soccernet']['classes']['goalkeeper']
        with self.assertRaises(ValueError):
            prepare(config, self.base)
        config = copy.deepcopy(self.config)
        config['sources']['soccernet']['kind'] = 'coco'
        with self.assertRaises(ValueError):
            prepare(config, self.base)
        config = copy.deepcopy(self.config)
        config['sets']['broken'] = ['football', 'skillcorner']
        with self.assertRaises(ValueError):
            build_set('broken', config, prepare(config, self.base), self.base / 'processed' / 'broken')

    def test_aspect_restore_keeps_labels_and_widens_image(self):
        self.assertEqual(target_size(1080, 1080, 'restore'), (1920, 1080))
        self.assertEqual(target_size(1280, 720, 'restore'), (1280, 720))
        self.assertIsNone(target_size(640, 640, 'drop'))
        with self.assertRaises(ValueError):
            target_size(800, 600, 'restore')
        _, manifest, rows, output = self.build('combined')
        squashed = next(r for r in rows if r['source_file'] == 'images/train/v3_1.jpg')
        self.assertEqual(squashed['image_transform'], 'resized 90x90 to 160x90')
        with Image.open(output / squashed['image']) as image:
            self.assertEqual(image.size, (160, 90))
        self.assertEqual((output / 'labels/train/soccana_v3_1.txt').read_text(),
                         '0 0.5 0.5 0.05625 0.1\n1 0.25 0.5 0.1 0.3\n')
        wide = next(r for r in rows if r['source_file'] == 'images/train/v3_2.jpg')
        self.assertTrue((output / wide['image']).samefile(self.base / 'soccana/images/train/v3_2.jpg'))
        ratio = manifest['ball_width_height']['soccana/resized']
        self.assertEqual((ratio['median_before'], ratio['median_after']), (0.5625, 1.0))
        config = copy.deepcopy(self.config)
        config['sources']['soccana']['squashed_frames'] = 'drop'
        prepared, _, rows, _ = self.build('combined_drop', {**config, 'sets': {'combined_drop': ['soccana']}})
        self.assertEqual([r['source_file'] for r in rows if r['dataset'] == 'soccana'], ['images/train/v3_2.jpg'])
        self.assertIn(dict(dataset='soccana', split='train', source_file='images/train/v3_1.jpg', reason='squashed'),
                      prepared['dropped'])

    def test_rotated_frames_are_dropped(self):
        # Upright players in a squashed square frame look 0.56 times as wide as they are.
        self.assertFalse(rotated(['1 0.5 0.5 0.0225 0.1', '1 0.3 0.5 0.03 0.12'], 640, 640, 3.0))
        self.assertTrue(rotated(['1 0.5 0.5 0.3 0.05', '2 0.3 0.5 0.2 0.05'], 640, 640, 3.0))
        self.assertTrue(rotated(['1 0.5 0.5 0.2 0.05'], 1920, 1080, 3.0))
        self.assertFalse(rotated(['0 0.5 0.5 0.3 0.01'], 640, 640, 3.0))
        self.assertFalse(rotated([], 640, 640, 3.0))
        prepared, manifest, rows, _ = self.build('combined')
        self.assertEqual(sorted(r['source_file'] for r in rows if r['dataset'] == 'soccana'),
                         ['images/train/v3_1.jpg', 'images/train/v3_2.jpg'])
        self.assertIn(dict(dataset='soccana', split='train', source_file='images/train/v3_3.jpg', reason='rotated'),
                      prepared['dropped'])
        self.assertEqual(manifest['dropped']['train/soccana/rotated'], 1)

    def test_held_out_sources_and_near_duplicates_stay_out_of_train(self):
        prepared, manifest, rows, _ = self.build('football_only')
        train = {r['source_file'] for r in rows if r['split'] == 'train'}
        val = {r['group'] for r in rows if r['split'] == 'val'}
        self.assertEqual(train, {'images/train/20250710_144746_aa.jpg', 'images/train/20250710_144746_bb.jpg'})
        self.assertEqual(val, {'20250812_101950', 'SNMOT-002'})
        reasons = {d['source_file']: d['reason'] for d in prepared['dropped']}
        self.assertEqual(reasons['images/train/20250710_145737_cc.jpg'], 'near_validation')
        self.assertEqual(reasons['images/train/20250710_145737_dd.jpg'], 'near_football_val')
        self.assertEqual(manifest['dropped'], {'train/football/near_football_val': 1,
                                               'train/football/near_validation': 1})

    def test_report_sets_are_refused(self):
        for source, key, value in (('football', 'split', 'val'), ('soccernet', 'val_games', [7]),
                                   ('soccernet', 'root', 'tracking/test')):
            config = copy.deepcopy(self.config)
            config['sources'][source][key] = value
            (self.base / 'tracking' / 'test').mkdir(parents=True, exist_ok=True)
            with self.assertRaises(ValueError):
                prepare(config, self.base)
        config = copy.deepcopy(self.config)
        config['sources']['soccernet']['train_games'] = [4, 9]
        with self.assertRaises(ValueError):
            prepare(config, self.base)

    def test_frames_with_other_are_dropped(self):
        prepared, manifest, rows, output = self.build('combined')
        frames = {r['source_file'] for r in rows if r['dataset'] == 'soccernet'}
        self.assertEqual(frames, {'SNMOT-001/img1/000006.jpg', 'SNMOT-002/img1/000001.jpg',
                                  'SNMOT-002/img1/000006.jpg'})
        self.assertEqual(manifest['dropped']['train/soccernet/other_role'], 1)
        self.assertEqual(manifest['images_by_source'], {'train/football': 2, 'train/soccana': 2, 'train/soccernet': 1,
                                                        'val/football': 1, 'val/soccernet': 2})
        self.assertEqual((output / 'labels/val/soccernet_SNMOT-002_000006.txt').read_text(), '')

    def test_manifest_counts_match_files(self):
        prepared, manifest, rows, output = self.build('combined')
        checks = verify_set(output, prepared['report_sha256'])
        self.assertEqual(checks['images'], manifest['images'])
        self.assertEqual(checks['images'], {'train': 5, 'val': 3})
        self.assertEqual(sum(checks['images'].values()), len(rows))
        self.assertEqual(checks['boxes'], manifest['boxes_by_class'])
        saved = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(saved['credits']['soccana'], {'licence': 'upstream', 'credit': 'soccana'})
        with self.assertRaises(FileExistsError):
            build_set('combined', self.config, prepared, output)
        with self.assertRaises(ValueError):
            verify_set(output, {rows[0]['source_sha256']})
        (output / 'labels/train/soccana_v3_2.txt').write_text('5 0.5 0.5 0.1 0.1\n')
        with self.assertRaises(ValueError):
            verify_set(output)


if __name__ == '__main__':
    unittest.main()
