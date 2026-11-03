from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import unittest

from PIL import Image

from sports_analyzer.annotations.yolo import read_labels
from sports_analyzer.soccernet import (IGNORE, NAMES, assert_disjoint_games, build_view,
                                       role_class, role_parts, split_games)

ROLES = {1: 'ball;1', 2: 'player team left;A', 3: 'goalkeepers team right;1',
         4: 'referee;main', 5: 'other;coach'}
GT = [(1, 1, 50, 40, 6, 6), (1, 2, 10, 10, 20, 30), (1, 3, 60, 20, 10, 25),
      (1, 4, 30, 50, 8, 20), (1, 5, 70, 60, 20, 10), (3, 2, -5, 70, 15, 20),
      (3, 1, 120, 10, 6, 6), (2, 2, 11, 11, 20, 30)]


def write_clip(root, name, game, length=4, width=100, height=80):
    clip = root / name
    (clip / 'img1').mkdir(parents=True)
    (clip / 'gt').mkdir()
    for frame in range(1, length+1):
        Image.new('RGB', (width, height), 'green').save(clip / 'img1' / f'{frame:06d}.jpg')
    tracks = ''.join(f'trackletID_{k}= {v}\n' for k, v in ROLES.items())
    (clip / 'gameinfo.ini').write_text(
        f'[Sequence]\nname={name}\nimDir=img1\nframeRate=25\nseqLength={length}\n'
        f'imWidth={width}\nimHeight={height}\nimExt=.jpg\ngameID={game}\nactionClass=Goal\n{tracks}')
    (clip / 'gt' / 'gt.txt').write_text(''.join(f'{f},{t},{x},{y},{w},{h},1,-1,-1,-1\n'
                                                for f, t, x, y, w, h in GT))
    return clip


class SoccerNetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split = self.root / 'raw' / 'test'
        write_clip(self.split, 'SNMOT-001', 7)
        write_clip(self.split, 'SNMOT-002', 8)
        self.view = self.root / 'view'

    def test_role_mapping(self):
        self.assertEqual([role_class(r) for r in ('ball', 'player team left', 'Goalkeepers team right',
                                                  'goalkeeper team left', 'referee', 'other')],
                         [0, 1, 1, 1, 2, IGNORE])
        with self.assertRaises(ValueError):
            role_class('coach')

    def test_exact_round_trip_and_sampling(self):
        manifest = build_view(self.split, self.view, 'test', 2)
        self.assertEqual(manifest['counts']['frames'], 4)
        self.assertEqual(manifest['games'], ['7', '8'])
        label = self.view / 'labels/test/SNMOT-001_000001.txt'
        boxes = read_labels(label, 100, 80, NAMES)
        self.assertEqual([b.label for b in boxes], ['ball', 'player', 'player', 'referee'])
        for box, (_, _, x, y, w, h) in zip(boxes, GT[:4]):
            for got, expected in zip(box.xyxy, (x, y, x+w, y+h)):
                self.assertAlmostEqual(got, expected, places=6)
        image = self.view / 'images/test/SNMOT-001_000001.jpg'
        source = self.split / 'SNMOT-001/img1/000001.jpg'
        self.assertTrue(image.samefile(source))
        self.assertFalse((self.view / 'labels/test/SNMOT-001_000002.txt').exists())
        with (self.view / 'frames.csv').open() as stream:
            frames = list(csv.DictReader(stream))
        self.assertEqual(frames[1], dict(image='images/test/SNMOT-001_000003.jpg', clip='SNMOT-001',
                                         game='7', frame='3'))

    def test_ignore_clipping_and_outside_are_reported(self):
        manifest = build_view(self.split, self.view, 'test', 2)
        with (self.view / 'ignore.csv').open() as stream:
            ignores = list(csv.DictReader(stream))
        self.assertEqual(len(ignores), 2)
        self.assertEqual(ignores[0], dict(image='images/test/SNMOT-001_000001.jpg',
                                          x1='70.0', y1='60.0', x2='90.0', y2='70.0'))
        counts = manifest['counts']
        self.assertEqual((counts['clipped_player'], counts['outside_ball']), (2, 2))
        self.assertEqual(counts['boxes_ignore'], 2)
        self.assertEqual(manifest['source_boxes_by_role']['other'], 2)
        boxes = read_labels(self.view / 'labels/test/SNMOT-001_000003.txt', 100, 80, NAMES)
        self.assertEqual([b.xyxy for b in boxes], [(0, 70, 10, 80)])
        self.assertEqual(json.loads((self.view / 'manifest.json').read_text())['step'], 2)
        with self.assertRaises(FileExistsError):
            build_view(self.split, self.view, 'test', 2)

    def test_tracks_keep_team_and_jersey(self):
        self.assertEqual([role_parts(r) for r in ('player team left', 'goalkeepers team right',
                                                  'goalkeeper team left', 'referee', 'ball', 'other')],
                         [('player', 'left'), ('goalkeeper', 'right'), ('goalkeeper', 'left'),
                          ('referee', 'none'), ('ball', 'none'), ('other', 'none')])
        manifest = build_view(self.split, self.view, 'test', 2)
        with (self.view / 'tracks.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), sum(manifest['counts'][f'boxes_{k}'] for k in [*NAMES.values(), IGNORE]))
        first = {r['track']: r for r in rows if r['image'].endswith('SNMOT-001_000001.jpg')}
        self.assertEqual({k: (v['label'], v['role'], v['team'], v['jersey']) for k, v in first.items()},
                         {'1': ('ball', 'ball', 'none', '1'), '2': ('player', 'player', 'left', 'A'),
                          '3': ('player', 'goalkeeper', 'right', '1'),
                          '4': ('referee', 'referee', 'none', 'main'), '5': (IGNORE, 'other', 'none', 'coach')})
        self.assertEqual((first['2']['x1'], first['2']['y2']), ('10.0', '40.0'))
        self.assertEqual((manifest['counts']['team_tracks'], manifest['counts']['team_tracks_numeric_jersey']),
                         (4, 2))
        self.assertEqual(manifest['action_classes'], {'Goal': 2})

    def test_unknown_track_and_game_overlap(self):
        self.assertEqual(split_games(self.split), {'SNMOT-001': '7', 'SNMOT-002': '8'})
        self.assertEqual(assert_disjoint_games({'train': ['4'], 'test': ['8', '7']}),
                         {'train': ['4'], 'test': ['7', '8']})
        with self.assertRaises(ValueError):
            assert_disjoint_games({'train': ['4', '7'], 'test': ['7', '8']})
        with (self.split / 'SNMOT-002/gt/gt.txt').open('a') as stream:
            stream.write('1,9,0,0,5,5,1,-1,-1,-1\n')
        with self.assertRaises(ValueError):
            build_view(self.split, self.view, 'test', 1)


if __name__ == '__main__':
    unittest.main()
