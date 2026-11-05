from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import unittest

from PIL import Image

from sports_analyzer.annotations.yolo import read_labels
from sports_analyzer.gsr import LABELS, audit_split, build_view, clip_key, match_clips, read_clip
from sports_analyzer.soccernet import NAMES

CATEGORIES = [dict(supercategory='object', id=1, name='player'), dict(supercategory='object', id=2, name='goalkeeper'),
              dict(supercategory='object', id=3, name='referee'), dict(supercategory='object', id=4, name='ball'),
              dict(supercategory='pitch', id=5, name='pitch'), dict(supercategory='camera', id=6, name='camera'),
              dict(supercategory='object', id=7, name='other')]
# frame, track, category, role, team, jersey, x, y, w, h, pitch
OBJECTS = [(1, 1, 1, 'player', 'left', '7', 10, 10, 20, 30, (1.5, -2.0)),
           (1, 2, 2, 'goalkeeper', 'right', None, 60, 20, 10, 25, (50.0, 3.0)),
           (1, 3, 3, 'referee', None, None, 30, 50, 8, 20, (0.0, 34.5)),
           (1, 4, 4, 'ball', None, None, 50, 40, 6, 6, (2.0, 1.0)),
           (1, 5, 7, 'other', None, None, 70, 60, 20, 10, None),
           (3, 1, 1, 'player', 'left', '7', -5, 70, 15, 20, (3.0, -2.0)),
           (3, 4, 4, 'ball', None, None, 120, 10, 6, 6, (2.5, 1.0)),
           (2, 1, 1, 'player', 'left', '7', 11, 11, 20, 30, (1.6, -2.0))]


def annotation(clip_id, frame, track, category, role, team, jersey, x, y, w, h, pitch):
    item = dict(id=f'{clip_id}{frame:03d}{track:03d}', image_id=f'{clip_id}{frame:06d}', track_id=track,
                supercategory='object', category_id=category,
                attributes=dict(role=role, jersey=jersey, team=team),
                bbox_image=dict(x=x, y=y, x_center=x + w/2, y_center=y + h/2, w=w, h=h),
                bbox_pitch=None)
    if pitch:
        item['bbox_pitch'] = dict(x_bottom_middle=pitch[0], y_bottom_middle=pitch[1])
    return item


def write_clip(root, name, game, objects=OBJECTS, length=4, width=100, height=80, unlabeled=()):
    clip, clip_id = root / name, '2' + name[-3:]
    (clip / 'img1').mkdir(parents=True)
    images = []
    for frame in range(1, length+1):
        Image.new('RGB', (width, height), 'green').save(clip / 'img1' / f'{frame:06d}.jpg')
        images.append(dict(is_labeled=frame not in unlabeled, image_id=f'{clip_id}{frame:06d}',
                           file_name=f'{frame:06d}.jpg', height=height, width=width,
                           has_labeled_person=frame not in unlabeled, has_labeled_pitch=True,
                           has_labeled_camera=True, ignore_regions_y=[], ignore_regions_x=[]))
    annotations = [annotation(clip_id, *o) for o in objects]
    annotations.append(dict(id=f'{clip_id}999', image_id=f'{clip_id}000001', supercategory='pitch',
                            category_id=5, lines={'Middle line': [dict(x=0.5, y=0.1), dict(x=0.5, y=0.9)]}))
    info = dict(version='1.3', game_id=str(game), id=name[-3:], action_class='Goal', game_time_start='1 - 16:09',
                clip_start='969000', clip_stop='999000', action_position='975293', name=name, im_dir='img1',
                frame_rate=25, seq_length=length, im_ext='.jpg')
    (clip / LABELS).write_text(json.dumps(dict(info=info, images=images, annotations=annotations,
                                               categories=CATEGORIES)), encoding='utf-8')
    return clip


class GameStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split = self.root / 'raw' / 'valid'
        write_clip(self.split, 'SNGS-021', 2)
        write_clip(self.split, 'SNGS-022', 3, unlabeled=(3,))
        self.view = self.root / 'view'

    def test_read_clip(self):
        clip = read_clip(self.split / 'SNGS-021')
        self.assertEqual((clip['clip'], clip['game'], clip['length'], len(clip['rows'])), ('SNGS-021', '2', 4, 8))
        row = clip['rows'][0]
        self.assertEqual((row['role'], row['category'], row['team'], row['jersey'], row['track']),
                         ('player', 'player', 'left', '7', 1))
        self.assertEqual((row['x'], row['w'], row['pitch_x'], row['pitch_y']), (10, 20, 1.5, -2.0))
        self.assertIsNone(clip['rows'][4]['pitch_x'])
        self.assertEqual(clip['pitch_images'], {'2021000001'})
        self.assertEqual([i['frame'] for i in clip['images']], [1, 2, 3, 4])

    def test_audit_counts_and_issues(self):
        summary = audit_split(self.split)['summary']
        self.assertEqual((summary['clips'], summary['frames'], summary['annotations']), (2, 8, 16))
        self.assertEqual(summary['games'], ['2', '3'])
        self.assertEqual(summary['resolutions'], {'100x80': 8})
        roles = summary['roles']
        self.assertEqual(roles['player']['annotations'], 6)
        self.assertEqual(roles['player']['team_left'], 6)
        self.assertEqual(roles['player']['jersey_share'], 1.0)
        self.assertEqual(roles['other']['pitch_share'], 0.0)
        self.assertEqual(roles['goalkeeper']['tracks'], 2)
        self.assertEqual(summary['issues'], {'missing_pitch': 2, 'off_pitch': 2, 'outside_image': 2,
                                             'partly_outside': 2})
        self.assertEqual(summary['errors'], 2)
        self.assertEqual(summary['counts']['images_is_labeled'], 7)

    def test_audit_flags_invalid_values(self):
        bad = [(1, 1, 1, 'player', 'blue', 'A', 10, 10, 0, 30, (1.0, 1.0)),
               (1, 2, 3, 'goalkeeper', None, None, 10, 10, 5, 5, (float('nan'), 1.0)),
               (2, 2, 3, 'referee', None, None, 10, 10, 5, 5, (1.0, 1.0)),
               (9, 3, 4, 'ball', None, None, 1, 1, 2, 2, (1.0, 1.0))]
        split = self.root / 'bad'
        write_clip(split, 'SNGS-030', 4, objects=bad)
        result = audit_split(split)
        issues = result['summary']['issues']
        for code in ('invalid_team', 'jersey_not_digit', 'non_positive_size', 'role_category_mismatch',
                     'non_finite_pitch', 'missing_team', 'unknown_image', 'track_role_changes'):
            self.assertIn(code, issues)
        self.assertEqual(result['clips'][0]['errors'], 5)

    def test_view_round_trip(self):
        manifest = build_view(self.split, self.view, 'valid', 2)
        self.assertEqual(manifest['counts']['frames'], 3)
        self.assertEqual(manifest['counts']['skipped_unlabeled_frames'], 1)
        boxes = read_labels(self.view / 'labels/valid/SNGS-021_000001.txt', 100, 80, NAMES)
        self.assertEqual([b.label for b in boxes], ['player', 'player', 'referee', 'ball'])
        for box, o in zip(boxes, OBJECTS[:4]):
            x, y, w, h = o[6:10]
            for got, expected in zip(box.xyxy, (x, y, x+w, y+h)):
                self.assertAlmostEqual(got, expected, places=6)
        self.assertTrue((self.view / 'images/valid/SNGS-021_000001.jpg').samefile(
            self.split / 'SNGS-021/img1/000001.jpg'))
        self.assertFalse((self.view / 'images/valid/SNGS-022_000003.jpg').exists())
        with (self.view / 'ignore.csv').open() as stream:
            ignores = list(csv.DictReader(stream))
        self.assertEqual(ignores[0], dict(image='images/valid/SNGS-021_000001.jpg', x1='70', y1='60',
                                          x2='90', y2='70', source='other'))
        with (self.view / 'tracks.csv').open() as stream:
            tracks = list(csv.DictReader(stream))
        self.assertEqual({k: tracks[0][k] for k in ('label', 'role', 'team', 'jersey', 'pitch_x', 'pitch_y')},
                         dict(label='player', role='player', team='left', jersey='7', pitch_x='1.5',
                              pitch_y='-2.0'))
        self.assertEqual((tracks[1]['role'], tracks[1]['label'], tracks[1]['team']), ('goalkeeper', 'player', 'right'))
        self.assertEqual(manifest['counts']['outside_ball'], 1)
        with self.assertRaises(FileExistsError):
            build_view(self.split, self.view, 'valid', 2)

    def test_clip_matching(self):
        info = read_clip(self.split / 'SNGS-021')['info']
        tracking = {'SNMOT-116': ('1 - 16:09', '969000', '999000', '975293', 'Goal'),
                    'SNMOT-117': ('1 - 17:30', '1050000', '1080000', '1074398', 'Corner')}
        self.assertEqual(match_clips({'SNGS-021': clip_key(info), 'SNGS-099': ('x',)}, tracking),
                         {'SNGS-021': ['SNMOT-116'], 'SNGS-099': []})


if __name__ == '__main__':
    unittest.main()
