from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import os
import pickle
import unittest

import cv2
import numpy as np

from sports_analyzer.annotations.yolo import read_labels
from sports_analyzer.soccernet import NAMES
from sports_analyzer.sportsaction import (audit_split, build_view, load_annotations, shared_sources, source_video,
                                          split_videos)

LABELS = ['aerobic push up', 'football shoot', 'football trap']


def tube(start, stop, box):
    return np.array([[f, *box] for f in range(start, stop+1)], np.float32)


def annotations():
    return dict(labels=LABELS,
                train_videos=[['football/v_aaa_c001', 'football/v_bbb_c002', 'aerobic_gymnastics/v_x_c1']],
                test_videos=[['football/v_aaa_c003']],
                nframes={'football/v_aaa_c001': 6, 'football/v_bbb_c002': 4, 'football/v_aaa_c003': 4,
                         'aerobic_gymnastics/v_x_c1': 4},
                resolution={'football/v_aaa_c001': (48, 64), 'football/v_bbb_c002': (48, 64),
                            'football/v_aaa_c003': (48, 64), 'aerobic_gymnastics/v_x_c1': (48, 64)},
                gttubes={'football/v_aaa_c001': {1: [tube(1, 4, (10, 10, 20, 30))], 2: [tube(3, 6, (50, 20, 70, 40))]},
                         'football/v_bbb_c002': {2: [tube(2, 3, (5, 5, 5, 20))]},
                         'football/v_aaa_c003': {1: [tube(1, 2, (1, 1, 9, 9))]},
                         'aerobic_gymnastics/v_x_c1': {0: [tube(1, 2, (1, 1, 9, 9))]}})


def write_video(path, frames, width=64, height=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    for number in range(frames):
        writer.write(np.full((height, width, 3), number*20, np.uint8))
    writer.release()


class SportsActionTests(unittest.TestCase):
    def test_loader_refuses_other_globals(self):
        with TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / 'good.pkl', Path(tmp) / 'bad.pkl'
            good.write_bytes(pickle.dumps(annotations()))
            bad.write_bytes(pickle.dumps(os.getcwd))
            self.assertEqual(load_annotations(good)['labels'], LABELS)
            with self.assertRaises(pickle.UnpicklingError):
                load_annotations(bad)

    def test_splits_and_sources(self):
        gt = annotations()
        self.assertEqual(split_videos(gt, 'train'), ['football/v_aaa_c001', 'football/v_bbb_c002'])
        self.assertEqual(split_videos(gt, 'val'), ['football/v_aaa_c003'])
        self.assertEqual(source_video('football/v_5uBLdpt5-oU_c602'), 'v_5uBLdpt5-oU')
        self.assertEqual(shared_sources(gt), ['v_aaa'])

    def test_audit(self):
        summary = audit_split(annotations(), 'train')['summary']
        self.assertEqual(summary['videos'], 2)
        self.assertEqual(summary['counts'], dict(frames=10, labelled_frames=8, tubes=3, boxes=10))
        self.assertEqual(summary['issues'], {'partly_outside': 4, 'non_positive_size': 2})
        self.assertEqual(summary['errors'], 2)
        self.assertEqual(summary['actions'], {'football trap': 2, 'football shoot': 1})

    def test_view(self):
        with TemporaryDirectory() as tmp:
            videos = Path(tmp) / 'videos'
            write_video(videos / 'football/v_aaa_c001.mp4', 6)
            write_video(videos / 'football/v_bbb_c002.mp4', 4)
            manifest = build_view(annotations(), videos, Path(tmp) / 'view', 'train', 2)
            view = Path(tmp) / 'view'
            # Step 2 keeps frames 1, 3, 5; v_bbb has boxes only on 2 and 3, and its box has no width.
            with (view / 'frames.csv').open(encoding='utf-8') as stream:
                frames = [(r['clip'], r['frame'], r['game']) for r in csv.DictReader(stream)]
            self.assertEqual(frames, [('v_aaa_c001', '1', 'v_aaa'), ('v_aaa_c001', '3', 'v_aaa'),
                                      ('v_aaa_c001', '5', 'v_aaa'), ('v_bbb_c002', '3', 'v_bbb')])
            boxes = read_labels(view / 'labels/train/v_aaa_c001_000003.txt', 64, 48, NAMES)
            self.assertEqual([b.label for b in boxes], ['player', 'player'])
            self.assertEqual(manifest['counts']['dropped_boxes'], 1)
            self.assertEqual(manifest['counts']['frame_count_mismatch'], 0)
            self.assertEqual((view / 'labels/train/v_bbb_c002_000003.txt').read_text(), '')
            self.assertTrue((view / 'images/train/v_aaa_c001_000005.jpg').is_file())


if __name__ == '__main__':
    unittest.main()
