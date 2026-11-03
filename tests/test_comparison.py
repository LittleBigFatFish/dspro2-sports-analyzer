import json
import tempfile
import unittest
from pathlib import Path
from sports_analyzer.annotations.yolo import Box
from sports_analyzer.comparison import (source_group, yolo_view, filter_by_audit, sample_view, build_view, group_counts,
                                        view_digest, check_against_audit, serialise, load_predictions,
                                        summarise_timings)
from sports_analyzer.inference import Detector

class ComparisonTests(unittest.TestCase):
    def test_source_group(self):
        self.assertEqual(source_group('images/val/20250710_144746_00024d.jpg'), '20250710_144746')
        self.assertEqual(source_group('images/val/NED_VS_FRA_mp4-17.jpg'), 'NED_VS_FRA')
        self.assertEqual(source_group('images/train/clip_-mp4-3.jpg'), 'clip')
        self.assertEqual(source_group('images/test/spt_v2_10.jpg'), 'spt_v2')
        self.assertEqual(source_group('images/train/other.jpg'), 'other')

    def test_view_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train/a_mp4-1.jpg', 'train/a_mp4-2.JPG', 'train/notes.txt', 'val/b_mp4-1.png'):
                (root/'images'/name).parent.mkdir(parents=True, exist_ok=True)
                (root/'images'/name).write_bytes(b'x')
            view = yolo_view(root, ['train', 'val'])
            self.assertEqual([r['image'] for r in view],
                             ['images/train/a_mp4-1.jpg', 'images/train/a_mp4-2.JPG', 'images/val/b_mp4-1.png'])
            self.assertEqual(group_counts(view), [{'split': 'train', 'source': 'a', 'images': 2},
                                                  {'split': 'val', 'source': 'b', 'images': 1}])
            with self.assertRaises(FileNotFoundError):
                yolo_view(root, ['test'])
            (root/'frames.csv').write_text('image,clip,game,frame\nimages/val/b_mp4-1.png,SNMOT-1,7,25\n')
            self.assertEqual(yolo_view(root, ['val']), [{'image': 'images/val/b_mp4-1.png', 'split': 'val',
                                                         'game': '7', 'source': 'SNMOT-1', 'frame': 25}])
            self.assertEqual(group_counts(yolo_view(root, ['val'])),
                             [{'split': 'val', 'game': '7', 'source': 'SNMOT-1', 'images': 1}])
            with self.assertRaises(ValueError):
                yolo_view(root, ['train'])

    def test_audit_filter_and_sample(self):
        view = [{'image': f'images/test/{n}.jpg'} for n in ('a', 'b', 'c', 'd')]
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory)
            (audit/'images.csv').write_text('image,crop\nimages/test/a.jpg,False\nimages/test/b.jpg,True\n'
                                            'images/test/c.jpg,False\n')
            (audit/'exclusions.txt').write_text('images/test/c.jpg\tleak_byte_identical\n')
            self.assertEqual([r['image'] for r in filter_by_audit(view, audit, True, True)], ['images/test/a.jpg'])
            self.assertEqual(len(filter_by_audit(view, audit, False, True)), 3)
            for name in ('v7_1.jpg', 'v3_1.jpg', 'v3_2.jpg'):
                (audit/'images'/'train').mkdir(parents=True, exist_ok=True)
                (audit/'images'/'train'/name).write_bytes(b'x')
            (audit/'overlap.txt').write_text('images/train/v3_2.jpg\tleak\n')
            settings = {'splits': ['train'], 'extra_exclusions': ['overlap.txt'], 'exclude_groups': ['v7']}
            self.assertEqual([r['image'] for r in build_view(audit, settings, base=audit)], ['images/train/v3_1.jpg'])
            with self.assertRaises(ValueError):
                build_view(audit, {'splits': ['train'], 'full_frames_only': True})
        sample = sample_view(view, 2, 42)
        self.assertEqual(sample, sample_view(view, 2, 42))
        self.assertEqual(sample, sorted(sample, key=lambda r: r['image']))
        self.assertEqual(sample_view(view, 10, 42), view)

    def test_digest_and_audit_check(self):
        view = [{'image': 'images/val/a.jpg', 'image_sha256': '1'}, {'image': 'images/val/b.jpg', 'image_sha256': '2'}]
        self.assertEqual(view_digest(view), view_digest(view[::-1]))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'images.csv').write_text('image,image_sha256\nimages/val/a.jpg,1\nimages/val/b.jpg,3\n')
            with self.assertRaises(ValueError):
                check_against_audit(view, Path(directory))
            check_against_audit(view[:1], Path(directory))

    def test_prediction_round_trip(self):
        record = {'image': 'images/val/a.jpg', 'split': 'val', 'source': 'a'}
        boxes = [Box('person', (1.23456, 2.0, 30.0, 40.0), .87654321), Box('sports ball', (5, 5, 9, 9), .001)]
        images = [serialise(record, boxes, {'inference': 12.34567})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'predictions.json'
            path.write_text(json.dumps({'images': images}))
            loaded = load_predictions(path)['images/val/a.jpg']
        self.assertEqual([b.label for b in loaded], ['person', 'sports ball'])
        self.assertEqual(loaded[0].xyxy, (1.235, 2.0, 30.0, 40.0))
        self.assertEqual(loaded[0].score, .876543)
        self.assertEqual(images[0]['speed_ms'], {'inference': 12.346})

    def test_timing_summary(self):
        summary = summarise_timings([float(v) for v in range(1, 11)])
        self.assertEqual((summary['n'], summary['median_ms']), (10, 5.5))
        self.assertAlmostEqual(summary['p90_ms'], 9.1)

    def test_detector_requires_local_weights(self):
        with self.assertRaises(FileNotFoundError):
            Detector(Path('models/missing.pt'), device=0, nms=False)

if __name__ == '__main__':
    unittest.main()
