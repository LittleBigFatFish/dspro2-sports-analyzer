from pathlib import Path
import importlib.util
import json
import random
import tempfile
import unittest

import numpy as np
import yaml

from sports_analyzer.annotations.yolo import Box
from sports_analyzer.comparison import view_digest
from sports_analyzer.dataset_preparation import sha256
from sports_analyzer.evaluation import PRED_MAPPING, centre_match, evaluate_image, average_precision, remap, wilson

spec = importlib.util.spec_from_file_location('score_comparison', Path(__file__).resolve().parents[1] / 'scripts/score_comparison.py')
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)

SETTINGS = {'report_confidence': .25, 'match_iou': .5, 'ap_iou': [.5, .75], 'grid_iou': [.5],
            'grid_confidence': [.25], 'ball_size_edges': [8, 16], 'ball_centre_margin': 4}
ROOT = Path(__file__).resolve().parents[1]


class ScoringTests(unittest.TestCase):
    def test_ignore_drops_only_unmatched_predictions(self):
        gt = [Box('ball', (0, 0, 10, 10))]
        ignore = [(0, 0, 10, 10), (100, 100, 200, 200)]
        pred = [Box('ball', (0, 0, 10, 10), .9), Box('ball', (0, 0, 10, 10), .8),
                Box('person', (100, 100, 200, 200), .7), Box('person', (100, 100, 150, 200), .6),
                Box('ball', (300, 300, 310, 310), .5)]
        rows, events = evaluate_image(gt, pred, .5, ignore)
        person, ball = rows
        # The matched box stays a TP, only the duplicate is ignored.
        self.assertEqual((ball['tp'], ball['fp'], ball['fn']), (1, 1, 0))
        # Ignore regions apply to every class.
        self.assertEqual((person['tp'], person['fp']), (0, 0))
        self.assertEqual([e['status'] for e in events], ['ignored', 'ignored', 'TP', 'ignored', 'FP'])
        self.assertEqual(evaluate_image(gt, pred, .5)[0][1]['fp'], 2)
        truth, predictions = {'a': gt}, {'a': [pred[1], pred[0]]}
        self.assertEqual(average_precision(truth, predictions, 'ball', .5, {'a': ignore}), 1)
        far_first = {'a': [Box('ball', (300, 300, 310, 310), .95), pred[0]]}
        self.assertLess(average_precision(truth, far_first, 'ball', .5, {'a': ignore}), 1)

    def test_wilson(self):
        low, high = wilson(5, 10)
        self.assertAlmostEqual(low, .236593, places=5)
        self.assertAlmostEqual(high, .763407, places=5)
        self.assertEqual(wilson(0, 10)[0], 0)
        self.assertAlmostEqual(wilson(0, 10)[1], .277533, places=5)
        self.assertEqual(wilson(0, 0), (None, None))

    def test_ball_size_bins(self):
        box = Box('ball', (0, 0, 20, 30))
        self.assertEqual(scoring.short_side(box, (1920, 1080), 640), 20/3)
        self.assertEqual([scoring.size_bin(s, [8, 16]) for s in (7.9, 8, 16, 16.1)], ['<8', '8-16', '8-16', '>16'])

    def test_one_ranking_serves_every_confidence(self):
        rng = random.Random(0)
        truth, predictions = {}, {}
        for name in 'abcdef':
            truth[name] = [Box(rng.choice(['player', 'referee', 'ball']), (x, 0, x+rng.uniform(5, 40), 30))
                           for x in range(0, 300, 50)]
            predictions[name] = [Box(rng.choice(['person', 'ball']), (x+rng.uniform(-4, 4), rng.uniform(-3, 3),
                                     x+rng.uniform(10, 40), 30), round(rng.random(), 2)) for x in range(0, 300, 25)]
        sizes = {name: (640, 360) for name in truth}
        scored = scoring.score_images(truth, predictions, sizes, {}, 640, [.5, .75], .5, [8, 16], 4)
        rows = {r['metric']: r for r in scoring.group_metrics(scored, list(truth), SETTINGS)}
        mapped = {n: [Box('ball' if b.label == 'ball' else 'person', b.xyxy) for b in boxes] for n, boxes in truth.items()}
        for label, column in (('person', 0), ('ball', 1)):
            counts = [evaluate_image(mapped[n], [b for b in predictions[n] if b.score >= .25], .5)[0][column]
                      for n in truth]
            tp, fp, fn = (sum(c[k] for c in counts) for k in ('tp', 'fp', 'fn'))
            self.assertEqual(rows[f'{label}_recall_c0.25_iou0.5']['value'], tp/(tp+fn))
            self.assertEqual(rows[f'{label}_precision_c0.25_iou0.5']['n'], tp+fp)
            self.assertAlmostEqual(rows[f'{label}_AP50']['value'], average_precision(mapped, predictions, label, .5))
        classes = sum(rows[f'class_{c}_recall_c0.25_iou0.5']['n'] for c in ('player', 'referee'))
        self.assertEqual(classes, rows['person_recall_c0.25_iou0.5']['n'])

    def test_groups_and_bootstrap(self):
        records = [{'image': f'{c}{i}', 'source': c, 'game': 'g'} for c, n in (('a', 3), ('b', 3), ('c', 1))
                   for i in range(n)]
        groups = scoring.build_groups(records, {'groups': [['game']], 'small_groups': [['source']]}, 2)
        self.assertEqual([(g, len(n), level) for g, n, level in groups],
                         [('all', 7, 'all'), ('game=g', 7, 'group'), ('source=a', 3, 'small'), ('source=b', 3, 'small')])
        truth = {r['image']: [Box('ball', (0, 0, 10, 10))] for r in records}
        predictions = {name: [Box('ball', (0, 0, 10, 10) if name[0] != 'c' else (50, 50, 60, 60), .5)] for name in truth}
        scored = scoring.score_images(truth, predictions, {n: (100, 100) for n in truth}, {}, 100, [.5], .5, [8, 16], 4)
        clusters = {c: [r['image'] for r in records if r['source'] == c] for c in 'abc'}
        interval = scoring.cluster_bootstrap_ap(scored, clusters, 'ball', .5, 200, 0)
        self.assertEqual(interval, scoring.cluster_bootstrap_ap(scored, clusters, 'ball', .5, 200, 0))
        self.assertLessEqual(interval[0], interval[1])
        self.assertEqual(interval[1], 1)

    def test_weighted_resample_equals_concatenation(self):
        rng = np.random.default_rng(1)
        parts = [(rng.random(n), rng.random(n) < .5, rng.integers(1, 6)) for n in (5, 0, 8, 3)]
        scores = np.concatenate([p[0] for p in parts])
        order = np.argsort(-scores, kind='stable')
        matched = np.concatenate([p[1] for p in parts])[order]
        owner = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])[order]
        totals = np.array([p[2] for p in parts])
        for pick in ([0, 0, 2, 3], [1, 1, 1, 1], [3, 2, 2, 0]):
            weights = np.bincount(pick, minlength=len(parts))
            expected = scoring.ap(np.concatenate([parts[i][0] for i in pick]),
                                  np.concatenate([parts[i][1] for i in pick]), sum(parts[i][2] for i in pick))
            self.assertAlmostEqual(scoring.weighted_ap(matched, owner, weights, totals), expected)

    def test_fine_tuned_labels(self):
        boxes = [Box(label, (0, 0, 10, 10), .5) for label in ('player', 'referee', 'ball', 'person', 'sports ball', 'goal')]
        self.assertEqual([b.label for b in remap(boxes, PRED_MAPPING, strict=False)],
                         ['person', 'person', 'ball', 'person', 'ball'])
        records = [{'image': 'images/val/a.jpg', 'image_sha256': '1',
                    'boxes': [{'label': 'referee', 'xyxy': [0, 0, 10, 20], 'score': .9},
                              {'label': 'ball', 'xyxy': [5, 5, 9, 9], 'score': .4}]}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'predictions.json').write_text(json.dumps({'images': records}))
            manifest = {'predictions_sha256': sha256(folder/'predictions.json'),
                        'dataset': {'images': 1, 'view_sha256': view_digest(records)}}
            _, predictions = scoring.load_run(folder, manifest)
        self.assertEqual([b.label for b in predictions['images/val/a.jpg']], ['person', 'ball'])
        self.assertEqual(scoring.model_name('outputs/training/20261111T120000000000Z/weights/best.pt'),
                         '20261111T120000000000Z/best')
        self.assertEqual(scoring.model_name('models/yolo26n.pt'), 'yolo26n')

    def test_centre_recall(self):
        gt = [Box('ball', (100, 100, 110, 110)), Box('ball', (300, 300, 310, 310))]
        # Centre 2 px right of the box: IoU about .11, centre hit with a 4 px margin.
        near = Box('ball', (108, 101, 116, 109), .9)
        self.assertEqual(centre_match(gt, [near], 4).tolist(), [.9, -1])
        self.assertEqual(centre_match(gt, [near], 0).tolist(), [-1, -1])
        # One prediction claims one ball; the higher score claims first.
        twins = [Box('ball', (99, 99, 111, 111), .3), Box('ball', (100, 100, 110, 110), .8)]
        self.assertEqual(centre_match(gt[:1] * 2, twins, 4).tolist(), [.8, .3])
        truth = {'a': gt + [Box('player', (0, 0, 10, 30))]}
        predictions = {'a': [near, Box('ball', (301, 301, 309, 309), .1), Box('person', (0, 0, 10, 30), .9)]}
        scored = scoring.score_images(truth, predictions, {'a': (640, 360)}, {}, 640, [.5, .75], .5, [8, 16], 4)
        rows = {r['metric']: r for r in scoring.group_metrics(scored, ['a'], SETTINGS)}
        self.assertEqual(rows['ball_recall_c0.25_iou0.5']['value'], 0)
        self.assertEqual((rows['ball_centre_recall_c0.25_m4px']['value'], rows['ball_centre_recall_c0.25_m4px']['n']),
                         (.5, 2))
        self.assertEqual(rows['person_recall_c0.25_iou0.5']['value'], 1)
        row = {'run_id': 'r', 'dataset': 'd', 'group': 'all', 'model': 'models/yolo26n.pt', 'variant': 'default', 'imgsz': 640}
        table = scoring.primary_table([{**row, **r} for r in rows.values()], {}, SETTINGS, set())
        self.assertEqual((table[0]['ball_recall'], table[0]['ball_centre_recall'], table[0]['ball_n']), (0, .5, 2))
        self.assertTrue(set(table[0]) <= set(scoring.PRIMARY_FIELDS))

    def test_pool_holds_report_games_only(self):
        settings = yaml.safe_load((ROOT/'configs/scoring.yaml').read_text(encoding='utf-8'))
        training = yaml.safe_load((ROOT/'configs/training_set.yaml').read_text(encoding='utf-8'))
        members = {m for pool in settings['pools'].values() for m in pool['members']}
        self.assertEqual(members, {'soccernet_test_1fps', 'gsr_valid_1fps'})
        self.assertTrue(members <= set(training['report_sets']))
        note = settings['datasets']['soccernet_train_1fps']['notes']['all']
        self.assertNotIn('never used to train', note)


if __name__ == '__main__':
    unittest.main()
