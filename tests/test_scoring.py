from pathlib import Path
import importlib.util
import random
import unittest

from sports_analyzer.annotations.yolo import Box
from sports_analyzer.evaluation import evaluate_image, average_precision, wilson

spec = importlib.util.spec_from_file_location('score_comparison', Path(__file__).resolve().parents[1] / 'scripts/score_comparison.py')
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)

SETTINGS = {'report_confidence': .25, 'match_iou': .5, 'ap_iou': [.5, .75], 'grid_iou': [.5],
            'grid_confidence': [.25], 'ball_size_edges': [8, 16]}


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
        scored = scoring.score_images(truth, predictions, sizes, {}, 640, [.5, .75], .5, [8, 16])
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
        scored = scoring.score_images(truth, predictions, {n: (100, 100) for n in truth}, {}, 100, [.5], .5, [8, 16])
        clusters = {c: [r['image'] for r in records if r['source'] == c] for c in 'abc'}
        interval = scoring.cluster_bootstrap_ap(scored, clusters, 'ball', .5, 200, 0)
        self.assertEqual(interval, scoring.cluster_bootstrap_ap(scored, clusters, 'ball', .5, 200, 0))
        self.assertLessEqual(interval[0], interval[1])
        self.assertEqual(interval[1], 1)


if __name__ == '__main__':
    unittest.main()
