import tempfile
import unittest
from pathlib import Path
from sports_analyzer.annotations.yolo import Box, read_labels
from sports_analyzer.evaluation import evaluate_image, aggregate, remap, GT_MAPPING

class BaselineTests(unittest.TestCase):
    def test_duplicate_prediction_is_false_positive(self):
        gt = [Box('ball', (0, 0, 10, 10))]
        pred = [Box('ball', (0, 0, 10, 10), .9), Box('ball', (0, 0, 10, 10), .8)]
        row = evaluate_image(gt, pred, .5)[0][1]
        self.assertEqual((row['tp'], row['fp'], row['fn']), (1, 1, 0))
        self.assertEqual(row['precision'], .5)

    def test_missed_and_wrong_class(self):
        gt = [Box('ball', (0, 0, 10, 10))]
        rows, _ = evaluate_image(gt, [Box('person', (0, 0, 10, 10), .9)], .5)
        self.assertEqual(rows[0]['fp'], 1)
        self.assertEqual(rows[1]['fn'], 1)
        self.assertIsNone(rows[1]['precision'])
        self.assertEqual(aggregate(rows)[1]['recall'], 0)

    def test_annotation_conversion_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.txt'
            path.write_text('0 0.5 0.5 0.2 0.4\n')
            self.assertEqual(read_labels(path, 100, 100, {0:'ball'})[0].xyxy, (40,30,60,70))
            for value in ['0 nan .5 .2 .2', '0 .5 .5 -.1 .2', '5 .5 .5 .2 .2', '0 .9 .5 .8 .2', '0 .5 .5 .2 .2 .9']:
                path.write_text(value)
                with self.assertRaises(ValueError):
                    read_labels(path, 100, 100, {0:'ball'})
            path.write_text('')
            self.assertEqual(read_labels(path, 100, 100, {0:'ball'}), [])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                read_labels(path, 100, 100, {0:'ball'})

    def test_role_mapping(self):
        boxes = remap([Box('referee',(0,0,10,10)), Box('player',(10,10,20,20))], GT_MAPPING, True)
        self.assertEqual([b.label for b in boxes], ['person', 'person'])

if __name__ == '__main__':
    unittest.main()
