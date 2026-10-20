from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from sports_analyzer.dataset_preparation import audit_dataset, save_audit


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ('train', 'val'):
            (self.root / 'images' / split).mkdir(parents=True)
            (self.root / 'labels' / split).mkdir(parents=True)
            Image.new('RGB', (100, 80), 'green').save(self.root / 'images' / split / 'frame.01.png')
            (self.root / 'labels' / split / 'frame.01.txt').write_text('1 0.5 0.5 0.2 0.4\n')
        self.metadata = self.root / 'data.yaml'
        self.metadata.write_text('names: [ball, player, referee]\n')

    def audit(self):
        return audit_dataset(self.root, self.metadata, ['ball', 'player', 'referee'])

    def test_exact_cross_split_and_coordinate_conversion(self):
        result = self.audit()
        self.assertEqual(result['summary']['errors'], 0)
        self.assertEqual(result['summary']['cross_split_exact_pairs'], 1)
        self.assertEqual(result['boxes'][0]['x1'], 40)
        self.assertEqual(result['boxes'][0]['height_pixels'], 32)
        self.assertTrue(all(r['status'] == 'ok' for r in result['images']))

    def test_collect_multiple_bad_rows_and_keep_valid_rows(self):
        label = self.root / 'labels/train/frame.01.txt'
        original = '9 0.5 0.5 0.2 0.4\n1 nan 0.5 0.2 0.4\n0 0.5 0.5 0.02 0.02\n'
        label.write_text(original)
        result = self.audit()
        self.assertEqual(result['summary']['errors'], 2)
        self.assertEqual(result['summary']['boxes'], 2)
        self.assertEqual(result['images'][0]['status'], 'error')
        self.assertIn('tiny_box', {r['code'] for r in result['issues']})
        self.assertEqual(label.read_text(), original)

    def test_missing_orphan_and_empty_are_distinct_and_reported(self):
        (self.root / 'labels/train/frame.01.txt').unlink()
        (self.root / 'labels/train/orphan.txt').write_text('')
        (self.root / 'labels/val/frame.01.txt').write_text('')
        result = self.audit()
        self.assertEqual({r['code'] for r in result['issues']},
                         {'missing_label', 'orphan_label', 'empty_label'})
        target = save_audit(result, self.root / 'reports')
        self.assertEqual(len(list(target.iterdir())), 5)
        self.assertTrue((target / 'boxes.csv').read_text().startswith('split,image,'))


if __name__ == '__main__':
    unittest.main()
