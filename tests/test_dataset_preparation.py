from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import random
import unittest
import zipfile

from PIL import Image

from sports_analyzer.dataset_preparation import (audit_dataset, extract_zip, hamming_matches,
                                                  image_hashes, save_audit, save_overlays)


def pattern(size, seed):
    # Smooth random blobs keep the dHash stable under JPEG compression.
    rng = random.Random(seed)
    small = Image.frombytes('RGB', (8, 6), bytes(rng.randrange(256) for _ in range(8*6*3)))
    return small.resize(size, Image.Resampling.BICUBIC)


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

    def audit(self, **options):
        return audit_dataset(self.root, self.metadata, ['ball', 'player', 'referee'], **options)

    def add(self, split, name, image, label='1 0.5 0.5 0.2 0.4\n'):
        for kind in ('images', 'labels'):
            (self.root / kind / split).mkdir(parents=True, exist_ok=True)
        image.save(self.root / 'images' / split / name)
        (self.root / 'labels' / split / Path(name).with_suffix('.txt')).write_text(label)

    def test_exact_cross_split_and_coordinate_conversion(self):
        result = self.audit()
        self.assertEqual(result['summary']['errors'], 0)
        self.assertEqual(result['summary']['cross_split_exact_pairs'], 1)
        self.assertEqual(result['boxes'][0]['x1'], 40)
        self.assertEqual(result['boxes'][0]['height_pixels'], 32)
        self.assertTrue(all(r['status'] == 'ok' for r in result['images']))
        self.assertEqual(result['exclusions'], [dict(image='images/val/frame.01.png',
                                                     reasons='leak_byte_identical')])

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
        self.assertEqual(result['summary']['empty_labels'], 1)
        target = save_audit(result, self.root / 'reports')
        self.assertEqual(len(list(target.iterdir())), 6)
        self.assertTrue((target / 'boxes.csv').read_text().startswith('split,image,'))
        self.assertIn('images/train/frame.01.png\tmissing_label', (target / 'exclusions.txt').read_text())

    def test_any_splits_and_missing_split_folder(self):
        self.add('test', 'other.png', pattern((64, 48), 1))
        result = self.audit(splits=('train', 'test'))
        self.assertEqual(result['summary']['images_by_split'], {'train': 1, 'test': 1})
        with self.assertRaises(FileNotFoundError):
            self.audit(splits=('train', 'holdout'))
        with self.assertRaises(ValueError):
            self.audit(splits=('train', 'train'))

    def test_out_of_bounds_rows_are_listed_and_excluded(self):
        self.add('train', 'edge.png', pattern((64, 48), 2),
                 '0 0.95 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.5 1.2 0.1 0.1\n')
        result = self.audit()
        rows = [(i['path'], i['line']) for i in result['issues'] if i['code'] == 'out_of_bounds']
        self.assertEqual(rows, [('labels/train/edge.txt', 1), ('labels/train/edge.txt', 3)])
        self.assertEqual((result['summary']['out_of_bounds_rows'], result['summary']['out_of_bounds_images']),
                         (2, 1))
        self.assertIn(dict(image='images/train/edge.png', reasons='out_of_bounds'), result['exclusions'])
        self.assertEqual(sum(b['image'] == 'images/train/edge.png' for b in result['boxes']), 1)

    def test_crop_flag_needs_name_and_slice_size(self):
        self.add('train', 'v1_sahi_64_3.png', pattern((64, 64), 3))
        self.add('train', 'v1_sahi_32_4.png', pattern((64, 64), 4))
        self.add('train', 'v1_sahi.png', pattern((64, 64), 5))
        result = self.audit()
        crops = {r['image']: r['crop'] for r in result['images'] if 'sahi' in r['image']}
        self.assertEqual(crops, {'images/train/v1_sahi.png': False, 'images/train/v1_sahi_32_4.png': False,
                                 'images/train/v1_sahi_64_3.png': True})
        self.assertEqual(result['summary']['crops'], 1)
        self.assertIn('crop_size_mismatch', {i['code'] for i in result['issues']})
        self.assertEqual(self.audit(crop_pattern=None)['summary']['crops'], 0)

    def test_mirrored_twin_across_splits(self):
        image = pattern((64, 48), 6)
        self.add('train', 'source.png', image)
        self.add('val', 'twin.png', image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        self.add('val', 'jpeg_twin.jpg', image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        result = self.audit()
        kinds = {(d['image_a'], d['image_b']): d['kind'] for d in result['duplicates']}
        self.assertEqual(kinds[('images/train/source.png', 'images/val/twin.png')], 'mirror_identical')
        self.assertEqual(kinds[('images/train/source.png', 'images/val/jpeg_twin.jpg')], 'mirror_candidate')
        self.assertEqual(result['summary']['cross_split_mirror_pairs'], 2)
        self.assertEqual(result['summary']['leaked_images_by_kind'],
                         {'byte_identical': 1, 'mirror_identical': 1, 'mirror_candidate': 1})
        self.assertNotIn('images/train/source.png', {r['image'] for r in result['exclusions']})

    def test_banded_search_matches_brute_force(self):
        rng = random.Random(7)
        centres = [rng.getrandbits(64) for _ in range(20)]
        hashes = [c ^ sum(1 << rng.randrange(64) for _ in range(rng.randrange(7)))
                  for c in centres for _ in range(10)]
        for distance in (0, 2, 4, 9):
            expected = {(i, j) for i, a in enumerate(hashes) for j, b in enumerate(hashes)
                        if (a ^ b).bit_count() <= distance}
            self.assertEqual({(i, j) for i, j, _ in hamming_matches(hashes, hashes, distance)}, expected)

    def test_overlays_are_written_per_class(self):
        self.add('train', 'ball.png', pattern((200, 120), 8), '0 0.5 0.5 0.05 0.05\n2 0.2 0.3 0.1 0.2\n')
        target = save_overlays(self.audit(), self.root / 'reports', per_class=1)
        self.assertEqual(sorted(p.name for p in target.iterdir()),
                         ['ball', 'ball.jpg', 'player', 'player.jpg', 'referee', 'referee.jpg'])
        self.assertEqual(len(list((target / 'player').iterdir())), 1)

    def test_overlap_script_finds_mirrored_and_named_images(self):
        spec = importlib.util.spec_from_file_location(
            'check_overlap', Path(__file__).resolve().parents[1] / 'scripts/check_overlap.py')
        check_overlap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check_overlap)
        image = pattern((64, 48), 9)
        self.add('train', 'SNMOT-060_000001.png', image)
        sources = [dict(image_sha256='', **image_hashes(i)) for i in (pattern((64, 48), 10), image)]
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        references = [dict(image_sha256='x', **image_hashes(flipped))]
        self.assertEqual([m[:3] for m in check_overlap.overlaps(references, sources, 4)],
                         [(0, 1, 'mirror_identical')])
        hits = list(check_overlap.name_hits(self.root, check_overlap.SOCCERNET_PATTERNS))
        self.assertEqual([h[0] for h in hits], ['images/train/SNMOT-060_000001.png',
                                                'labels/train/SNMOT-060_000001.txt'])

    def test_extract_zip_rejects_unsafe_members_and_strips_prefix(self):
        archive = self.root / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('V1/labels/a.txt', '0 0.5 0.5 0.1 0.1\n')
            stream.writestr('V1/labels/train.cache', 'x')
        self.assertEqual(extract_zip(archive, self.root / 'out', 'V1/', ['.cache']), ['V1/labels/train.cache'])
        self.assertEqual([p.name for p in (self.root / 'out').rglob('*') if p.is_file()], ['a.txt'])
        with self.assertRaises(FileExistsError):
            extract_zip(archive, self.root / 'out', 'V1/')
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('V1/../evil.txt', 'x')
        with self.assertRaises(ValueError):
            extract_zip(archive, self.root / 'bad', 'V1/')
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
