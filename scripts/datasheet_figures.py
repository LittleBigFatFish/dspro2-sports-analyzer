"""Figures for docs/datasheet, read only from audit and scoring run folders."""
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [('NED_VS_FRA', 'NED-FRA'), ('Newcastle', 'Newcastle-Chelsea'), ('Serbia', 'Serbia-England U21'),
           ('PES_21', 'PES 2021 (video game)')]
CLASSES = {'player': '#0a50a0', 'referee': '#e69f00', 'ball': '#c0392b'}
GROUPS = [('football', 'split=val|footage=real', 'Football val\nreal'),
          ('football', 'split=val|footage=video_game', 'Football val\nvideo game'),
          ('soccana_train_fullframes', 'all', 'Soccana train\nfull frames'),
          ('soccernet_test_1fps', 'all', 'SN Tracking\ntest'),
          ('gsr_valid_1fps', 'all', 'SN GSR\nvalid'),
          ('soccernet_all_games', 'all', 'SoccerNet\nall 9 games')]
MODELS = ['yolov8n', 'yolo12n', 'yolo26n']
SIZES = {640: '#c0392b', 1280: '#0a50a0'}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def source(image: str) -> str:
    return next(label for key, label in SOURCES if key in image)


def ball_side(box, image, target: int) -> float:
    scale = target / max(int(image['width']), int(image['height']))
    return min(float(box['width_pixels']), float(box['height_pixels'])) * scale


def football_val(audit: Path, target: Path):
    images = {r['image']: r for r in read_csv(audit / 'images.csv')}
    boxes = [r for r in read_csv(audit / 'boxes.csv') if r['split'] == 'val']
    balls = [r for r in boxes if r['label'] == 'ball']
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 1.75), gridspec_kw={'width_ratios': [1.15, 1]})
    labels = [label for _, label in SOURCES]
    start = [0] * len(labels)
    for name, color in CLASSES.items():
        counts = [sum(source(r['image']) == label and r['label'] == name for r in boxes) for label in labels]
        left.barh(labels, counts, left=start, color=color, label=name)
        start = [a + b for a, b in zip(start, counts)]
    for y, total in enumerate(start):
        left.text(total + 10, y, str(total), va='center')
    left.invert_yaxis()
    left.set_xlim(0, 1150)
    left.set_xlabel(f'val boxes (n = {len(boxes)})')
    left.legend(frameon=False, loc='center right')
    left.set_title('(a) Val boxes by source and class', loc='left')
    for size, color in SIZES.items():
        sides = [ball_side(r, images[r['image']], size) for r in balls]
        right.hist(sides, bins=range(0, 25), alpha=0.6, color=color, label=f'{size} px input')
    right.axvline(8, color='black', lw=0.8, ls='--')
    right.text(8.3, right.get_ylim()[1] * 0.9, '8 px')
    right.set_xlabel('ball box, shorter side after resizing (px)')
    right.set_ylabel(f'val balls (n = {len(balls)})')
    right.legend(frameon=False)
    right.set_title('(b) Val ball size at 640 and 1280', loc='left')
    save(fig, (left, right), target / 'football_val.pdf')


def ball_recall(scoring: Path, target: Path):
    rows = {(r['dataset'], r['group'], r['model'], int(r['imgsz'])): r for r in read_csv(scoring / 'primary.csv')
            if r.get('variant', 'default') == 'default'}
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(7.0, 2.0), sharey=True)
    for ax, (dataset, group, title) in zip(axes, GROUPS):
        for offset, (size, color) in zip((-0.12, 0.12), SIZES.items()):
            picked = [rows[(dataset, group, m, size)] for m in MODELS]
            values = [float(r['ball_recall']) for r in picked]
            errors = [[v - float(r['ball_recall_low']) for v, r in zip(values, picked)],
                      [float(r['ball_recall_high']) - v for v, r in zip(values, picked)]]
            ax.errorbar([i + offset for i in range(len(MODELS))], values, yerr=errors, fmt='o', ms=3,
                        capsize=2, lw=0.8, color=color, label=f'{size} px')
        ax.set_title(f"{title} (n = {picked[0]['ball_n']})", fontsize=7)
        ax.set_xticks(range(len(MODELS)), ['v8n', '12n', '26n'])
        ax.set_ylim(0, 0.7)
        ax.grid(axis='y', lw=0.3)
    axes[0].set_ylabel('ball recall')
    axes[2].legend(frameon=False, loc='upper left')
    save(fig, axes, target / 'ball_recall.pdf')


def save(fig, axes, path: Path):
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, metadata={'Creator': None, 'Producer': None})
    plt.close(fig)
    print(path.relative_to(ROOT).as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--football-audit', type=Path,
                        default=ROOT / 'outputs/dataset_preparation/20261022T191930876487Z')
    parser.add_argument('--scoring', type=Path, default=ROOT / 'outputs/scoring/20261103T142737368356Z')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/datasheet/figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    football_val(args.football_audit, args.output)
    ball_recall(args.scoring, args.output)


if __name__ == '__main__':
    main()
