"""Input validation, provenance and figures for the baseline notebook."""
from pathlib import Path
import hashlib
import math
from PIL import Image
from .annotations.yolo import read_labels

def sha256(path: Path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_sample(image_dir: Path, label_dir: Path, names):
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError('Image or label directory missing; check baseline.yaml')
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not images:
        raise ValueError('No JPG/PNG images found')
    if len({p.stem for p in images}) != len(images):
        raise ValueError('Duplicate image stems would make label matching ambiguous')
    records = []
    for path in images:
        with Image.open(path) as image:
            image.load()
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f'{path}: EXIF orientation requires explicit annotation-safe handling')
            width, height = image.size
            mode = image.mode
        label_path = label_dir / f'{path.stem}.txt'
        boxes = read_labels(label_path, width, height, names)
        records.append({'path': path, 'label_path': label_path, 'width': width, 'height': height,
                        'mode': mode, 'image_sha256': sha256(path), 'label_sha256': sha256(label_path),
                        'ground_truth': boxes})
    return records


def show_grid(records, boxes_by_name=None, title='Original images'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colors = {'player': '#0072B2', 'referee': '#CC79A7', 'person': '#0072B2', 'ball': '#D55E00'}
    fig, axes = plt.subplots(math.ceil(len(records)/2), 2, figsize=(16, 5*math.ceil(len(records)/2)), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, record in zip(axes.flat, records):
        with Image.open(record['path']) as image:
            ax.imshow(image.convert('RGB'))
        if boxes_by_name is not None:
            for box in boxes_by_name[record['path'].name]:
                x1, y1, x2, y2 = box.xyxy
                color = colors.get(box.label, '#999999')
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1))
                ax.text(x1, y1, box.label, fontsize=6, color='white', backgroundcolor=color)
        ax.set_title(record['path'].name, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
