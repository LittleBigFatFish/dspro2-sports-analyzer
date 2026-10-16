"""Development-sample precision/recall, not a COCO mAP implementation."""
from dataclasses import replace
from .annotations.yolo import Box

GT_MAPPING = {'player': 'person', 'referee': 'person', 'ball': 'ball'}
PRED_MAPPING = {'person': 'person', 'sports ball': 'ball'}


def remap(boxes: list[Box], mapping: dict[str, str], strict: bool) -> list[Box]:
    output = []
    for box in boxes:
        if box.label not in mapping:
            if strict:
                raise ValueError(f'No evaluation mapping for {box.label!r}')
            continue
        output.append(replace(box, label=mapping[box.label]))
    return output


def iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_image(ground_truth: list[Box], predictions: list[Box], threshold: float):
    if not 0 < threshold <= 1:
        raise ValueError('Matching IoU threshold must be in (0, 1]')
    rows, events = [], []
    for label in ('person', 'ball'):
        gt = [b for b in ground_truth if b.label == label]
        pred = [b for b in predictions if b.label == label]
        if any(b.score is None or not 0 <= b.score <= 1 for b in pred):
            raise ValueError('Predictions require finite scores in [0, 1]')
        pred.sort(key=lambda b: b.score, reverse=True)
        available = set(range(len(gt)))
        tp = 0
        for box in pred:
            best = max(sorted(available), key=lambda j: iou(box.xyxy, gt[j].xyxy), default=None)
            overlap = iou(box.xyxy, gt[best].xyxy) if best is not None else 0.0
            matched = best is not None and overlap >= threshold
            if matched:
                available.remove(best)
                tp += 1
            events.append({'label': label, 'status': 'TP' if matched else 'FP',
                           'xyxy': box.xyxy, 'score': box.score, 'matched_iou': overlap if matched else None})
        for j in sorted(available):
            events.append({'label': label, 'status': 'FN', 'xyxy': gt[j].xyxy,
                           'score': None, 'matched_iou': None})
        fp, fn = len(pred)-tp, len(gt)-tp
        rows.append({'label': label, 'tp': tp, 'fp': fp, 'fn': fn,
                     'precision': tp/(tp+fp) if tp+fp else None,
                     'recall': tp/(tp+fn) if tp+fn else None})
    return rows, events


def aggregate(rows):
    output = []
    for label in ('person', 'ball'):
        selected = [r for r in rows if r['label'] == label]
        tp, fp, fn = (sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn'))
        output.append({'label': label, 'tp': tp, 'fp': fp, 'fn': fn,
                       'precision': tp/(tp+fp) if tp+fp else None,
                       'recall': tp/(tp+fn) if tp+fn else None})
    return output
