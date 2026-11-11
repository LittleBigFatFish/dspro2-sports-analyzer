"""Precision/recall and 101-point AP; no COCO area ranges, crowd or detection limit."""
from dataclasses import replace
import math
import numpy as np
from .annotations.yolo import Box

GT_MAPPING = {'player': 'person', 'referee': 'person', 'ball': 'ball'}
# COCO models emit person and sports ball; fine-tuned models emit the dataset classes.
PRED_MAPPING = {'person': 'person', 'sports ball': 'ball', **GT_MAPPING}


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


def iou_matrix(a, b) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    width = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    height = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    intersection = width * height
    union = ((a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]))[:, None] + ((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]))[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def match(ground_truth: list[Box], predictions: list[Box], thresholds, ignore=(), ignore_iou=0.5):
    """Per threshold: matched GT index (-1 if none) and ignore flag, in score order."""
    if any(not 0 < t <= 1 for t in thresholds):
        raise ValueError('Matching IoU threshold must be in (0, 1]')
    if any(b.score is None or not 0 <= b.score <= 1 for b in predictions):
        raise ValueError('Predictions require finite scores in [0, 1]')
    order = np.argsort(-np.array([b.score for b in predictions], float), kind='stable')
    boxes = [predictions[i].xyxy for i in order]
    overlaps = iou_matrix(boxes, [b.xyxy for b in ground_truth])
    best = overlaps.max(1) if len(ground_truth) else np.zeros(len(boxes))
    in_ignore = (iou_matrix(boxes, list(ignore)).max(1) >= ignore_iou) if len(ignore) and len(boxes) \
        else np.zeros(len(boxes), bool)
    results = []
    for threshold in thresholds:
        assigned = np.full(len(boxes), -1)
        available = np.ones(len(ground_truth), bool)
        for k in np.flatnonzero(best >= threshold):
            j = int(np.argmax(np.where(available, overlaps[k], -1.0)))
            if available[j] and overlaps[k, j] >= threshold:
                assigned[k] = j
                available[j] = False
        results.append((assigned, in_ignore & (assigned < 0)))
    return order, results


def centre(xyxy):
    return (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2


def centre_match(ground_truth: list[Box], predictions: list[Box], margin: float) -> np.ndarray:
    """Per GT box, score of the prediction whose centre lies in it grown by margin, else -1; greedy in score order."""
    found = np.full(len(ground_truth), -1.0)
    for box in sorted(predictions, key=lambda b: -b.score):
        x, y = centre(box.xyxy)
        inside = [j for j, g in enumerate(ground_truth) if found[j] < 0 and g.xyxy[0]-margin <= x <= g.xyxy[2]+margin
                  and g.xyxy[1]-margin <= y <= g.xyxy[3]+margin]
        if inside:
            found[min(inside, key=lambda j: math.dist((x, y), centre(ground_truth[j].xyxy)))] = box.score
    return found


def evaluate_image(ground_truth: list[Box], predictions: list[Box], threshold: float, ignore=()):
    if not 0 < threshold <= 1:
        raise ValueError('Matching IoU threshold must be in (0, 1]')
    rows, events = [], []
    for label in ('person', 'ball'):
        gt = [b for b in ground_truth if b.label == label]
        pred = [b for b in predictions if b.label == label]
        order, [(assigned, ignored)] = match(gt, pred, [threshold], ignore)
        for k, i in enumerate(order):
            box, j = pred[i], assigned[k]
            status = 'TP' if j >= 0 else 'ignored' if ignored[k] else 'FP'
            events.append({'label': label, 'status': status, 'xyxy': box.xyxy, 'score': box.score,
                           'matched_iou': iou(box.xyxy, gt[j].xyxy) if j >= 0 else None})
        matched = set(assigned[assigned >= 0].tolist())
        for j in range(len(gt)):
            if j not in matched:
                events.append({'label': label, 'status': 'FN', 'xyxy': gt[j].xyxy,
                               'score': None, 'matched_iou': None})
        tp = len(matched)
        fp, fn = len(pred)-tp-int(ignored.sum()), len(gt)-tp
        rows.append({'label': label, 'tp': tp, 'fp': fp, 'fn': fn,
                     'precision': tp/(tp+fp) if tp+fp else None,
                     'recall': tp/(tp+fn) if tp+fn else None})
    return rows, events


def ranked_curve(scores, matched, total: int):
    order = np.argsort(-np.asarray(scores, float), kind='stable')
    tp = np.cumsum(np.asarray(matched, bool)[order])
    recalls = tp/total if total else np.zeros(len(tp))
    return recalls, tp/np.arange(1, len(tp)+1)


def interpolated_ap(recalls, precisions) -> float:
    if not len(precisions):
        return 0.0
    envelope = np.maximum.accumulate(np.asarray(precisions, float)[::-1])[::-1]
    points = np.searchsorted(recalls, np.arange(101)/100, side='left')
    return float(np.where(points < len(envelope), envelope[np.minimum(points, len(envelope)-1)], 0.0).sum()) / 101


def precision_recall_curve(ground_truth: dict[str, list[Box]], predictions: dict[str, list[Box]],
                           label: str, threshold: float, ignore=None):
    """Inputs map image names to boxes; ignore maps them to ignore-region xyxy."""
    if not 0 < threshold <= 1:
        raise ValueError('Matching IoU threshold must be in (0, 1]')
    if set(predictions) - set(ground_truth):
        raise ValueError('Predictions for images without ground truth')
    scores, matched, total = [], [], 0
    for name, boxes in ground_truth.items():
        gt = [b for b in boxes if b.label == label]
        pred = [b for b in predictions.get(name, []) if b.label == label]
        order, [(assigned, ignored)] = match(gt, pred, [threshold], (ignore or {}).get(name, ()))
        total += len(gt)
        scores.extend(pred[i].score for k, i in enumerate(order) if not ignored[k])
        matched.extend((assigned >= 0)[~ignored].tolist())
    recalls, precisions = ranked_curve(scores, matched, total)
    return recalls.tolist(), precisions.tolist(), total


def average_precision(ground_truth, predictions, label: str, threshold: float, ignore=None):
    recalls, precisions, total = precision_recall_curve(ground_truth, predictions, label, threshold, ignore)
    return interpolated_ap(recalls, precisions) if total else None


def wilson(successes: int, n: int, z=1.959963984540054):
    if not n:
        return None, None
    p = successes/n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, centre-half), min(1.0, centre+half)


def aggregate(rows):
    output = []
    for label in ('person', 'ball'):
        selected = [r for r in rows if r['label'] == label]
        tp, fp, fn = (sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn'))
        output.append({'label': label, 'tp': tp, 'fp': fp, 'fn': fn,
                       'precision': tp/(tp+fp) if tp+fp else None,
                       'recall': tp/(tp+fn) if tp+fn else None})
    return output
