"""MultiSports (SportsAction) football tubes: audit and a YOLO view of sampled video frames."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import pickle

import numpy as np

from sports_analyzer.dataset_preparation import sha256
from sports_analyzer.soccernet import NAMES, clip_box, yolo_line

SPORT = 'football'
PLAYER = 1
# Only numpy arrays may be rebuilt; the pickle is downloaded, so anything else is refused.
SAFE = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
        ('numpy', 'ndarray'), ('numpy', 'dtype')}
ERRORS = {'non_positive_size', 'outside_image', 'frame_out_of_range', 'duplicate_frame_in_tube', 'unknown_label'}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in SAFE:
            raise pickle.UnpicklingError(f'Refusing {module}.{name}')
        return super().find_class(module, name)


def load_annotations(path):
    with Path(path).open('rb') as stream:
        return SafeUnpickler(stream, encoding='latin1').load()


def source_video(video: str) -> str:
    # 'football/v_5uBLdpt5-oU_c602' -> 'v_5uBLdpt5-oU', the YouTube video the clip was cut from.
    return video.split('/', 1)[1].rsplit('_c', 1)[0]


def split_videos(gt, split, sport=SPORT):
    # The pickle calls its validation list test_videos.
    videos = gt['train_videos' if split == 'train' else 'test_videos'][0]
    return sorted(v for v in videos if v.startswith(f'{sport}/'))


def tube_rows(gt, video):
    rows = []
    for label_id, tubes in gt['gttubes'].get(video, {}).items():
        for number, tube in enumerate(tubes):
            for frame, x1, y1, x2, y2 in np.asarray(tube, float):
                rows.append(dict(frame=int(frame), label_id=int(label_id), tube=f'{label_id}_{number}',
                                 x1=x1, y1=y1, x2=x2, y2=y2))
    return rows


def row_issues(row, labels, width, height, nframes):
    codes = []
    if not 0 <= row['label_id'] < len(labels) or not labels[row['label_id']].startswith(SPORT):
        codes.append('unknown_label')
    if not 1 <= row['frame'] <= nframes:
        codes.append('frame_out_of_range')
    w, h = row['x2'] - row['x1'], row['y2'] - row['y1']
    if w <= 0 or h <= 0:
        codes.append('non_positive_size')
    else:
        xyxy, clipped = clip_box(row['x1'], row['y1'], w, h, width, height)
        if xyxy is None:
            codes.append('outside_image')
        elif clipped:
            codes.append('partly_outside')
    return codes


def audit_split(gt, split):
    labels, issues, examples = gt['labels'], Counter(), {}
    counts, actions, resolutions, sources = Counter(), Counter(), Counter(), Counter()
    videos, heights = [], []
    for video in split_videos(gt, split):
        height, width = gt['resolution'][video]
        nframes = gt['nframes'][video]
        rows = tube_rows(gt, video)
        resolutions[f'{width}x{height}'] += 1
        sources[source_video(video)] += 1
        by_tube, per_frame = defaultdict(list), Counter()
        for row in rows:
            by_tube[row['tube']].append(row['frame'])
            per_frame[row['frame']] += 1
            heights.append(row['y2'] - row['y1'])
            for code in row_issues(row, labels, width, height, nframes):
                issues[code] += 1
                examples.setdefault(code, f'{video} tube {row["tube"]} frame {row["frame"]}')
        for tube, frames in by_tube.items():
            actions[labels[int(tube.split('_')[0])]] += 1
            if len(set(frames)) != len(frames):
                issues['duplicate_frame_in_tube'] += 1
                examples.setdefault('duplicate_frame_in_tube', f'{video} tube {tube}')
            if max(frames) - min(frames) + 1 != len(set(frames)):
                issues['gap_in_tube'] += 1
                examples.setdefault('gap_in_tube', f'{video} tube {tube}')
        counts['frames'] += nframes
        counts['labelled_frames'] += len(per_frame)
        counts['tubes'] += len(by_tube)
        counts['boxes'] += len(rows)
        videos.append(dict(video=video, source=source_video(video), width=width, height=height, nframes=nframes,
                           tubes=len(by_tube), boxes=len(rows), labelled_frames=len(per_frame),
                           max_boxes_per_frame=max(per_frame.values(), default=0)))
    summary = dict(split=split, videos=len(videos), source_videos=len(sources), counts=dict(counts),
                   resolutions=dict(resolutions), actions=dict(actions.most_common()),
                   box_height_px=dict(zip(('p5', 'median', 'p95'),
                                          np.percentile(heights, [5, 50, 95]).round(1).tolist())) if heights else {},
                   max_boxes_per_frame=max((v['max_boxes_per_frame'] for v in videos), default=0),
                   errors=sum(v for k, v in issues.items() if k in ERRORS),
                   warnings=sum(v for k, v in issues.items() if k not in ERRORS),
                   issues=dict(sorted(issues.items())), examples=examples)
    return dict(summary=summary, videos=videos)


def shared_sources(gt, splits=('train', 'val')):
    sources = {s: {source_video(v) for v in split_videos(gt, s)} for s in splits}
    return sorted(set.intersection(*sources.values()))


def read_frames(path, wanted):
    """Decode in order like the dataset's generate_rgb.py: frame numbers start at 1."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    frames, number = {}, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        number += 1
        if number in wanted:
            frames[number] = frame
    capture.release()
    return frames, number


def build_view(gt, video_root, output_dir, split, step):
    """Every `step`-th frame that has at least one tube box; only acting players are labelled."""
    import cv2
    if step < 1:
        raise ValueError('Frame step must be positive')
    videos = split_videos(gt, split)
    missing = [v for v in videos if not (Path(video_root) / f'{v}.mp4').is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} videos missing under {video_root}, e.g. {missing[0]}')
    output_dir = Path(output_dir)
    image_root, label_root = output_dir / 'images' / split, output_dir / 'labels' / split
    output_dir.mkdir(parents=True, exist_ok=False)
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    counts, frames, tracks, decoded = Counter(), [], [], []
    for video in videos:
        height, width = gt['resolution'][video]
        by_frame = defaultdict(list)
        for row in tube_rows(gt, video):
            by_frame[row['frame']].append(row)
        wanted = {f for f in by_frame if (f - 1) % step == 0}
        images, total = read_frames(Path(video_root) / f'{video}.mp4', wanted)
        decoded.append(dict(video=video, annotated=gt['nframes'][video], decoded=total))
        counts['videos'] += 1
        counts['frame_count_mismatch'] += total != gt['nframes'][video]
        counts['missing_frames'] += len(wanted - set(images))
        clip = video.split('/', 1)[1]
        for number in sorted(images):
            image = images[number]
            if image.shape[:2] != (height, width):
                raise ValueError(f'{video} frame {number}: size {image.shape[1]}x{image.shape[0]}, '
                                 f'annotations say {width}x{height}')
            name = f'{clip}_{number:06d}'
            relative = f'images/{split}/{name}.jpg'
            # Same encoder and default quality as the dataset's own frame script.
            cv2.imwrite(str(output_dir / relative), image)
            lines = []
            for row in by_frame[number]:
                xyxy, was_clipped = clip_box(row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1'],
                                             width, height)
                counts['clipped_boxes'] += was_clipped and xyxy is not None
                if xyxy is None:
                    counts['dropped_boxes'] += 1
                    continue
                lines.append(yolo_line(PLAYER, xyxy, width, height))
                tracks.append(dict(image=relative, clip=clip, frame=number, x1=xyxy[0], y1=xyxy[1], x2=xyxy[2],
                                   y2=xyxy[3], label=NAMES[PLAYER], track=row['tube'],
                                   action=gt['labels'][row['label_id']]))
            (label_root / f'{name}.txt').write_text(''.join(f'{v}\n' for v in lines), encoding='utf-8')
            counts['frames'] += 1
            counts['boxes'] += len(lines)
            frames.append(dict(image=relative, clip=clip, game=source_video(video), frame=number))
    # No ignore regions in MultiSports; the empty file keeps the layout of the other views.
    tables = {'frames': (['image', 'clip', 'game', 'frame'], frames),
              'ignore': (['image', 'x1', 'y1', 'x2', 'y2', 'source'], []),
              'tracks': (['image', 'clip', 'frame', 'x1', 'y1', 'x2', 'y2', 'label', 'track', 'action'], tracks),
              'decoded': (['video', 'annotated', 'decoded'], decoded)}
    for name, (columns, rows) in tables.items():
        with (output_dir / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    key = 'train' if split == 'train' else 'val'
    (output_dir / 'data.yaml').write_text(
        f'path: .\n{key}: images/{split}\nnc: {len(NAMES)}\nnames:\n'
        + ''.join(f'  {k}: {v}\n' for k, v in NAMES.items()), encoding='utf-8')
    manifest = dict(source=str(Path(video_root).resolve()), split=split, step=step,
                    videos=len(videos), source_videos=len({source_video(v) for v in videos}), names=NAMES,
                    counts=dict(sorted(counts.items())),
                    files={f'{name}.csv': sha256(output_dir / f'{name}.csv') for name in tables},
                    code_sha256=sha256(Path(__file__)), opencv=cv2.__version__,
                    images='decoded from the source mp4 and written with cv2.imwrite',
                    labels='acting players only (one box per action tube); other people and the ball are not labelled',
                    created_utc=datetime.now(timezone.utc).isoformat())
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
