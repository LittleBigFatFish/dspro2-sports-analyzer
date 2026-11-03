"""SkillCorner open data loader and possession rules."""
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
import csv
import json
import math

FPS = 10


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def match_files(match_dir):
    match_dir = Path(match_dir)
    names = dict(match='match.json', tracking='tracking_extrapolated.jsonl',
                 events='dynamic_events.csv', phases='phases_of_play.csv')
    return {key: match_dir / f'{match_dir.name}_{name}' for key, name in names.items()}


def player_teams(match):
    sides = {match['home_team']['id']: 'home', match['away_team']['id']: 'away'}
    teams = {}
    for player in match['players']:
        if player['team_id'] not in sides:
            raise ValueError(f'Player {player["id"]} has unknown team {player["team_id"]}')
        teams[player['id']] = sides[player['team_id']]
    return teams


def seconds(timestamp):
    if timestamp is None:
        return None
    hours, minutes, rest = timestamp.split(':')
    return int(hours)*3600 + int(minutes)*60 + float(rest)


def iter_frames(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{number}: {error.msg}') from None


def nearest_player(frame):
    ball = frame['ball_data']
    if ball['x'] is None or ball['y'] is None:
        return None, None
    best, distance = None, None
    for player in frame['player_data']:
        if player['x'] is None or player['y'] is None:
            continue
        d = math.hypot(player['x']-ball['x'], player['y']-ball['y'])
        if distance is None or d < distance:
            best, distance = player['player_id'], d
    return best, distance


def frame_record(frame):
    ball, possession = frame['ball_data'], frame['possession']
    player, distance = nearest_player(frame)
    return dict(frame=frame['frame'], period=frame['period'], ball_z=ball['z'],
                ball_detected=ball['is_detected'], group=possession['group'],
                holder=possession['player_id'], nearest=player, distance=distance)


def summarise_tracking(frames, teams, length, width, margin=5.0):
    counts, groups, unknown, periods = Counter(), Counter(), Counter(), Counter()
    players, detected, records, seen = [], [], [], set()
    last, max_gap, first_frame = None, 0, None
    half_x, half_y = length/2 + margin, width/2 + margin
    for frame in frames:
        number = frame['frame']
        counts['frames'] += 1
        first_frame = number if first_frame is None else first_frame
        if number in seen:
            counts['duplicate_frames'] += 1
        seen.add(number)
        if last is not None and number != last['frame'] + 1:
            counts['frame_gaps'] += 1
            max_gap = max(max_gap, number - last['frame'])
        if frame['period'] is None:
            counts['frames_no_period'] += 1
            last = frame
            continue
        counts['period_frames'] += 1
        periods[str(frame['period'])] += 1
        if last is not None and last['period'] == frame['period'] and last['timestamp'] and frame['timestamp']:
            step = seconds(frame['timestamp']) - seconds(last['timestamp'])
            counts['timestamp_steps_not_0.1s'] += abs(step - (number - last['frame']) / FPS) > 1e-6
        if frame['timestamp'] is None:
            counts['period_no_timestamp'] += 1
        ball = frame['ball_data']
        if ball['x'] is not None and ball['y'] is not None:
            counts['ball_xy'] += 1
            counts['ball_detected'] += bool(ball['is_detected'])
            counts['ball_z'] += ball['z'] is not None
            counts['ball_outside_pitch'] += abs(ball['x']) > half_x or abs(ball['y']) > half_y
        counts['camera_polygon'] += frame['image_corners_projection'].get('x_top_left') is not None
        groups[frame['possession']['group']] += 1
        counts['possession_player'] += frame['possession']['player_id'] is not None
        ids = [p['player_id'] for p in frame['player_data']]
        counts['duplicate_player_in_frame'] += len(ids) != len(set(ids))
        for p in frame['player_data']:
            if p['player_id'] not in teams:
                unknown[p['player_id']] += 1
            if p['x'] is None or p['y'] is None:
                counts['player_xy_missing'] += 1
                continue
            counts['player_positions'] += 1
            counts['player_detected'] += bool(p['is_detected'])
            counts['player_outside_pitch'] += abs(p['x']) > half_x or abs(p['y']) > half_y
        players.append(len(ids))
        detected.append(sum(bool(p['is_detected']) for p in frame['player_data']))
        records.append(frame_record(frame))
        last = frame
    # Frames without any player have no tracking at all, so they are counted apart.
    tracked = [v for v in players if v]
    seen_detected = [d for v, d in zip(players, detected) if v]
    n = counts['period_frames'] or None
    share = (lambda k, d=n: counts[k] / d if d else None)
    summary = dict(
        frames=counts['frames'], first_frame=first_frame, last_frame=last['frame'] if last else None,
        period_frames=counts['period_frames'], period_minutes=counts['period_frames'] / FPS / 60,
        frames_by_period=dict(sorted(periods.items())),
        ball_xy_share=share('ball_xy'), ball_detected_share=share('ball_detected'),
        ball_detected_of_xy_share=share('ball_detected', counts['ball_xy']),
        ball_z_share=share('ball_z', counts['ball_xy']), camera_polygon_share=share('camera_polygon'),
        players_per_frame_mean=sum(players) / len(players) if players else None,
        players_per_frame_min=min(players, default=None), players_per_frame_max=max(players, default=None),
        detected_per_frame_mean=sum(detected) / len(detected) if detected else None,
        detected_per_frame_min=min(detected, default=None), detected_per_frame_max=max(detected, default=None),
        tracked_frames=len(tracked), players_per_tracked_frame_mean=sum(tracked) / len(tracked) if tracked else None,
        players_per_tracked_frame_min=min(tracked, default=None),
        detected_per_tracked_frame_mean=sum(seen_detected) / len(seen_detected) if seen_detected else None,
        detected_per_tracked_frame_min=min(seen_detected, default=None),
        player_extrapolated_share=1 - share('player_detected', counts['player_positions'])
        if counts['player_positions'] else None,
        possession_group_share={str(k): v / n for k, v in sorted(groups.items(), key=str)} if n else {},
        possession_player_share=share('possession_player'),
        unknown_player_ids=sorted(unknown), max_frame_gap=max_gap,
        issues={k: counts[k] for k in ('duplicate_frames', 'frame_gaps', 'frames_no_period', 'period_no_timestamp',
                                       'timestamp_steps_not_0.1s', 'duplicate_player_in_frame',
                                       'player_xy_missing', 'player_outside_pitch', 'ball_outside_pitch')})
    summary['issues']['unknown_player_positions'] = sum(unknown.values())
    summary['issues']['period_frames_without_ball'] = counts['period_frames'] - counts['ball_xy']
    summary['issues']['period_frames_without_players'] = players.count(0)
    return summary, records


PASS_VALUE_FIELDS = ('x_start', 'y_start', 'x_end', 'y_end', 'pass_outcome', 'player_targeted_id',
                     'player_targeted_x_reception', 'player_targeted_y_reception', 'player_targeted_xthreat',
                     'player_targeted_xpass_completion', 'pass_epv_total', 'pass_distance', 'pass_angle')


def summarise_events(events, length, width, first_frame, last_frame, margin=5.0):
    counts = Counter()
    ids = Counter(r['event_id'] for r in events)
    types = Counter(r['event_type'] for r in events)
    subtypes = Counter((r['event_type'], r['event_subtype']) for r in events)
    possessions = [r for r in events if r['event_type'] == 'player_possession']
    passes = [r for r in possessions if r['end_type'] == 'pass']
    for r in events:
        for key in ('frame_start', 'frame_end', 'period', 'player_id', 'team_id', 'x_start', 'y_start'):
            counts[f'missing_{key}'] += not r[key]
        if r['frame_start'] and r['frame_end']:
            start, end = int(r['frame_start']), int(r['frame_end'])
            counts['frame_end_before_start'] += end < start
            counts['frames_outside_tracking'] += start < first_frame or end > last_frame
        for axis, limit in (('x', length/2 + margin), ('y', width/2 + margin)):
            for key in (f'{axis}_start', f'{axis}_end'):
                counts['coordinates_outside_pitch'] += bool(r[key]) and abs(float(r[key])) > limit
        counts['fully_extrapolated'] += r.get('fully_extrapolated') == 'True'
    flags = ('carry', 'one_touch', 'quick_pass', 'high_pass', 'lead_to_shot', 'lead_to_goal')
    return dict(
        rows=len(events), columns=len(events[0]) if events else 0,
        duplicate_event_ids=sum(v - 1 for v in ids.values() if v > 1),
        event_types=dict(types.most_common()),
        event_subtypes={f'{t}/{s or "-"}': v for (t, s), v in subtypes.most_common()},
        possession_start_types=dict(Counter(r['start_type'] for r in possessions).most_common()),
        possession_end_types=dict(Counter(r['end_type'] for r in possessions).most_common()),
        pass_outcomes=dict(Counter(r['pass_outcome'] or '-' for r in passes).most_common()),
        possession_flags={k: sum(r.get(k) == 'True' for r in possessions) for k in flags},
        pass_field_share={k: sum(bool(r.get(k)) for r in passes) / len(passes) if passes else None
                          for k in PASS_VALUE_FIELDS},
        issues=dict(sorted(counts.items())))


def summarise_phases(phases):
    return dict(rows=len(phases), seconds=sum(float(r['duration']) for r in phases),
                in_possession_types=dict(Counter(r['team_in_possession_phase_type'] for r in phases).most_common()),
                out_of_possession_types=dict(
                    Counter(r['team_out_of_possession_phase_type'] for r in phases).most_common()))


def nearest_owners(records, radius, max_z=None):
    owners = []
    for r in records:
        near = r['distance'] is not None and r['distance'] <= radius
        low = max_z is None or r['ball_z'] is None or r['ball_z'] <= max_z
        owners.append((r['frame'], r['period'], r['nearest'] if near and low else None))
    return owners


def provided_owners(records):
    return [(r['frame'], r['period'], r['holder']) for r in records]


def possessions(owners, hold=3):
    """The ball changes owner only after `hold` consecutive frames."""
    if hold < 1:
        raise ValueError('Hold must be at least one frame')
    segments, current, candidate, streak, start, last = [], None, None, 0, None, None
    for frame, period, player in owners:
        if current is not None and period != current['period']:
            current = None
        if last is None or frame != last[0] + 1 or period != last[1]:
            candidate, streak = None, 0
        last = (frame, period)
        if period is None or player is None:
            candidate, streak = None, 0
            continue
        if player == candidate:
            streak += 1
        else:
            candidate, streak, start = player, 1, frame
        if current is not None and player == current['player']:
            current['end'] = frame
        elif streak >= hold:
            current = dict(player=player, period=period, start=start, end=frame)
            segments.append(current)
    return segments


def transitions(segments, teams, max_gap=50):
    found = []
    for a, b in zip(segments, segments[1:]):
        if a['period'] != b['period'] or b['start'] - a['end'] > max_gap:
            continue
        team_a, team_b = teams.get(a['player']), teams.get(b['player'])
        if team_a is None or team_b is None:
            continue
        found.append(dict(frame=b['start'], passer=a['player'], receiver=b['player'], team=team_a,
                          kind='pass' if team_a == team_b else 'turnover', gap=b['start'] - a['end']))
    return found


def event_transitions(events):
    rows = sorted((r for r in events if r['event_type'] == 'player_possession'),
                  key=lambda r: (int(r['period']), int(r['frame_start'])))
    found = []
    for a, b in zip(rows, rows[1:]):
        if a['period'] != b['period']:
            continue
        if a['team_id'] != b['team_id']:
            kind = 'turnover'
        elif a['player_id'] != b['player_id'] and a['end_type'] == 'pass':
            kind = 'pass'
        else:
            kind = 'other'
        found.append(dict(frame=int(b['frame_start']), passer=int(a['player_id']), receiver=int(b['player_id']),
                          kind=kind, outcome=a['pass_outcome'], gap=int(b['frame_start']) - int(a['frame_end'])))
    return found


def match_transitions(found, truth, kind, tolerance=10):
    found = [t for t in found if t['kind'] == kind]
    truth = sorted((t for t in truth if t['kind'] == kind), key=lambda t: t['frame'])
    frames = [t['frame'] for t in truth]
    pairs = sorted((abs(f['frame'] - truth[j]['frame']), i, j) for i, f in enumerate(found)
                   for j in range(bisect_left(frames, f['frame'] - tolerance),
                                  bisect_right(frames, f['frame'] + tolerance)))
    used_found, used_truth, matched = set(), set(), []
    for offset, i, j in pairs:
        if i not in used_found and j not in used_truth:
            used_found.add(i)
            used_truth.add(j)
            matched.append((found[i], truth[j], offset))
    tp = len(matched)
    precision = tp / len(found) if found else None
    recall = tp / len(truth) if truth else None
    f1 = 2*precision*recall / (precision+recall) if precision and recall else 0.0
    same = sum(f['passer'] == t['passer'] and f['receiver'] == t['receiver'] for f, t, _ in matched)
    offsets = sorted(o for _, _, o in matched)
    return dict(kind=kind, truth=len(truth), found=len(found), matched=tp, precision=precision, recall=recall,
                f1=f1, same_players=same, median_offset_frames=offsets[len(offsets)//2] if offsets else None)


def possession_share(segments, teams):
    frames = Counter()
    for s in segments:
        frames[teams.get(s['player'], 'unknown')] += s['end'] - s['start'] + 1
    total = frames['home'] + frames['away']
    return frames['home'] / total if total else None
