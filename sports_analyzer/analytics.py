"""Ball holder, possessions, passes, holding share and goal-mouth events on positions.

A frame is dict(frame, period, ball, players), each player dict(id, team, role, position).
Pitch mode: metres, ball (x, y) or (x, y, z). Image mode: position is the feet point and `height`
the box height in pixels, ball is the box centre; distances are in player heights.
"""
from collections import Counter, defaultdict
import math

MODES = ('pitch', 'image')
HOLDER_ROLES = ('player', 'goalkeeper')
# Goal mouth in metres: half the distance between the posts, and the crossbar height.
POST, BAR = 3.66, 2.44


def to_frames(seconds, fps):
    return round(seconds * fps)


def feet(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2)/2, y2), y2 - y1


def centre(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2)/2, (y1 + y2)/2


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown mode {mode!r}, expected one of {MODES}')


def nearest(frame, mode='pitch'):
    check_mode(mode)
    ball = frame['ball']
    if ball is None:
        return None, None
    best, distance = None, None
    for p in frame['players']:
        if p['role'] not in HOLDER_ROLES or p['position'] is None:
            continue
        d = math.hypot(p['position'][0]-ball[0], p['position'][1]-ball[1])
        if mode == 'image':
            d /= p['height']
        if distance is None or d < distance:
            best, distance = p['id'], d
    return best, distance


def holder(frame, radius, mode='pitch', max_z=None):
    """Nearest player within `radius` (metres or player heights); no holder above `max_z` metres."""
    player, distance = nearest(frame, mode)
    if distance is None or distance > radius:
        return None
    z = frame['ball'][2] if len(frame['ball']) > 2 else None
    return None if max_z is not None and z is not None and z > max_z else player


def holders(frames, radius, mode='pitch', max_z=None):
    return [(f['frame'], f['period'], holder(f, radius, mode, max_z)) for f in frames]


def known_teams(frames):
    return {p['id']: p['team'] for f in frames for p in f['players'] if p.get('team') is not None}


def interpolate_ball(frames, max_gap, fps):
    """Fill ball gaps of at most `max_gap` seconds linearly inside one period."""
    limit = to_frames(max_gap, fps)
    out = [dict(f, interpolated=False) for f in frames]
    last = None
    for i, f in enumerate(out):
        if f['ball'] is None:
            continue
        if last is not None and i > last + 1:
            a, span = out[last], f['frame'] - out[last]['frame']
            same = {g['period'] for g in out[last:i+1]} == {a['period']}
            if a['period'] is not None and same and span - 1 <= limit:
                for g in out[last+1:i]:
                    t = (g['frame'] - a['frame']) / span
                    g['ball'] = tuple(None if u is None or v is None else u + (v-u)*t
                                      for u, v in zip(a['ball'], f['ball']))
                    g['interpolated'] = True
        last = i
    return out


def possessions(owners, hold, fps):
    """The ball changes owner only after `hold` seconds of the same nearest player."""
    hold = to_frames(hold, fps)
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


def passes(segments, teams, max_gap, fps):
    """Holder changes within `max_gap` seconds: pass, turnover, or transfer when a team is unknown."""
    limit, teams, found = to_frames(max_gap, fps), teams or {}, []
    for a, b in zip(segments, segments[1:]):
        if a['period'] != b['period'] or b['start'] - a['end'] > limit:
            continue
        team_a, team_b = teams.get(a['player']), teams.get(b['player'])
        kind = 'transfer' if team_a is None or team_b is None else 'pass' if team_a == team_b else 'turnover'
        found.append(dict(frame=b['start'], passer=a['player'], receiver=b['player'], team=team_a,
                          kind=kind, gap=b['start'] - a['end']))
    return found


def pass_counts(found):
    teams, players = defaultdict(Counter), defaultdict(Counter)
    for t in found:
        teams[t['team']][t['kind']] += 1
        players[t['passer']][t['kind']] += 1
        if t['kind'] == 'pass':
            players[t['receiver']]['received'] += 1
    return dict(teams={k: dict(v) for k, v in teams.items()}, players={k: dict(v) for k, v in players.items()})


def shares(counts):
    counts = {k: v for k, v in counts.items() if k is not None}
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items(), key=str)} if total else {}


def holding(owners, segments, teams):
    """Team shares: `held` counts frames with a holder, `last_holder` also the ball in flight after them."""
    held, kept, i = Counter(), Counter(), 0
    for frame, period, player in owners:
        while i < len(segments) and segments[i]['start'] <= frame:
            i += 1
        active = segments[i-1] if i else None
        if active is None or period is None or active['period'] != period:
            continue
        team = teams.get(active['player'])
        kept[team] += 1
        held[team] += player == active['player']
    return dict(held=shares(held), last_holder=shares(kept))


def inside(point, polygon):
    x, y, polygon, hit = point[0], point[1], list(polygon), False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def goal_side(frame, mode='pitch', length=105.0, post=POST, bar=BAR):
    """Side whose goal mouth holds the ball: pitch beyond the line, image inside a `goals` polygon."""
    check_mode(mode)
    ball = frame['ball']
    if ball is None:
        return None
    if mode == 'image':
        return next((side for side, polygon in (frame.get('goals') or {}).items()
                     if polygon and inside(ball, polygon)), None)
    z = ball[2] if len(ball) > 2 else None
    if abs(ball[0]) > length/2 and abs(ball[1]) <= post and (z is None or z <= bar):
        return 'left' if ball[0] < 0 else 'right'
    return None


def goal_mouth_events(frames, fps, mode='pitch', length=105.0, min_duration=0.0, post=POST, bar=BAR):
    """Consecutive goal-mouth frames merged into events lasting at least `min_duration` seconds."""
    events, current = [], None
    for f in frames:
        side = goal_side(f, mode, length, post, bar)
        follows = current is not None and f['period'] == current['period'] and f['frame'] == current['end'] + 1
        if follows and side == current['side']:
            current['end'] = f['frame']
        elif side:
            current = dict(side=side, period=f['period'], start=f['frame'], end=f['frame'])
            events.append(current)
        else:
            current = None
    limit = to_frames(min_duration, fps)
    return [dict(e, seconds=(e['end'] - e['start'] + 1) / fps) for e in events
            if e['end'] - e['start'] + 1 >= limit]
