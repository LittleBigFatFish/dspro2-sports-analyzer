"""SkillCorner evaluation of the analytics rules: passes, holding share, goals and a degraded ball track."""
from bisect import bisect_left, bisect_right
from collections import Counter
import math

from sports_analyzer import analytics
from sports_analyzer.skillcorner import (FPS, analytics_frame, iter_frames, match_files, player_teams,
                                         possession_share, read_csv, read_json)

GOAL_FLAGS = ('goal_for', 'goal_against')


def check_split(tune, test, available=None):
    tune, test = [int(m) for m in tune], [int(m) for m in test]
    if len(set(tune)) != len(tune) or len(set(test)) != len(test):
        raise ValueError('Split lists a match twice')
    if set(tune) & set(test):
        raise ValueError(f'Matches in both tune and test: {sorted(set(tune) & set(test))}')
    if available is not None and set(tune + test) - set(available):
        raise ValueError(f'Unknown matches: {sorted(set(tune + test) - set(available))}')
    return tune, test


def date_split(dates, tune):
    """Match ids sorted by kick-off: the first `tune` to tune, the rest held out."""
    ordered = [m for m, _ in sorted(dates.items(), key=lambda item: (item[1], item[0]))]
    return ordered[:tune], ordered[tune:]


def load_match(match_dir):
    paths = match_files(match_dir)
    match = read_json(paths['match'])
    teams, frames, groups = player_teams(match), [], Counter()
    for raw in iter_frames(paths['tracking']):
        frame = analytics_frame(raw, teams)
        frame['detected'] = bool(raw['ball_data']['is_detected'])
        frames.append(frame)
        if raw['period'] is not None:
            groups[raw['possession']['group']] += 1
    return dict(id=match['id'], match=match, teams=teams, frames=frames, groups=groups,
                events=read_csv(paths['events']))


def periods(match):
    return {p['period']: (p['start_frame'], p['end_frame']) for p in match['match_periods']}


def side_of(team_id, match):
    return 'home' if int(team_id) == match['home_team']['id'] else 'away'


def possession_rows(events):
    return sorted((r for r in events if r['event_type'] == 'player_possession'),
                  key=lambda r: (int(r['period']), int(r['frame_start'])))


def nearest_table(frames, known=None):
    """Nearest player per frame; rows of `known` are reused while the frame keeps the same ball object."""
    rows = []
    for i, f in enumerate(frames):
        if known is not None and known[i][4] is f['ball']:
            rows.append(known[i])
            continue
        player, distance = analytics.nearest(f)
        rows.append((f['frame'], f['period'], player, distance, f['ball']))
    return rows


def table_owners(rows, radius, max_z=None):
    # Same result as analytics.holders on the frames behind the rows.
    out = []
    for frame, period, player, distance, ball in rows:
        if distance is None or distance > radius:
            player = None
        elif max_z is not None and len(ball) > 2 and ball[2] is not None and ball[2] > max_z:
            player = None
        out.append((frame, period, player))
    return out


def ball_table(frames, gap, known=None):
    return nearest_table(analytics.interpolate_ball(frames, gap, FPS) if gap else frames, known)


def ball_tables(frames, gaps):
    clean = nearest_table(frames)
    return {0.0: clean, **{g: ball_table(frames, g, clean) for g in gaps if g}}


def rule_passes(rows, teams, rule, max_gap):
    owners = table_owners(rows, rule['radius'], rule['max_z'])
    segments = analytics.possessions(owners, rule['hold'], FPS)
    found = [t for t in analytics.passes(segments, teams, max_gap, FPS) if t['kind'] == 'pass']
    return owners, segments, found


def truth_passes(events, match, set_pieces):
    """Possessions ending in a pass, with the next possession as receiver; completed = successful to a teammate."""
    rows, out = possession_rows(events), []
    for a, b in zip(rows, rows[1:]):
        if a['period'] != b['period'] or a['end_type'] != 'pass':
            continue
        completed = a['pass_outcome'] == 'successful' and a['team_id'] == b['team_id'] and a['player_id'] != b['player_id']
        target = int(float(a['player_targeted_id'])) if a['player_targeted_id'] else None
        out.append(dict(frame=int(b['frame_start']), period=int(a['period']), passer=int(a['player_id']),
                        receiver=int(b['player_id']), team=side_of(a['team_id'], match), outcome=a['pass_outcome'],
                        completed=completed, set_piece=a['start_type'] in set_pieces, start_type=a['start_type'],
                        target_is_receiver=target == int(b['player_id'])))
    return out


def attempted_counts(events, match):
    return Counter(side_of(r['team_id'], match) for r in possession_rows(events) if r['end_type'] == 'pass')


def same_players(found, truth):
    return found['passer'] == truth['passer'] and found['receiver'] == truth['receiver']


def one_to_one(found, truth, low, high, same=None):
    """Pairs with the found frame in truth frame + [low, high] frames, closest first, each item used once."""
    order = sorted(range(len(truth)), key=lambda j: truth[j]['frame'])
    frames = [truth[j]['frame'] for j in order]
    pairs = sorted((abs(f['frame'] - truth[order[k]]['frame']), i, order[k]) for i, f in enumerate(found)
                   for k in range(bisect_left(frames, f['frame'] - high), bisect_right(frames, f['frame'] - low))
                   if same is None or same(f, truth[order[k]]))
    used_found, used_truth, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_found and j not in used_truth:
            used_found.add(i)
            used_truth.add(j)
            matched.append((i, j))
    return matched


def scores(found, truth, matched):
    precision = matched / found if found else None
    recall = matched / truth if truth else None
    f1 = 2*precision*recall / (precision+recall) if precision and recall else 0.0
    return dict(found=found, truth=truth, matched=matched, precision=precision, recall=recall, f1=f1)


def shifted(found, spans, rng):
    """Found passes moved to random frames inside their own period."""
    out = []
    for t in found:
        start, end = next(s for s in spans.values() if s[0] <= t['frame'] <= s[1])
        out.append(dict(t, frame=rng.randint(start, end)))
    return out


def home_share(shares):
    return shares.get('home', 0.0) if shares else None


def holding_shares(owners, segments, teams):
    held = analytics.holding(owners, segments, teams)
    return dict(held=home_share(held['held']), last_holder=home_share(held['last_holder']),
                segments=possession_share(segments, teams))


def truth_shares(groups, match):
    home, away = groups['home team'], groups['away team']
    tip = match['home_team_playing_time']['minutes_tip'], match['away_team_playing_time']['minutes_tip']
    return dict(group=home / (home + away) if home + away else None, minutes_tip=tip[0] / sum(tip))


def parts(total, count, rng):
    cuts = sorted(rng.sample(range(1, total), count - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def degrade(frames, recall, run, false_rate, rng, length=105.0, width=68.0, fps=FPS):
    """Ball misses in runs of mean `run` seconds down to `recall` of the in-play ball frames, then false balls
    at random pitch positions in `false_rate` of the in-play frames; frames outside play are left as they are."""
    play = [i for i, f in enumerate(frames) if f['period'] is not None]
    seen = [i for i in play if frames[i]['ball'] is not None]
    missed = round((1 - recall) * len(seen))
    runs = min(max(1, round(missed / max(1, analytics.to_frames(run, fps)))), missed, len(seen) - missed + 1)
    out = list(frames)
    if missed:
        gaps = parts(len(seen) - missed + 2, runs + 1, rng)
        gaps[0], gaps[-1] = gaps[0] - 1, gaps[-1] - 1
        position = 0
        for gap, size in zip(gaps, parts(missed, runs, rng)):
            position += gap
            for i in seen[position:position + size]:
                out[i] = dict(out[i], ball=None)
            position += size
    for i in rng.sample(play, round(false_rate * len(play))):
        out[i] = dict(out[i], ball=(rng.uniform(-length/2, length/2), rng.uniform(-width/2, width/2), None))
    return out


def ball_stats(frames, degraded):
    play = [(f, g) for f, g in zip(frames, degraded) if f['period'] is not None]
    seen = [(f, g) for f, g in play if f['ball'] is not None]
    kept = [g['ball'] is f['ball'] for f, g in seen]
    runs, length = [], 0
    for k in kept + [True]:
        if not k:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    return dict(recall=sum(kept) / len(seen) if seen else None,
                false_share=sum(g['ball'] is not None and g['ball'] is not f['ball'] for f, g in play) / len(play),
                miss_run_mean=sum(runs) / len(runs) / FPS if runs else 0.0)


def goal_side_expected(row):
    # The possession's attacking direction: left_to_right attacks the goal at positive x.
    return 'right' if row['attacking_side'] == 'left_to_right' else 'left'


def goal_truth(events, match):
    """Flagged goals at the end frame of the flagged possession; the scorer is the team in possession."""
    return [dict(frame=int(r['frame_end']), period=int(r['period']), team=side_of(r['team_id'], match),
                 player=int(r['player_id']), flag=r['game_interruption_after'], end_type=r['end_type'],
                 side=goal_side_expected(r))
            for r in possession_rows(events) if r['game_interruption_after'] in GOAL_FLAGS]


def score_changes(events, match):
    """Goals read from the running score columns: possession pairs where a team's score goes up."""
    rows, out, last = possession_rows(events), [], None
    for r in rows:
        home = side_of(r['team_id'], match) == 'home'
        own, other = int(float(r['team_score'] or 0)), int(float(r['opponent_team_score'] or 0))
        score = (own, other) if home else (other, own)
        if last is not None:
            for k, team in enumerate(('home', 'away')):
                if score[k] > last[1][k]:
                    out.append(dict(team=team, before=last[0], after=r, score='-'.join(map(str, score))))
        last = (r, score)
    final = match['home_team_score'], match['away_team_score']
    for k, team in enumerate(('home', 'away')):
        if last is not None and final[k] > last[1][k]:
            out.append(dict(team=team, before=last[0], after=None, score=f'{final[0]}-{final[1]}'))
    return out


def unflagged_goals(events, match):
    """Score changes without a flagged goal just before them, or on the next possession of the scoring team."""
    out = []
    for change in score_changes(events, match):
        before, after = change['before'], change['after']
        late = after and after['game_interruption_after'] in GOAL_FLAGS and side_of(after['team_id'], match) == change['team']
        if before['game_interruption_after'] in GOAL_FLAGS or late:
            continue
        out.append(dict(team=change['team'], score_after=change['score'], period=int(before['period']),
                        frame_from=int(before['frame_end']),
                        frame_to=int(after['frame_start']) if after else None,
                        last_end_type=before['end_type'], last_interruption_after=before['game_interruption_after'],
                        next_interruption_before=after['game_interruption_before'] if after else None))
    return out


def rule_goals(frames, goal, length, detected_only=False):
    if detected_only:
        frames = [f if f['detected'] or f['ball'] is None else dict(f, ball=None) for f in frames]
    events = analytics.goal_mouth_events(frames, FPS, 'pitch', length, goal['min_duration'], goal['post'], goal['bar'])
    return [dict(e, frame=e['start']) for e in events if e['period'] is not None]


def roles(match):
    return {p['id']: p['player_role']['acronym'] for p in match['players']}


def what_happened(last, after, positions):
    if last is None:
        return 'no possession before'
    end, stop = last['end_type'], last['game_interruption_after']
    keeper = after is not None and after['team_id'] != last['team_id'] and positions.get(int(after['player_id'])) == 'GK'
    if end == 'shot':
        if stop in GOAL_FLAGS:
            return 'after a flagged goal'
        if stop.startswith('goal_kick'):
            return 'shot wide'
        if stop.startswith('corner'):
            return 'shot to corner'
        return 'shot saved' if keeper else 'shot, play on'
    if stop.startswith('goal_kick') or stop.startswith('corner'):
        return f'{end}, out for {stop.rsplit("_", 1)[0].replace("_", " ")}'
    return f'{end}, keeper takes it' if keeper else f'{end}{", " + stop if stop else ""}'


def context(events, frame, match):
    """What the events say around `frame`: the last possession started before it and the one after."""
    rows = possession_rows(events)
    starts = [int(r['frame_start']) for r in rows]
    i = bisect_right(starts, frame)
    last, after = rows[i-1] if i else None, rows[i] if i < len(rows) else None
    return dict(what=what_happened(last, after, roles(match)),
                last_team=side_of(last['team_id'], match) if last else None,
                last_end_type=last['end_type'] if last else None,
                last_frame_end=int(last['frame_end']) if last else None,
                last_interruption_after=last['game_interruption_after'] if last else None,
                next_start_type=after['start_type'] if after else None,
                next_interruption_before=after['game_interruption_before'] if after else None)


def ball_window(frames, start, end, length):
    """Ball coverage in frames start..end and its closest point to a goal centre."""
    window = [f for f in frames if start <= f['frame'] <= end and f['period'] is not None]
    balls = [f for f in window if f['ball'] is not None]
    near = min(balls, key=lambda f: math.hypot(abs(f['ball'][0]) - length/2, f['ball'][1]), default=None)
    return dict(window_frames=len(window), ball_share=len(balls) / len(window) if window else None,
                detected_share=sum(f['detected'] for f in balls) / len(window) if window else None,
                closest_beyond_line_m=abs(near['ball'][0]) - length/2 if near else None,
                closest_y=near['ball'][1] if near else None, closest_z=near['ball'][2] if near else None,
                closest_detected=near['detected'] if near else None)
