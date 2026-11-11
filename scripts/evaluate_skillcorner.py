"""SkillCorner check of the analytics rules: passes, holding share, goals and the ball recall they need."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import csv
import json
import platform
import random
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports_analyzer.analytics import to_frames
from sports_analyzer.dataset_preparation import git_state, sha256
from sports_analyzer.skillcorner import FPS, match_files, read_json
from sports_analyzer.skillcorner_eval import (attempted_counts, ball_stats, ball_table, ball_tables, ball_window,
                                              check_split, context, date_split, degrade, goal_truth, holding_shares,
                                              load_match, one_to_one, periods, rule_goals, rule_passes, same_players,
                                              scores, shifted, truth_passes, truth_shares, unflagged_goals)

RULE = ('radius', 'hold', 'max_z', 'ball_gap')
DEFINITIONS = ('held', 'last_holder', 'segments')


def write_csv(path, rows, columns=None):
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def prepare(match_dir, gaps, settings, goal, split):
    data = load_match(match_dir)
    match, frames = data['match'], data['frames']
    data.update(spans=periods(match), tables=ball_tables(frames, gaps), shares=truth_shares(data['groups'], match),
                truth=truth_passes(data['events'], match, set(settings['set_pieces'])),
                attempted=attempted_counts(data['events'], match), goals=goal_rows(data, settings, goal))
    play = [f for f in frames if f['period'] is not None]
    truth = [t for t in data['truth'] if t['completed']]
    data['facts'] = dict(match_id=data['id'], date_utc=match['date_time'], split=split,
                         home=match['home_team']['name'], away=match['away_team']['name'],
                         score=f'{match["home_team_score"]}-{match["away_team_score"]}',
                         pitch_length=match['pitch_length'], play_frames=len(play),
                         ball_share=sum(f['ball'] is not None for f in play) / len(play),
                         detected_share=sum(f['ball'] is not None and f['detected'] for f in play) / len(play),
                         attempted_passes=sum(data['attempted'].values()), completed_passes=len(truth),
                         completed_target_is_receiver=sum(t['target_is_receiver'] for t in truth),
                         set_piece_passes=sum(t['set_piece'] for t in truth),
                         flagged_goals=len(data['goals']['truth']) // 2, group_home_share=data['shares']['group'],
                         tip_home_share=data['shares']['minutes_tip'])
    return data


def goal_rows(data, settings, goal):
    match, frames, events = data['match'], data['frames'], data['events']
    length, window = match['pitch_length'], [to_frames(s, FPS) for s in settings['goal_window']]
    index = {f['frame']: i for i, f in enumerate(frames)}
    truth, alarms, rows = goal_truth(events, match), [], []
    for ball in ('all', 'detected'):
        found = rule_goals(frames, goal, length, ball == 'detected')
        pairs = dict((j, i) for i, j in one_to_one(found, truth, *window))
        for j, t in enumerate(truth):
            hit = found[pairs[j]] if j in pairs else None
            rows.append(dict(match_id=data['id'], ball=ball, frame=t['frame'], period=t['period'], team=t['team'],
                             flag=t['flag'], end_type=t['end_type'], expected_side=t['side'], hit=hit is not None,
                             offset_s=(hit['start'] - t['frame']) / FPS if hit else None,
                             rule_side=hit['side'] if hit else None,
                             side_agrees=hit['side'] == t['side'] if hit else None,
                             **ball_window(frames, t['frame'] + window[0], t['frame'] + window[1], length)))
        matched = set(pairs.values())
        for i, e in enumerate(found):
            inside = [frames[index[n]] for n in range(e['start'], e['end'] + 1) if n in index]
            alarms.append(dict(match_id=data['id'], ball=ball, start=e['start'], end=e['end'], period=e['period'],
                               side=e['side'], seconds=e['seconds'], matched=i in matched,
                               detected_share=sum(f['detected'] for f in inside) / len(inside),
                               **context(events, e['start'], match)))
    unflagged = [dict(match_id=data['id'], **u) for u in unflagged_goals(events, match)]
    return dict(truth=rows, alarms=alarms, unflagged=unflagged)


def pass_rows(data, split, name, rule, found, settings):
    tolerance = to_frames(settings['tolerance'], FPS)
    truths = dict(completed=[t for t in data['truth'] if t['completed']], attempted=data['truth'])
    rows = []
    for truth_set, truth in truths.items():
        trials = [('strict', None, found), ('time', None, found)]
        trials += [(f'chance_{m}', seed, shifted(found, data['spans'], random.Random(f'{seed}-{data["id"]}')))
                   for seed in settings['chance_seeds'] for m in ('strict', 'time')]
        for method, seed, items in trials:
            matched = one_to_one(items, truth, -tolerance, tolerance, same_players if method.endswith('strict') else None)
            rows.append(dict(match_id=data['id'], split=split, rule=name, truth_set=truth_set, method=method, seed=seed,
                             found=len(items), truth=len(truth), matched=len(matched),
                             truth_set_piece=sum(t['set_piece'] for t in truth),
                             matched_set_piece=sum(truth[j]['set_piece'] for _, j in matched)))
    return rows


def evaluate(data, split, name, rule, settings, max_gap):
    owners, segments, found = rule_passes(data['tables'][rule['ball_gap']], data['teams'], rule, max_gap)
    counts, completed = Counter(t['team'] for t in found), Counter(t['team'] for t in data['truth'] if t['completed'])
    count_rows = [dict(match_id=data['id'], split=split, rule=name, team=team, attempted=data['attempted'][team],
                       completed=completed[team], found=counts[team], error=counts[team] - completed[team],
                       relative_error=(counts[team] - completed[team]) / completed[team] if completed[team] else None)
                  for team in ('home', 'away')]
    shares, holding_rows = holding_shares(owners, segments, data['teams']), []
    for definition in DEFINITIONS:
        value = shares[definition]
        holding_rows.append(dict(match_id=data['id'], split=split, rule=name, definition=definition, home_share=value,
                                 group=data['shares']['group'], minutes_tip=data['shares']['minutes_tip'],
                                 **{f'error_{k}_pp': None if value is None else 100*abs(value - v)
                                    for k, v in data['shares'].items()}))
    return pass_rows(data, split, name, rule, found, settings), count_rows, holding_rows


def pooled(rows, keys):
    groups = defaultdict(Counter)
    for r in rows:
        groups[tuple(r[k] for k in keys)].update({k: r[k] for k in ('found', 'truth', 'matched', 'truth_set_piece',
                                                                       'matched_set_piece')})
    out = []
    for key, c in groups.items():
        open_truth = c['truth'] - c['truth_set_piece']
        out.append(dict(zip(keys, key), **scores(c['found'], c['truth'], c['matched']),
                        recall_set_piece=c['matched_set_piece'] / c['truth_set_piece'] if c['truth_set_piece'] else None,
                        recall_open_play=(c['matched'] - c['matched_set_piece']) / open_truth if open_truth else None,
                        truth_set_piece=c['truth_set_piece']))
    return out


def pass_summary(rows):
    out = []
    for split in ('tune', 'test', 'all'):
        chosen = [r for r in rows if split == 'all' or r['split'] == split]
        for r in pooled(chosen, ('rule', 'truth_set', 'method', 'seed')):
            out.append(dict(split=split, **r))
    summary = []
    for key in sorted({(r['split'], r['rule'], r['truth_set'], r['method']) for r in out}):
        items = [r for r in out if (r['split'], r['rule'], r['truth_set'], r['method']) == key]
        summary.append(dict(zip(('split', 'rule', 'truth_set', 'method'), key), seeds=len(items),
                            **{k: mean(r[k] for r in items) for k in ('found', 'truth', 'matched', 'precision',
                                                                      'recall', 'f1', 'recall_set_piece',
                                                                      'recall_open_play', 'truth_set_piece')},
                            f1_min=min(r['f1'] for r in items), f1_max=max(r['f1'] for r in items)))
    return summary


def error_summary(rows, keys, columns):
    out = []
    for split in ('tune', 'test', 'all'):
        groups = defaultdict(list)
        for r in rows:
            if split == 'all' or r['split'] == split:
                groups[tuple(r[k] for k in keys)].append(r)
        for key, items in sorted(groups.items(), key=str):
            row = dict(split=split, **dict(zip(keys, key)), n=len(items))
            for c in columns:
                values = [abs(r[c]) for r in items if r[c] is not None]
                row.update({f'{c}_mean_abs': mean(values), f'{c}_max_abs': max(values, default=None)})
            row.update({f'{c}_mean': mean(r[c] for r in items) for c in columns})
            out.append(row)
    return out


def grid_rows(data, rules, settings, max_gap):
    tolerance = to_frames(settings['tolerance'], FPS)
    truth = [t for t in data['truth'] if t['completed']]
    rows = []
    for rule in rules:
        _, _, found = rule_passes(data['tables'][rule['ball_gap']], data['teams'], rule, max_gap)
        rows.append(dict(match_id=data['id'], **rule, found=len(found), truth=len(truth),
                         strict=len(one_to_one(found, truth, -tolerance, tolerance, same_players)),
                         time=len(one_to_one(found, truth, -tolerance, tolerance))))
    return rows


def degrade_rows(data, rule, settings, max_gap):
    spec, frames, tolerance = settings['degrade'], data['frames'], to_frames(settings['tolerance'], FPS)
    truth = [t for t in data['truth'] if t['completed']]
    completed = Counter(t['team'] for t in truth)
    length, width, clean, rows = data['match']['pitch_length'], data['match']['pitch_width'], data['tables'][0.0], []
    for recall, run, rate in product(spec['recall'], spec['run'], spec['false_rate']):
        rng = random.Random(f'{spec["seed"]}-{data["id"]}-{recall}-{run}-{rate}')
        degraded = degrade(frames, recall, run, rate, rng, length, width)
        stats = ball_stats(frames, degraded)
        for gap in spec['ball_gaps']:
            owners, segments, found = rule_passes(ball_table(degraded, gap, clean), data['teams'], rule, max_gap)
            counts, shares = Counter(t['team'] for t in found), holding_shares(owners, segments, data['teams'])
            rows.append(dict(match_id=data['id'], recall=recall, run=run, false_rate=rate, ball_gap=gap,
                             measured_recall=stats['recall'], measured_false=stats['false_share'],
                             measured_run=stats['miss_run_mean'], found=len(found), truth=len(truth),
                             matched=len(one_to_one(found, truth, -tolerance, tolerance, same_players)),
                             count_error=mean(abs(counts[k] - completed[k]) for k in ('home', 'away')),
                             count_relative_error=mean(abs(counts[k] - completed[k]) / completed[k]
                                                       for k in ('home', 'away') if completed[k]),
                             **{f'{d}_error_pp': None if shares[d] is None else 100*abs(shares[d] - data['shares']['group'])
                                for d in DEFINITIONS}))
    return rows


def tuning_table(rows):
    groups = defaultdict(Counter)
    for r in rows:
        groups[tuple(r[k] for k in RULE)].update({k: r[k] for k in ('found', 'truth', 'strict', 'time')})
    out = []
    for key, c in groups.items():
        strict, loose = scores(c['found'], c['truth'], c['strict']), scores(c['found'], c['truth'], c['time'])
        out.append(dict(zip(RULE, key), found=c['found'], truth=c['truth'],
                        **{f'strict_{k}': strict[k] for k in ('matched', 'precision', 'recall', 'f1')},
                        **{f'time_{k}': loose[k] for k in ('matched', 'precision', 'recall', 'f1')}))
    return out


def degrade_table(rows):
    keys = ('recall', 'run', 'false_rate', 'ball_gap')
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, items in sorted(groups.items()):
        total = scores(*(sum(r[k] for r in items) for k in ('found', 'truth', 'matched')))
        out.append(dict(zip(keys, key), matches=len(items), strict_precision=total['precision'],
                        strict_recall=total['recall'], strict_f1=total['f1'],
                        **{k: mean(r[k] for r in items) for k in ('measured_recall', 'measured_false', 'measured_run',
                                                                   'count_error', 'count_relative_error',
                                                                   *(f'{d}_error_pp' for d in DEFINITIONS))}))
    return out


def half_recall(cells):
    """Ball recall where strict pass F1 falls to half its value at full recall, linear between grid points."""
    cells = sorted(cells, key=lambda c: -c['recall'])
    half = cells[0]['strict_f1'] / 2
    for a, b in zip(cells, cells[1:]):
        if b['strict_f1'] <= half:
            step = b['strict_f1'] - a['strict_f1']
            return b['recall'] if not step else a['recall'] + (half - a['strict_f1']) * (b['recall'] - a['recall']) / step
    return None


def goal_summary(truth, alarms, unflagged, sample, seed):
    out = {}
    for ball in ('all', 'detected'):
        rows, found = [r for r in truth if r['ball'] == ball], [a for a in alarms if a['ball'] == ball]
        hits = [r for r in rows if r['hit']]
        false = [a for a in found if not a['matched']]
        out[ball] = dict(truth=len(rows), rule_events=len(found), hits=len(hits), misses=len(rows) - len(hits),
                         false_alarms=len(false), side_agrees=sum(bool(r['side_agrees']) for r in hits),
                         false_alarm_what=dict(Counter(a['what'] for a in false).most_common()),
                         event_frames_detected_share=sum(a['detected_share'] * a['seconds'] for a in found)
                         / sum(a['seconds'] for a in found) if found else None)
    false = [a for a in alarms if a['ball'] == 'all' and not a['matched']]
    out['misses'] = [r for r in truth if r['ball'] == 'all' and not r['hit']]
    out['false_alarm_sample'] = random.Random(seed).sample(false, min(sample, len(false)))
    out['unflagged'] = unflagged
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'data/raw/skillcorner_opendata')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/analytics.yaml')
    parser.add_argument('--runs', type=Path, default=ROOT / 'outputs/skillcorner_eval')
    args = parser.parse_args()
    began = time.time()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    settings, pitch, goal = config['skillcorner'], config['pitch'], config['goal']
    folder = args.source / 'data/matches'
    dates = {int(p.name): read_json(match_files(p)['match'])['date_time'] for p in folder.iterdir() if p.is_dir()}
    tune, test = check_split(settings['tune'], settings['test'], dates)
    by_date = date_split({m: dates[m] for m in tune + test}, len(tune))
    target = args.runs / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)

    default = {k: pitch[k] for k in RULE}
    named = dict(audit=dict(default, ball_gap=0.0), default=default)
    grid = [dict(zip(RULE, values)) for values in product(*(settings['grid'][k] for k in RULE))]
    gaps = sorted(set(settings['grid']['ball_gap']) | {default['ball_gap']})
    cache, grid_out = {}, []
    for match_id in tune:
        data = prepare(folder / str(match_id), gaps, settings, goal, 'tune')
        grid_out += grid_rows(data, grid, settings, pitch['max_gap'])
        data.pop('frames')
        cache[match_id] = data
        print(f'{match_id}: tune, {len(grid)} rules, {time.time() - began:.0f} s', flush=True)
    tuning = sorted(tuning_table(grid_out), key=lambda r: -r['strict_f1'])
    named['tuned'] = {k: tuning[0][k] for k in RULE}

    pass_out, count_out, holding_out, degrade_out = [], [], [], []
    for match_id in tune + test:
        split = 'tune' if match_id in tune else 'test'
        data = cache.get(match_id) or prepare(folder / str(match_id), gaps, settings, goal, split)
        for name, rule in named.items():
            passes, counts, holding = evaluate(data, split, name, rule, settings, pitch['max_gap'])
            pass_out += passes
            count_out += counts
            holding_out += holding
        if split == 'test':
            degrade_out += degrade_rows(data, named['tuned'], settings, pitch['max_gap'])
            data.pop('frames')
            cache[match_id] = data
            print(f'{match_id}: test, degraded, {time.time() - began:.0f} s', flush=True)

    facts = [cache[m]['facts'] for m in tune + test]
    goal_list = [r for m in tune + test for r in cache[m]['goals']['truth']]
    alarms = [r for m in tune + test for r in cache[m]['goals']['alarms']]
    unflagged = [r for m in tune + test for r in cache[m]['goals']['unflagged']]
    passes = pass_summary(pass_out)
    counts = error_summary(count_out, ('rule',), ('error', 'relative_error'))
    holding = error_summary(holding_out, ('rule', 'definition'), ('error_group_pp', 'error_minutes_tip_pp'))
    degraded = degrade_table(degrade_out)
    series = defaultdict(list)
    for r in degraded:
        series[(r['run'], r['false_rate'], r['ball_gap'])].append(r)
    halves = [dict(run=k[0], false_rate=k[1], ball_gap=k[2], f1_full=max(c['strict_f1'] for c in v if c['recall'] == 1.0),
                   half_f1_recall=half_recall(v)) for k, v in sorted(series.items())]
    goals = goal_summary(goal_list, alarms, unflagged, settings['false_alarm_sample'], settings['degrade']['seed'])
    final = sum(int(s) for f in facts for s in f['score'].split('-'))

    write_csv(target / 'matches.csv', facts)
    write_csv(target / 'tuning.csv', tuning)
    write_csv(target / 'passes_by_match.csv', pass_out)
    write_csv(target / 'passes.csv', passes)
    write_csv(target / 'pass_counts_by_match.csv', count_out)
    write_csv(target / 'pass_counts.csv', counts)
    write_csv(target / 'holding_by_match.csv', holding_out)
    write_csv(target / 'holding.csv', holding)
    write_csv(target / 'goals.csv', goal_list)
    write_csv(target / 'goal_mouth_events.csv', alarms)
    write_csv(target / 'unflagged_goals.csv', unflagged, ['match_id', 'team', 'score_after', 'period', 'frame_from',
                                                          'frame_to', 'last_end_type', 'last_interruption_after',
                                                          'next_interruption_before'])
    write_csv(target / 'degrade_by_match.csv', degrade_out)
    write_csv(target / 'degrade.csv', degraded)
    write_csv(target / 'degrade_half_f1.csv', halves)
    headline = [{k: r[k] for k in ('rule', 'method', 'precision', 'recall', 'f1', 'f1_min', 'f1_max',
                                   'recall_set_piece', 'recall_open_play')}
                for r in passes if r['split'] == 'test' and r['truth_set'] == 'completed']
    provider = {k: sum(f[k] for f in facts) for k in ('attempted_passes', 'completed_passes',
                                                       'completed_target_is_receiver', 'set_piece_passes')}
    provider['pass_definition'] = (
        'player_possession row with end_type pass; pass_outcome successful, unsuccessful or offside; receiver is '
        'the next player_possession in the period and its frame_start is the pass frame; completed means successful '
        'to a different teammate; set piece means the passer possession has a restart start_type')
    summary = dict(split=dict(tune=tune, test=test, date_order=by_date == (tune, test)), rules=named,
                   tuning_best=tuning[:5], passes_test=headline,
                   pass_counts=[r for r in counts if r['split'] in ('test', 'all')],
                   holding=[r for r in holding if r['split'] in ('test', 'all')],
                   goals=dict(flagged=sum(f['flagged_goals'] for f in facts), final_score_goals=final,
                              window_s=settings['goal_window'], **goals),
                   degrade_half_f1=halves, provider=provider)
    (target / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    commit = subprocess.run(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], capture_output=True,
                            text=True).stdout.strip() or None
    outputs = sorted(p.name for p in target.iterdir())
    manifest = dict(source=str(args.source.resolve()), source_commit=commit, config=str(args.config),
                    config_sha256=sha256(args.config), settings=settings, pitch=pitch, goal=goal,
                    outputs={name: sha256(target / name) for name in outputs},
                    arguments={k: str(v) for k, v in vars(args).items()}, git=git_state(),
                    python=platform.python_version(), device='cpu', platform=platform.platform(),
                    script_sha256=sha256(Path(__file__)),
                    module_sha256={name: sha256(ROOT / f'sports_analyzer/{name}.py')
                                   for name in ('skillcorner_eval', 'skillcorner', 'analytics')},
                    seconds=round(time.time() - began, 1), created_utc=datetime.now(timezone.utc).isoformat())
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=str(target), tuned=named['tuned'], passes_test=headline, goals=goals['all'],
                          degrade_half_f1=halves), indent=2, default=str))


if __name__ == '__main__':
    main()
