"""Data-sheet audit of SkillCorner open data plus a possession/pass check."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import json
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports_analyzer.dataset_preparation import git_state, sha256
from sports_analyzer.skillcorner import (FPS, event_transitions, iter_frames, match_files, match_transitions,
                                         nearest_owners, player_teams, possession_share, possessions,
                                         provided_owners, read_csv, read_json, summarise_events,
                                         summarise_phases, summarise_tracking, transitions)

LFS_PREFIX = b'version https://git-lfs'
RADII, HOLDS = (1.0, 1.5, 2.0, 3.0), (1, 3, 5)
DEFAULT = dict(radius=1.5, hold=3, max_z=None)


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def list_files(source):
    rows = []
    for path in sorted(p for p in source.rglob('*') if p.is_file() and '.git' not in p.relative_to(source).parts):
        with path.open('rb') as stream:
            head = stream.read(len(LFS_PREFIX))
        rows.append(dict(path=path.relative_to(source).as_posix(), bytes=path.stat().st_size,
                         sha256=sha256(path), lfs_pointer=head == LFS_PREFIX))
    return rows


def match_facts(match):
    edition = match['competition_edition']
    periods = {p['period']: p for p in match['match_periods']}
    sides = Counter(p['team_id'] for p in match['players'])
    return dict(match_id=match['id'], date_utc=match['date_time'], competition=edition['competition']['name'],
                area=edition['competition']['area'], season=edition['season']['name'],
                round=match['competition_round']['name'], stadium=match['stadium']['name'],
                home=match['home_team']['name'], away=match['away_team']['name'],
                score=f'{match["home_team_score"]}-{match["away_team_score"]}',
                pitch_length=match['pitch_length'], pitch_width=match['pitch_width'],
                periods=len(periods), period_minutes='/'.join(f'{p["duration_minutes"]}' for p in periods.values()),
                home_players=sides[match['home_team']['id']], away_players=sides[match['away_team']['id']],
                referees=len(match['referees']), status=match['status'],
                home_tip_share=match['home_team_playing_time']['minutes_tip']
                / (match['home_team_playing_time']['minutes_tip'] + match['away_team_playing_time']['minutes_tip']))


def agreement(match_id, records, teams, truth):
    rows, segments_by_rule = [], {}
    rules = [dict(method='provided', radius=None, hold=1, max_z=None)]
    rules += [dict(method='nearest', radius=r, hold=h, max_z=None) for r in RADII for h in HOLDS]
    rules.append(dict(method='nearest', **{**DEFAULT, 'max_z': 2.0}))
    for rule in rules:
        owners = (provided_owners(records) if rule['method'] == 'provided'
                  else nearest_owners(records, rule['radius'], rule['max_z']))
        segments = possessions(owners, rule['hold'])
        found = transitions(segments, teams)
        segments_by_rule[tuple(rule.values())] = (segments, found)
        share = possession_share(segments, teams)
        for kind in ('pass', 'turnover'):
            rows.append(dict(match_id=match_id, **rule, possessions=len(segments), home_share=share,
                             **match_transitions(found, truth, kind)))
    return rows, segments_by_rule


def aggregates_summary(folder):
    out = {}
    for path in sorted(folder.glob('*.csv')):
        rows = read_csv(path)
        out[path.name] = dict(bytes=path.stat().st_size, rows=len(rows), columns=len(rows[0]) if rows else 0,
                              players=len({r['player_id'] for r in rows}),
                              duplicate_rows=len(rows) - len({(r['player_id'], r['team_id'], r['position_group'])
                                                              for r in rows}),
                              teams=len({r['team_id'] for r in rows}),
                              seasons=sorted({r['season_name'] for r in rows}),
                              empty_cells=sum(v == '' for r in rows for v in r.values()),
                              column_names=list(rows[0]) if rows else [])
    return out


def bodypose_summary(folder):
    manifest = read_json(folder / 'MANIFEST.json')
    frames, players, with_joints = 0, 0, 0
    sample = folder / manifest['sample']['file']
    with gzip.open(sample, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                frame = json.loads(line)
                frames += 1
                players += len(frame['player_data'])
                with_joints += sum(p.get('joints') is not None for p in frame['player_data'])
    return dict(local_files={p.name: p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()},
                remote_dataset=manifest['hf_dataset'], remote_revision=manifest['hf_revision'],
                remote_files=manifest['files'], remote_matches=manifest['matches'],
                frame_rate_fps=manifest['frame_rate_fps'], sample=manifest['sample'],
                sample_counted=dict(frames=frames, player_rows=players, player_rows_with_joints=with_joints))


def total(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'data/raw/skillcorner_opendata')
    parser.add_argument('--source-json', type=Path, default=ROOT / 'data/raw/skillcorner_opendata.source.json')
    parser.add_argument('--runs', type=Path, default=ROOT / 'outputs/skillcorner_audit')
    parser.add_argument('--focus', type=int, default=1925299, help='match with detailed transition lists')
    args = parser.parse_args()
    began = time.time()
    source, data = args.source, args.source / 'data'
    target = args.runs / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target.mkdir(parents=True, exist_ok=False)

    files = list_files(source)
    commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True,
                            text=True).stdout.strip() or None
    index = read_json(data / 'matches.json')
    listed = {m['id'] for m in index}
    folders = {int(p.name) for p in (data / 'matches').iterdir() if p.is_dir()}
    matches, event_rows, agreement_rows, issue_rows, details = [], [], [], [], {}
    for match_id in sorted(listed & folders):
        paths = match_files(data / 'matches' / str(match_id))
        missing = [k for k, p in paths.items() if not p.is_file()]
        if missing:
            issue_rows.append(dict(match_id=match_id, file='-', code='missing_files', count=len(missing),
                                   detail=';'.join(missing)))
            continue
        match = read_json(paths['match'])
        teams = player_teams(match)
        facts = match_facts(match)
        tracking, records = summarise_tracking(iter_frames(paths['tracking']), teams,
                                               match['pitch_length'], match['pitch_width'])
        events = read_csv(paths['events'])
        event_summary = summarise_events(events, match['pitch_length'], match['pitch_width'],
                                         tracking['first_frame'], tracking['last_frame'])
        phases = summarise_phases(read_csv(paths['phases']))
        truth = event_transitions(events)
        rows, by_rule = agreement(match_id, records, teams, truth)
        agreement_rows += rows
        if match_id == args.focus:
            segments, found = by_rule[('nearest', DEFAULT['radius'], DEFAULT['hold'], None)]
            write_csv(target / 'focus_derived_transitions.csv', list(found[0]), found)
            write_csv(target / 'focus_event_transitions.csv', list(truth[0]), truth)
        for (kind, subtype), count in Counter((r['event_type'], r['event_subtype']) for r in events).most_common():
            event_rows.append(dict(match_id=match_id, event_type=kind, event_subtype=subtype, count=count))
        for name, issues in (('tracking', tracking['issues']), ('events', event_summary['issues'])):
            issue_rows += [dict(match_id=match_id, file=name, code=k, count=v, detail='') for k, v in issues.items()]
        default = {r['kind']: r for r in rows if r['method'] == 'nearest' and r['radius'] == DEFAULT['radius']
                   and r['hold'] == DEFAULT['hold'] and r['max_z'] is None}
        matches.append(dict(
            **facts, frames=tracking['frames'], period_frames=tracking['period_frames'],
            duration_minutes=tracking['period_minutes'], ball_xy_share=tracking['ball_xy_share'],
            ball_detected_share=tracking['ball_detected_share'],
            players_per_frame_mean=tracking['players_per_frame_mean'],
            players_per_frame_min=tracking['players_per_frame_min'], tracked_frames=tracking['tracked_frames'],
            players_per_tracked_frame_mean=tracking['players_per_tracked_frame_mean'],
            players_per_tracked_frame_min=tracking['players_per_tracked_frame_min'],
            detected_per_tracked_frame_mean=tracking['detected_per_tracked_frame_mean'],
            detected_per_tracked_frame_min=tracking['detected_per_tracked_frame_min'],
            detected_per_frame_mean=tracking['detected_per_frame_mean'],
            detected_per_frame_min=tracking['detected_per_frame_min'],
            player_extrapolated_share=tracking['player_extrapolated_share'],
            possession_player_share=tracking['possession_player_share'],
            events=event_summary['rows'], player_possessions=event_summary['event_types'].get('player_possession', 0),
            passes=event_summary['possession_end_types'].get('pass', 0),
            passes_successful=event_summary['pass_outcomes'].get('successful', 0),
            carries=event_summary['possession_flags']['carry'],
            on_ball_engagements=event_summary['event_types'].get('on_ball_engagement', 0),
            off_ball_runs=event_summary['event_types'].get('off_ball_run', 0),
            passing_options=event_summary['event_types'].get('passing_option', 0), phases=phases['rows'],
            event_passes_between_teammates=default['pass']['truth'],
            derived_passes=default['pass']['found'], derived_pass_precision=default['pass']['precision'],
            derived_pass_recall=default['pass']['recall'], derived_home_share=default['pass']['home_share']))
        details[match_id] = dict(facts=facts, tracking=tracking, events=event_summary, phases=phases)
        print(f'{match_id}: {tracking["frames"]} frames, {event_summary["rows"]} events, '
              f'pass P/R {default["pass"]["precision"]:.3f}/{default["pass"]["recall"]:.3f}', flush=True)

    write_csv(target / 'matches.csv', list(matches[0]), matches)
    write_csv(target / 'event_types.csv', ['match_id', 'event_type', 'event_subtype', 'count'], event_rows)
    write_csv(target / 'pass_agreement.csv', list(agreement_rows[0]), agreement_rows)
    write_csv(target / 'issues.csv', ['match_id', 'file', 'code', 'count', 'detail'], issue_rows)
    write_csv(target / 'files.csv', ['path', 'bytes', 'sha256', 'lfs_pointer'], files)

    sweep = {}
    for r in agreement_rows:
        key = f'{r["method"]}|r={r["radius"]}|hold={r["hold"]}|max_z={r["max_z"]}|{r["kind"]}'
        item = sweep.setdefault(key, Counter())
        item.update(truth=r['truth'], found=r['found'], matched=r['matched'], same_players=r['same_players'])
    for item in sweep.values():
        item['precision'] = item['matched'] / item['found'] if item['found'] else None
        item['recall'] = item['matched'] / item['truth'] if item['truth'] else None
    event_types, subtypes = Counter(), Counter()
    for d in details.values():
        event_types.update(d['events']['event_types'])
        subtypes.update(d['events']['event_subtypes'])
    pitch = Counter(f'{m["pitch_length"]}x{m["pitch_width"]}' for m in matches)
    summary = dict(
        source=str(source.resolve()), source_commit=commit, readme_says_matches=10,
        matches_listed=len(listed), match_folders=len(folders), matches_audited=len(matches),
        listed_without_folder=sorted(listed - folders), folders_not_listed=sorted(folders - listed),
        files=len(files), bytes=sum(f['bytes'] for f in files), lfs_pointers=sum(f['lfs_pointer'] for f in files),
        competitions=sorted({f'{m["competition"]} ({m["area"]}) {m["season"]}' for m in matches}),
        date_range=[min(m['date_utc'] for m in matches), max(m['date_utc'] for m in matches)],
        teams=sorted({m['home'] for m in matches} | {m['away'] for m in matches}),
        coordinates=dict(unit='metre', origin='pitch centre', x='long side', y='short side',
                         frame_rate_fps=FPS, pitch_sizes=dict(pitch),
                         event_xy='metres on the real pitch size, not scaled to 105x68'),
        totals=dict(frames=sum(m['frames'] for m in matches), period_frames=sum(m['period_frames'] for m in matches),
                    hours=sum(m['duration_minutes'] for m in matches) / 60, events=sum(m['events'] for m in matches),
                    event_types=dict(event_types.most_common()), event_subtypes=dict(subtypes.most_common()),
                    passes=sum(m['passes'] for m in matches),
                    passes_successful=sum(m['passes_successful'] for m in matches),
                    carries=sum(m['carries'] for m in matches), phases=sum(m['phases'] for m in matches),
                    tracked_frames=sum(m['tracked_frames'] for m in matches),
                    ball_xy_share_mean=total(matches, 'ball_xy_share'),
                    ball_detected_share_mean=total(matches, 'ball_detected_share'),
                    players_per_frame_mean=total(matches, 'players_per_frame_mean'),
                    detected_per_frame_mean=total(matches, 'detected_per_frame_mean'),
                    players_per_tracked_frame_mean=total(matches, 'players_per_tracked_frame_mean'),
                    detected_per_tracked_frame_mean=total(matches, 'detected_per_tracked_frame_mean'),
                    detected_per_tracked_frame_min=min(m['detected_per_tracked_frame_min'] for m in matches),
                    player_extrapolated_share_mean=total(matches, 'player_extrapolated_share'),
                    issues={k: sum(r['count'] for r in issue_rows if r['code'] == k)
                            for k in sorted({r['code'] for r in issue_rows})}),
        play_analysis=dict(
            default_rule=DEFAULT, tolerance_frames=10, max_gap_frames=50, focus_match=args.focus,
            focus=[r for r in agreement_rows if r['match_id'] == args.focus and r['method'] == 'nearest'
                   and r['radius'] == DEFAULT['radius'] and r['hold'] == DEFAULT['hold']]
            + [r for r in agreement_rows if r['match_id'] == args.focus and r['method'] == 'provided'],
            all_matches=sweep,
            home_share_abs_error_vs_minutes_tip=dict(
                mean=total([dict(e=abs(m['derived_home_share'] - m['home_tip_share'])) for m in matches], 'e'),
                max=max(abs(m['derived_home_share'] - m['home_tip_share']) for m in matches)),
            pass_value_fields=details[args.focus]['events']['pass_field_share'] if args.focus in details else None),
        aggregates=aggregates_summary(data / 'aggregates'), bodypose=bodypose_summary(data / 'bodypose'),
        per_match=details)
    (target / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    outputs = sorted(p.name for p in target.iterdir())
    manifest = dict(source=str(source.resolve()), source_commit=commit,
                    source_json=str(args.source_json),
                    source_json_sha256=sha256(args.source_json) if args.source_json.is_file() else None,
                    inputs_listing_sha256=sha256(target / 'files.csv'),
                    outputs={name: sha256(target / name) for name in outputs},
                    arguments={k: str(v) for k, v in vars(args).items()}, git=git_state(),
                    python=platform.python_version(), device='cpu', platform=platform.platform(),
                    script_sha256=sha256(Path(__file__)),
                    module_sha256=sha256(ROOT / 'sports_analyzer/skillcorner.py'),
                    seconds=round(time.time() - began, 1), created_utc=datetime.now(timezone.utc).isoformat())
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(run=str(target), matches=len(matches), totals={k: v for k, v in summary['totals'].items()
                                                                        if not isinstance(v, dict)}), indent=2))


if __name__ == '__main__':
    main()
