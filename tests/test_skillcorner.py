from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from sports_analyzer.skillcorner import (event_transitions, iter_frames, match_transitions, nearest_owners,
                                         player_teams, possession_share, possessions, seconds,
                                         summarise_events, summarise_tracking, transitions)

MATCH = dict(home_team=dict(id=1), away_team=dict(id=2),
             players=[dict(id=10, team_id=1), dict(id=11, team_id=1), dict(id=20, team_id=2)])
TEAMS = {10: 'home', 11: 'home', 20: 'away'}


def frame(number, ball, players, period=1, group=None, holder=None):
    timestamp = None if period is None else f'00:00:{(number - 1) / 10:05.2f}'
    return dict(frame=number, timestamp=timestamp, period=period,
                ball_data=dict(x=ball[0], y=ball[1], z=0.1, is_detected=ball[0] is not None),
                possession=dict(player_id=holder, group=group),
                image_corners_projection=dict(x_top_left=None),
                player_data=[dict(player_id=i, x=x, y=y, is_detected=d) for i, x, y, d in players])


def event(index, frame_start, frame_end, player, team, end_type='pass', start_type='pass_reception', **extra):
    row = dict(event_id=f'8_{index}', event_type='player_possession', event_subtype='', period='1',
               frame_start=str(frame_start), frame_end=str(frame_end), player_id=str(player), team_id=str(team),
               start_type=start_type, end_type=end_type, pass_outcome='successful' if end_type == 'pass' else '',
               x_start='1.0', y_start='2.0', x_end='3.0', y_end='4.0', carry='False')
    row.update(extra)
    return row


class SkillCornerTests(unittest.TestCase):
    def test_teams_and_timestamps(self):
        self.assertEqual(player_teams(MATCH), TEAMS)
        with self.assertRaises(ValueError):
            player_teams({**MATCH, 'players': [dict(id=30, team_id=9)]})
        self.assertAlmostEqual(seconds('01:02:03.40'), 3723.4)
        self.assertIsNone(seconds(None))

    def test_possession_needs_hold_and_resets_on_period(self):
        owners = [(1, 1, 10), (2, 1, 10), (3, 1, 10), (4, 1, None), (5, 1, 11), (6, 1, 20), (7, 1, 11),
                  (8, 1, 11), (9, 1, 11), (10, 1, 10), (11, 2, 10), (12, 2, 10), (13, 2, 10)]
        self.assertEqual(possessions(owners, 3), [dict(player=10, period=1, start=1, end=3),
                                                  dict(player=11, period=1, start=7, end=9),
                                                  dict(player=10, period=2, start=11, end=13)])
        self.assertEqual(len(possessions(owners, 1)), 6)
        gap = [(1, 1, 10), (2, 1, 10), (5, 1, 10)]
        self.assertEqual(possessions(gap, 3), [])
        with self.assertRaises(ValueError):
            possessions(owners, 0)

    def test_transitions_against_events(self):
        segments = [dict(player=10, period=1, start=1, end=3), dict(player=11, period=1, start=8, end=9),
                    dict(player=20, period=1, start=12, end=15), dict(player=10, period=1, start=200, end=210)]
        found = transitions(segments, TEAMS, max_gap=50)
        self.assertEqual([(t['kind'], t['frame'], t['passer'], t['receiver']) for t in found],
                         [('pass', 8, 10, 11), ('turnover', 12, 11, 20)])
        self.assertAlmostEqual(possession_share(segments, TEAMS), 16 / 20)
        events = [event(1, 1, 4, 10, 1), event(2, 9, 10, 11, 1, end_type='possession_loss'),
                  event(3, 13, 16, 20, 2, start_type='recovery'), event(4, 40, 42, 20, 2, start_type='keep')]
        truth = event_transitions(events)
        self.assertEqual([t['kind'] for t in truth], ['pass', 'turnover', 'other'])
        result = match_transitions(found, truth, 'pass', tolerance=2)
        self.assertEqual((result['truth'], result['found'], result['matched'], result['same_players']), (1, 1, 1, 1))
        self.assertEqual(match_transitions(found, truth, 'pass', tolerance=0)['matched'], 0)
        self.assertEqual(match_transitions(found, truth, 'turnover')['recall'], 1.0)

    def test_tracking_summary_and_nearest_rule(self):
        frames = [frame(0, (None, None), [], period=None),
                  frame(1, (0.0, 0.0), [(10, 0.5, 0.0, True), (20, 3.0, 0.0, False)], group='home team', holder=10),
                  frame(2, (0.0, 0.0), [(10, 2.0, 0.0, True), (20, 0.4, 0.0, True), (99, 80.0, 0.0, True)]),
                  frame(2, (None, None), [(10, 2.0, 0.0, True), (10, 1.0, 0.0, False)]),
                  frame(5, (1.0, 0.0), [(11, 1.0, 1.0, True)])]
        summary, records = summarise_tracking(frames, TEAMS, 105, 68)
        self.assertEqual((summary['frames'], summary['period_frames'], summary['max_frame_gap']), (5, 4, 3))
        self.assertEqual(summary['issues']['duplicate_frames'], 1)
        self.assertEqual(summary['issues']['frame_gaps'], 2)
        self.assertEqual(summary['issues']['duplicate_player_in_frame'], 1)
        self.assertEqual(summary['issues']['player_outside_pitch'], 1)
        self.assertEqual(summary['issues']['period_frames_without_ball'], 1)
        self.assertEqual(summary['unknown_player_ids'], [99])
        self.assertEqual((summary['players_per_frame_min'], summary['players_per_frame_max']), (1, 3))
        self.assertEqual((summary['tracked_frames'], summary['detected_per_tracked_frame_min']), (4, 1))
        self.assertAlmostEqual(summary['ball_xy_share'], 0.75)
        self.assertAlmostEqual(summary['player_extrapolated_share'], 2 / 8)
        self.assertEqual([r['nearest'] for r in records], [10, 20, None, 11])
        self.assertEqual([o[2] for o in nearest_owners(records, 1.5)], [10, 20, None, 11])
        self.assertEqual([o[2] for o in nearest_owners(records, 0.45)], [None, 20, None, None])
        self.assertEqual([o[2] for o in nearest_owners(records, 1.5, max_z=0.05)], [None] * 4)

    def test_events_summary_and_bad_jsonl(self):
        events = [event(1, 1, 4, 10, 1), event(1, 9, 5, 11, 1, x_end='70.0'),
                  dict(event(3, 10, 12, 20, 2), event_type='on_ball_engagement', event_subtype='pressure')]
        summary = summarise_events(events, 105, 68, 0, 10)
        self.assertEqual(summary['duplicate_event_ids'], 1)
        self.assertEqual(summary['event_types'], {'player_possession': 2, 'on_ball_engagement': 1})
        self.assertEqual(summary['pass_outcomes'], {'successful': 2})
        self.assertEqual(summary['issues']['frame_end_before_start'], 1)
        self.assertEqual(summary['issues']['frames_outside_tracking'], 1)
        self.assertEqual(summary['issues']['coordinates_outside_pitch'], 1)
        self.assertEqual(summary['pass_field_share']['player_targeted_xthreat'], 0.0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'tracking.jsonl'
            path.write_text(json.dumps(dict(frame=0)) + '\n\n{"frame": 1\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                list(iter_frames(path))


if __name__ == '__main__':
    unittest.main()
