import random
import unittest

from sports_analyzer.analytics import holders
from sports_analyzer.skillcorner_eval import (ball_stats, check_split, date_split, degrade, goal_truth, nearest_table,
                                              one_to_one, same_players, scores, shifted, table_owners, truth_passes,
                                              unflagged_goals)

MATCH = dict(home_team=dict(id=1), away_team=dict(id=2))
SPANS = {1: (0, 999), 2: (1000, 1999)}


def frame(number, ball, players=(), period=1):
    return dict(frame=number, period=period, ball=ball, detected=True,
                players=[dict(id=i, team=None, role='player', position=(x, y)) for i, x, y in players])


def found(frame, passer, receiver):
    return dict(frame=frame, passer=passer, receiver=receiver)


def possession(start, end, player, team, end_type='pass', outcome='successful', start_type='pass_reception', **extra):
    row = dict(event_type='player_possession', period='1', frame_start=str(start), frame_end=str(end),
               player_id=str(player), team_id=str(team), end_type=end_type, pass_outcome=outcome,
               start_type=start_type, player_targeted_id='', game_interruption_before='', game_interruption_after='',
               attacking_side='left_to_right', team_score='0', opponent_team_score='0')
    return dict(row, **extra)


def track(n=2000, period_none=(0, 100)):
    return [frame(i, (float(i % 50), 0.0, 0.1), [(7, float(i % 50), 1.0)], None if i < period_none[1] else 1 + i // 1000)
            for i in range(n)]


class MatchingTests(unittest.TestCase):
    def test_strict_needs_same_players(self):
        truth = [found(100, 1, 2), found(200, 2, 3)]
        items = [found(103, 1, 2), found(198, 2, 4), found(500, 1, 2)]
        self.assertEqual(one_to_one(items, truth, -10, 10), [(1, 1), (0, 0)])
        self.assertEqual(one_to_one(items, truth, -10, 10, same_players), [(0, 0)])
        # Time only takes the closer wrong pair; strict waits for the right players.
        truth = [found(100, 1, 2)]
        items = [found(101, 5, 6), found(108, 1, 2)]
        self.assertEqual(one_to_one(items, truth, -10, 10), [(0, 0)])
        self.assertEqual(one_to_one(items, truth, -10, 10, same_players), [(1, 0)])
        result = scores(3, 2, 1)
        self.assertEqual((result['precision'], result['recall']), (1/3, 0.5))
        self.assertAlmostEqual(result['f1'], 0.4)
        self.assertEqual(scores(0, 2, 0)['f1'], 0.0)

    def test_goal_matching_is_one_to_one_inside_window(self):
        truth = [dict(frame=100), dict(frame=130)]
        events = [dict(frame=95), dict(frame=120), dict(frame=125), dict(frame=200)]
        self.assertEqual(one_to_one(events, truth, -10, 50), [(0, 0), (2, 1)])
        self.assertEqual(one_to_one(events, truth, 0, 50), [(1, 0)])
        self.assertEqual(one_to_one([dict(frame=101)], truth, -10, 50), [(0, 0)])
        self.assertEqual(one_to_one(events, [], -10, 50), [])

    def test_chance_baseline_is_reproducible(self):
        items = [found(10, 1, 2), found(500, 2, 3), found(1500, 3, 4)]
        a, b = shifted(items, SPANS, random.Random(3)), shifted(items, SPANS, random.Random(3))
        self.assertEqual(a, b)
        self.assertNotEqual(a, shifted(items, SPANS, random.Random(4)))
        self.assertTrue(all(s[0] <= t['frame'] <= s[1] for t, s in zip(a, (SPANS[1], SPANS[1], SPANS[2]))))
        self.assertEqual([(t['passer'], t['receiver']) for t in a], [(1, 2), (2, 3), (3, 4)])
        self.assertEqual(items[0]['frame'], 10)


class SplitTests(unittest.TestCase):
    def test_split_refuses_overlap_and_unknown(self):
        self.assertEqual(check_split(['3', 1], [2], [1, 2, 3]), ([3, 1], [2]))
        with self.assertRaises(ValueError):
            check_split([1, 2], [2, 3])
        with self.assertRaises(ValueError):
            check_split([1, 1], [2])
        with self.assertRaises(ValueError):
            check_split([1], [9], [1, 2])
        dates = {5: '2025-02-01T06:00:00Z', 4: '2025-02-01T04:00:00Z', 9: '2024-11-23T06:00:00Z'}
        self.assertEqual(date_split(dates, 2), ([9, 4], [5]))


class DegradeTests(unittest.TestCase):
    def test_target_recall_and_runs(self):
        frames = track()
        for recall, run in ((0.9, 0.3), (0.5, 1.5), (0.3, 1.5), (0.0, 0.3), (1.0, 1.5)):
            out = degrade(frames, recall, run, 0.0, random.Random(1))
            stats = ball_stats(frames, out)
            self.assertAlmostEqual(stats['recall'], recall, places=3)
            self.assertEqual(stats['false_share'], 0.0)
            if 0 < recall < 1:
                self.assertAlmostEqual(stats['miss_run_mean'], run, delta=0.1)
        self.assertEqual(degrade(frames, 0.5, 1.5, 0.05, random.Random(2)), degrade(frames, 0.5, 1.5, 0.05, random.Random(2)))

    def test_outside_play_untouched(self):
        frames = track()
        out = degrade(frames, 0.3, 0.3, 0.2, random.Random(5), 105, 68)
        self.assertTrue(all(out[i] is frames[i] for i in range(100)))
        changed = [g for f, g in zip(frames, out) if g['ball'] is not f['ball'] and g['ball'] is not None]
        self.assertTrue(all(abs(g['ball'][0]) <= 52.5 and abs(g['ball'][1]) <= 34 for g in changed))
        self.assertAlmostEqual(ball_stats(frames, out)['false_share'], 0.2, places=2)
        self.assertEqual(frames[500]['ball'], (0.0, 0.0, 0.1))

    def test_owner_table_matches_holders(self):
        frames = [frame(1, (0.0, 0.0, 0.1), [(7, 1.0, 0.0), (8, 3.0, 0.0)]), frame(2, None, [(7, 1.0, 0.0)]),
                  frame(3, (0.0, 0.0, 2.5), [(8, 0.5, 0.0)]), frame(4, (0.0, 0.0, None), [(8, 0.5, 0.0)]),
                  frame(5, (2.9, 0.0, 0.0), [(7, 1.0, 0.0), (8, 3.0, 0.0)], period=None), frame(6, (0.0, 0.0), [])]
        rows = nearest_table(frames)
        for radius, max_z in ((1.5, None), (1.5, 2.0), (0.4, None), (3.0, 1.0)):
            self.assertEqual(table_owners(rows, radius, max_z), holders(frames, radius, max_z=max_z))
        moved = [dict(frames[0], ball=(3.0, 0.0, 0.0))] + frames[1:]
        again = nearest_table(moved, rows)
        self.assertEqual(again[0][2], 8)
        self.assertIs(again[2], rows[2])


class TruthTests(unittest.TestCase):
    def test_provider_passes(self):
        events = [possession(1, 10, 10, 1, start_type='throw_in_reception'), possession(15, 20, 11, 1),
                  possession(25, 30, 20, 2, outcome='unsuccessful'), possession(35, 40, 21, 2, end_type='shot', outcome=''),
                  dict(possession(45, 50, 10, 1), event_type='on_ball_engagement')]
        truth = truth_passes(events, MATCH, {'throw_in_reception'})
        self.assertEqual([(t['frame'], t['passer'], t['receiver'], t['team'], t['completed'], t['set_piece'])
                          for t in truth], [(15, 10, 11, 'home', True, True), (25, 11, 20, 'home', False, False),
                                            (35, 20, 21, 'away', False, False)])

    def test_flagged_and_unflagged_goals(self):
        match = dict(MATCH, home_team_score=2, away_team_score=1)
        events = [possession(1, 10, 10, 1, 'shot', game_interruption_after='goal_for'),
                  possession(20, 30, 20, 2, 'shot', team_score='0', opponent_team_score='1',
                             game_interruption_after='goal_kick_against'),
                  possession(40, 50, 11, 1, 'shot', team_score='1', opponent_team_score='1',
                             attacking_side='right_to_left', game_interruption_after='goal_for'),
                  possession(60, 70, 21, 2, 'pass', team_score='1', opponent_team_score='2'),
                  possession(80, 90, 21, 2, 'shot', team_score='1', opponent_team_score='2')]
        goals = goal_truth(events, match)
        self.assertEqual([(g['frame'], g['team'], g['side']) for g in goals], [(10, 'home', 'right'), (50, 'home', 'left')])
        missing = unflagged_goals(events, match)
        self.assertEqual([(u['team'], u['frame_from'], u['frame_to'], u['last_interruption_after']) for u in missing],
                         [('away', 30, 40, 'goal_kick_against')])


if __name__ == '__main__':
    unittest.main()
