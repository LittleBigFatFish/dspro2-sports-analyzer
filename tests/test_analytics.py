import unittest

from sports_analyzer.analytics import (centre, feet, goal_mouth_events, goal_side, holder, holders, holding,
                                       interpolate_ball, known_teams, nearest, pass_counts, passes, possessions)

FPS = 10
TEAMS = {'a1': 'home', 'a2': 'home', 'b1': 'away'}
GOAL = [(0, 0), (100, 0), (100, 40), (0, 40)]


def player(id, x, y, team=None, role='player', height=None):
    item = dict(id=id, team=team, role=role, position=(x, y))
    return item if height is None else dict(item, height=height)


def frame(number, ball, players=(), period=1, **extra):
    return dict(frame=number, period=period, ball=ball, players=list(players), **extra)


def owners(*runs):
    out = []
    for player_id, length in runs:
        out += [(len(out) + i + 1, 1, player_id) for i in range(length)]
    return out


class HolderTests(unittest.TestCase):
    def test_referee_is_never_holder(self):
        scene = frame(1, (0.0, 0.0, 0.1), [player('ref', 0.2, 0.0, role='referee'), player('a1', 1.0, 0.0, 'home'),
                                            player('b1', 5.0, 0.0, 'away'), player('gk', 0.1, 9.0, role='goalkeeper')])
        self.assertEqual(nearest(scene), ('a1', 1.0))
        self.assertEqual(holder(scene, 1.5), 'a1')
        self.assertIsNone(holder(scene, 0.5))
        self.assertIsNone(holder(frame(2, None, scene['players']), 1.5))
        with self.assertRaises(ValueError):
            nearest(scene, 'video')

    def test_ball_height_limit(self):
        players = [player('a1', 1.0, 0.0)]
        self.assertIsNone(holder(frame(1, (0.0, 0.0, 3.0), players), 1.5, max_z=2.0))
        self.assertEqual(holder(frame(1, (0.0, 0.0, None), players), 1.5, max_z=2.0), 'a1')
        self.assertEqual(holder(frame(1, (0.0, 0.0), players), 1.5, max_z=2.0), 'a1')

    def test_same_scene_in_pitch_and_image_mode(self):
        # 20 px per metre, players 1.8 m tall.
        px = lambda x, y: (100 + 20*x, 400 + 20*y)
        people = [('a1', 10.0, 5.0), ('b1', 12.0, 5.0), ('ref', 10.6, 5.2)]
        pitch, image = [], []
        for number, (bx, by) in enumerate([(10.5, 5.0), (11.8, 5.0), (20.0, 5.0), (10.6, 5.3)], 1):
            pitch.append(frame(number, (bx, by), [player(i, x, y, TEAMS.get(i), 'referee' if i == 'ref' else 'player')
                                                  for i, x, y in people]))
            players = []
            for i, x, y in people:
                u, v = px(x, y)
                point, height = feet((u - 8, v - 36, u + 8, v))
                players.append(player(i, *point, TEAMS.get(i), 'referee' if i == 'ref' else 'player', height))
            u, v = px(bx, by)
            image.append(frame(number, centre((u - 3, v - 3, u + 3, v + 3)), players))
        found = [h for _, _, h in holders(pitch, 1.5)]
        self.assertEqual(found, ['a1', 'b1', None, 'a1'])
        self.assertEqual([h for _, _, h in holders(image, 0.5, 'image')], found)
        self.assertAlmostEqual(nearest(image[0], 'image')[1], 10 / 36)


class SequenceTests(unittest.TestCase):
    def test_hold_suppresses_flicker(self):
        found = owners(('a1', 5), ('b1', 1), ('a1', 4), ('b1', 5))
        self.assertEqual(possessions(found, 0.3, FPS), [dict(player='a1', period=1, start=1, end=10),
                                                        dict(player='b1', period=1, start=11, end=15)])
        self.assertEqual(len(possessions(found, 0.1, FPS)), 4)
        self.assertEqual(len(possessions(found, 0.04, 25)), 4)
        with self.assertRaises(ValueError):
            possessions(found, 0.04, FPS)

    def test_pass_turnover_and_transfer(self):
        segments = [dict(player='a1', period=1, start=1, end=3), dict(player='a2', period=1, start=8, end=10),
                    dict(player='b1', period=1, start=15, end=20), dict(player='a1', period=1, start=80, end=90),
                    dict(player='a2', period=2, start=91, end=95)]
        found = passes(segments, TEAMS, 5.0, FPS)
        self.assertEqual([(t['kind'], t['frame'], t['passer'], t['receiver'], t['gap']) for t in found],
                         [('pass', 8, 'a1', 'a2', 5), ('turnover', 15, 'a2', 'b1', 5)])
        self.assertEqual(len(passes(segments, TEAMS, 6.0, FPS)), 3)
        self.assertEqual(pass_counts(found), dict(teams={'home': {'pass': 1, 'turnover': 1}},
                                                  players={'a1': {'pass': 1}, 'a2': {'received': 1, 'turnover': 1}}))
        neutral = passes(segments, None, 5.0, FPS)
        self.assertEqual([(t['kind'], t['team']) for t in neutral], [('transfer', None), ('transfer', None)])
        self.assertEqual(pass_counts(neutral)['teams'], {None: {'transfer': 2}})

    def test_ball_gap_interpolation(self):
        frames = [frame(1, (0.0, 0.0, 0.0)), frame(2, None), frame(3, None), frame(4, (3.0, 6.0, None)),
                  frame(5, (3.0, 6.0, 0.0))]
        filled = interpolate_ball(frames, 0.2, FPS)
        self.assertEqual([f['ball'] for f in filled[1:3]], [(1.0, 2.0, None), (2.0, 4.0, None)])
        self.assertEqual([f['interpolated'] for f in filled], [False, True, True, False, False])
        self.assertIsNone(frames[1]['ball'])
        short = interpolate_ball(frames, 0.1, FPS)
        self.assertEqual([f['ball'] for f in short[1:3]], [None, None])
        self.assertFalse(any(f['interpolated'] for f in short))
        skipped = [frame(1, (0.0, 0.0)), frame(4, (3.0, 0.0))]
        self.assertEqual(interpolate_ball(skipped, 0.2, FPS), [dict(f, interpolated=False) for f in skipped])
        halves = [frame(1, (0.0, 0.0), period=1), frame(2, None, period=None), frame(3, (1.0, 0.0), period=2)]
        self.assertIsNone(interpolate_ball(halves, 1.0, FPS)[1]['ball'])

    def test_holding_with_ball_in_flight(self):
        found = owners(('a1', 4), (None, 4), ('b1', 4))
        segments = possessions(found, 0.2, FPS)
        self.assertEqual([(s['player'], s['start'], s['end']) for s in segments], [('a1', 1, 4), ('b1', 9, 12)])
        shares = holding(found, segments, TEAMS)
        self.assertEqual(shares['held'], {'away': 0.5, 'home': 0.5})
        self.assertEqual(shares['last_holder'], {'away': 4 / 12, 'home': 8 / 12})
        self.assertEqual(holding(found, segments, {}), dict(held={}, last_holder={}))

    def test_image_pipeline_with_teams(self):
        frames = []
        for number in range(1, 13):
            x = 0.0 if number <= 5 else 100.0
            ball = None if number in (6, 7) else (x, 0.0)
            frames.append(frame(number, ball, [player('a1', 0.0, 5.0, 'home', height=40),
                                               player('a2', 100.0, 5.0, 'home', height=40)]))
        teams = known_teams(frames)
        self.assertEqual(teams, {'a1': 'home', 'a2': 'home'})
        filled = interpolate_ball(frames, 0.2, FPS)
        segments = possessions(holders(filled, 0.5, 'image'), 0.3, FPS)
        self.assertEqual([t['kind'] for t in passes(segments, teams, 5.0, FPS)], ['pass'])


class GoalTests(unittest.TestCase):
    def test_pitch_goal_mouth(self):
        self.assertEqual(goal_side(frame(1, (53.0, 0.0, 1.0))), 'right')
        self.assertIsNone(goal_side(frame(1, (53.0, 4.0, 0.5))))
        self.assertIsNone(goal_side(frame(1, (53.0, 1.0, 3.0))))
        self.assertIsNone(goal_side(frame(1, (52.0, 0.0, 0.0))))
        self.assertEqual(goal_side(frame(1, (-53.0, -3.0, None))), 'left')
        self.assertIsNone(goal_side(frame(1, (53.0, 0.0, 1.0)), length=110))
        balls = [(53.0, 0.0, 1.0), (53.5, 1.0, 0.5), (54.0, 2.0, 0.2), (53.0, 4.0, 0.5), (53.0, 1.0, 3.0),
                 (-53.0, -3.0, None), None, (53.0, 0.0, 0.0)]
        frames = [frame(n, b) for n, b in enumerate(balls, 1)]
        events = goal_mouth_events(frames, FPS)
        self.assertEqual([(e['side'], e['start'], e['end']) for e in events],
                         [('right', 1, 3), ('left', 6, 6), ('right', 8, 8)])
        self.assertAlmostEqual(events[0]['seconds'], 0.3)
        self.assertEqual([e['start'] for e in goal_mouth_events(frames, FPS, min_duration=0.2)], [1])

    def test_image_goal_polygon(self):
        balls = [(50, 20), (99, 39), (150, 20), (10, 10), (10, 10)]
        frames = [frame(n, b, goals={'left': GOAL}) for n, b in enumerate(balls, 1)]
        frames[3]['period'] = 2
        frames.append(frame(6, (50, 20)))
        events = goal_mouth_events(frames, FPS, 'image')
        self.assertEqual([(e['side'], e['period'], e['start'], e['end']) for e in events],
                         [('left', 1, 1, 2), ('left', 2, 4, 4), ('left', 1, 5, 5)])
        self.assertIsNone(goal_side(frames[2], 'image'))


if __name__ == '__main__':
    unittest.main()
