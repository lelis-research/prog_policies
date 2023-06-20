import unittest, sys

sys.path.append('.')

import prog_policies.karel as karel
import prog_policies.karel_tasks as karel_tasks

class TestKarelEnvironment(unittest.TestCase):
    
    def testEnvironment(self):
        str_initial_state = (
            '|  |\n'
            '| *|\n'
            '|  |\n'
            '|  |\n'
            '| *|\n'
            '| 1|\n'
            '| *|\n'
            '|^ |'
        )
        env = karel.KarelEnvironment.from_string(str_initial_state)
        self.assertEqual(env.to_string(), str_initial_state)
        if env.get_bool_feature('rightIsClear'):
            env.run_action('turnRight')
            env.run_action('move')
            env.run_action('putMarker')
            env.run_action('turnLeft')
            env.run_action('turnLeft')
            env.run_action('move')
            env.run_action('turnRight')
        while env.get_bool_feature('frontIsClear'):
            env.run_action('move')
            if env.get_bool_feature('rightIsClear'):
                env.run_action('turnRight')
                env.run_action('move')
                env.run_action('putMarker')
                env.run_action('turnLeft')
                env.run_action('turnLeft')
                env.run_action('move')
                env.run_action('turnRight')
        expected_final_state = (
            '|^1|\n'
            '| *|\n'
            '| 1|\n'
            '| 1|\n'
            '| *|\n'
            '| 2|\n'
            '| *|\n'
            '| 1|'
        )
        self.assertEqual(env.to_string(), expected_final_state)

    def testDSL(self):
        str_initial_state = (
            '|  |\n'
            '| *|\n'
            '|  |\n'
            '|  |\n'
            '| *|\n'
            '| 1|\n'
            '| *|\n'
            '|^ |'
        )
        env = karel.KarelEnvironment.from_string(str_initial_state)
        dsl = karel.KarelDSL()
        prog = dsl.parse_str_to_node((
            'DEF run m( '
                'IF c( rightIsClear c) i( turnRight move putMarker turnLeft turnLeft move turnRight i) '
                'WHILE c( frontIsClear c) w( '
                    'move '
                    'IF c( rightIsClear c) i( turnRight move putMarker turnLeft turnLeft move turnRight i) '
                'w) '
            'm)'
        ))
        prog.run(env)
        expected_final_state = (
            '|^1|\n'
            '| *|\n'
            '| 1|\n'
            '| 1|\n'
            '| *|\n'
            '| 2|\n'
            '| *|\n'
            '| 1|'
        )
        self.assertEqual(env.to_string(), expected_final_state)

    def testStairClimber(self):
        dsl = karel.KarelDSL()
        prog_that_crashes = dsl.parse_str_to_node((
            'DEF run m( '
                'turnLeft move move '
            'm)'
        ))
        env_args = {
            'env_height': 12,
            'env_width': 12
        }
        task = karel_tasks.StairClimber(seed=0, env_args=env_args)
        crash_reward = task.evaluate_program(prog_that_crashes)
        self.assertLess(crash_reward, 0.)
        gt_prog = dsl.parse_str_to_node((
            'DEF run m( '
                'WHILE c( noMarkersPresent c) w( '
                    'turnLeft move turnRight move '
                'w) '
            'm)'
        ))
        for seed in range(10):
            task = karel_tasks.StairClimber(seed=seed, env_args=env_args)
            gt_reward = task.evaluate_program(gt_prog)
            self.assertAlmostEqual(gt_reward, 1.)
        for seed in range(10):
            sparse_task = karel_tasks.StairClimberSparse(seed=seed, env_args=env_args)
            gt_sparse_reward = sparse_task.evaluate_program(gt_prog)
            self.assertAlmostEqual(gt_sparse_reward, 1.)

if __name__ == '__main__':
    unittest.main()