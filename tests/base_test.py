from typing import Union
import unittest, sys

sys.path.append('.')

import numpy as np

from prog_policies import base
from prog_policies.base import dsl_nodes

class MockDSL(base.BaseDSL):

    def __init__(self):
        nodes_list = [
            dsl_nodes.While(), dsl_nodes.If(), dsl_nodes.ITE(), dsl_nodes.Repeat(),
            dsl_nodes.Concatenate(), dsl_nodes.Not(), dsl_nodes.And(), dsl_nodes.Or(),
            dsl_nodes.ConstInt(5), dsl_nodes.ConstInt(10), dsl_nodes.ConstInt(20),
            dsl_nodes.Action('action1'), dsl_nodes.Action('action2'),
            dsl_nodes.BoolFeature('feature1'), dsl_nodes.BoolFeature('feature2'),
            dsl_nodes.IntFeature('feature3'), dsl_nodes.IntFeature('feature4')
        ]
        super().__init__(nodes_list)


class MockEnvironment(base.BaseEnvironment):

    def __init__(self, initial_state: Union[np.ndarray, None] = None):
        actions = {
            "action1": self.action1,
            "action2": self.action2
        }
        bool_features = {
            "feature1": self.feature1,
            "feature2": self.feature2
        }
        int_features = {
            "feature3": self.feature3,
            "feature4": self.feature4
        }
        super().__init__(actions, bool_features, int_features, initial_state, max_calls=10)
    
    def default_state(self):
        return np.array([0, 0])
    def get_state(self):
        return self.s
    def set_state(self, s):
        self.s = s
    
    def action1(self):
        self.s[0] += 1
    def action2(self):
        self.s[1] += 1
    def feature1(self):
        return self.s[0] == 0
    def feature2(self):
        return self.s[1] == 0
    def feature3(self):
        return self.s[0]
    def feature4(self):
        return self.s[1]
    
    @classmethod
    def from_string(cls, state_str: str):
        raise NotImplementedError()
    def to_string(self):
        return str(self.s)
    
    def to_image(self):
        raise NotImplementedError()


class MockTask1(base.BaseTask):
    
    def generate_initial_environment(self, env_args = {}):
        return MockEnvironment(np.array([0, 0]))
    
    def reset_environment(self):
        super().reset_environment()
        self.previous_state_0 = self.environment.get_state()[0]
    
    def get_reward(self, env: base.BaseEnvironment):
        terminated = env.get_state()[0] == 5
        reward = (env.get_state()[0] - self.previous_state_0) / 5.
        self.previous_state_0 = env.get_state()[0]
        return terminated, reward


class TestBaseDSL(unittest.TestCase):
    
    def testParse(self):
        dsl = MockDSL()
        prog_str = "DEF run m( IFELSE c( feature1 c) i( action1 i) ELSE e( action2 e) m)"
        prog = dsl.parse_str_to_node(prog_str)
        self.assertIsInstance(prog, dsl_nodes.Program)
        self.assertTrue(prog.is_complete())
        prog_nodes = dsl_nodes.Program.new(
            dsl_nodes.ITE.new(dsl_nodes.BoolFeature('feature1'),
                              dsl_nodes.Action('action1'),
                              dsl_nodes.Action('action2'))
        )
        self.assertEqual(dsl.parse_node_to_str(prog_nodes), prog_str)
        
    def testExecute(self):
        dsl = MockDSL()
        env = MockEnvironment(np.array([0, 0]))
        prog = dsl.parse_str_to_node("DEF run m( IFELSE c( feature1 c) i( action1 i) ELSE e( action2 e) m)")
        prog.run(env)
        self.assertEqual(env.get_state()[0], 1)
        self.assertEqual(env.get_state()[1], 0)
        
    def testCrash(self):
        dsl = MockDSL()
        env = MockEnvironment(np.array([0, 0]))
        prog = dsl.parse_str_to_node("DEF run m( WHILE c( feature1 c) w( action2 w) m)")
        # This should crash because the max number of calls is 10
        prog.run(env)
        self.assertTrue(env.is_crashed())
        self.assertEqual(env.get_state()[1], 5)
        
    def testRunGenerator(self):
        pass
    
    def testTask(self):
        dsl = MockDSL()
        task = MockTask1()
        prog1 = dsl.parse_str_to_node("DEF run m( WHILE c( feature1 c) w( action2 w) m)")
        reward1 = task.evaluate_program(prog1)
        self.assertEqual(reward1, 0.)
        prog2 = dsl.parse_str_to_node("DEF run m( REPEAT R=20 r( action1 r) m)")
        reward2 = task.evaluate_program(prog2)
        # Even though the program executes action1 20 times, the reward is 1 because the task
        # terminates after 5 steps
        self.assertEqual(reward2, 1.)


if __name__ == '__main__':
    unittest.main()