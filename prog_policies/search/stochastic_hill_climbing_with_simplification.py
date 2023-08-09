from .stochastic_hill_climbing import StochasticHillClimbing
from .utils import evaluate_and_assign_credit, simplify_program

class StochasticHillClimbingWithSimplification(StochasticHillClimbing):
    
    def search_iteration(self):
        if self.current_iteration % 100 == 0:
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_reward = self.current_reward
            self.best_program = self.dsl.parse_node_to_str(self.current_program)
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        next_program = self.mutate_current_program()
        next_reward, _, next_count = evaluate_and_assign_credit(next_program, self.dsl, self.task_envs)
        next_program = simplify_program(next_program, next_count)
        self.num_evaluations += 1
        
        if next_reward >= self.current_reward:
            self.current_program = next_program
            self.current_reward = next_reward
