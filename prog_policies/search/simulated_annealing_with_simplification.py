from .simulated_annealing import SimulatedAnnealing
from .utils import evaluate_and_assign_credit, simplify_program

class SimulatedAnnealingWithSimplification(SimulatedAnnealing):
    
    def search_iteration(self):
        if self.current_iteration % 100 == 0:
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, temperature {self.current_temperature}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_reward = self.current_reward
            self.best_program = self.dsl.parse_node_to_str(self.current_program)
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        if self.current_temperature > 1.0:
            next_program = self.mutate_current_program()
            next_reward, _, next_count = evaluate_and_assign_credit(next_program, self.dsl, self.task_envs)
            next_program = simplify_program(next_program, next_count)
            self.num_evaluations += 1
            
            if next_reward > self.best_reward:
                self.best_reward = next_reward
                self.best_program = self.dsl.parse_node_to_str(next_program)
                self.save_best()
            
            if self.np_rng.rand() < self.accept_function(self.current_reward, next_reward):
                self.current_program = next_program
                self.current_reward = next_reward
                
            self.iterations_since_restart += 1
            self.decrease_temperature(self.iterations_since_restart)
            
        else:
            if self.best_reward > 0.0:
                self.current_program = self.dsl.parse_str_to_node(self.best_program)

            else:
                self.current_program = self.random_program()
                self.current_reward, _, curr_count = evaluate_and_assign_credit(self.current_program, self.dsl, self.task_envs)
                self.current_program = simplify_program(self.current_program, curr_count)
                self.num_evaluations += 1

            self.current_temperature = self.initial_temperature
            self.iterations_since_restart = 0