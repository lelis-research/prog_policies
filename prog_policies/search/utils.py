from prog_policies.base import BaseTask, dsl_nodes

def evaluate_program(program: dsl_nodes.Program, task_envs: list[BaseTask],
                     best_reward: float = None) -> float:
    mean_reward = 0.
    for task_env in task_envs:
        reward = task_env.evaluate_program(program)
        if reward < best_reward:
            return -float('inf')
        mean_reward += reward
    return mean_reward / len(task_envs)