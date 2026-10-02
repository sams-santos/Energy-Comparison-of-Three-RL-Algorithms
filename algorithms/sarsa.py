import random

from .base_trainer import BaseTrainer
from utils.reward_utils import reward_function


class SarsaTrainer(BaseTrainer):

    def __init__(self, *args, run_index: int = 0, **kwargs) -> None:
        super().__init__(
            *args,
            algorithm_name="sarsa",
            results_subdir="sarsa",
            run_index=run_index,
            **kwargs,
        )

    def _run_episode(self) -> float:
        start_point = current_point = random.randint(0, self.n_points - 1)
        unvisited = list(range(self.n_points))
        unvisited.remove(current_point)
        current_distance = 0.0

        while unvisited:
            if random.uniform(0, 1) < self.epsilon:
                next_point = random.choice(unvisited)
            else:
                q_values = {p: self.q_table[current_point, p] for p in unvisited}
                next_point = max(q_values, key=q_values.get)

            distance = float(self.matrix_d[current_point][next_point])
            reward = reward_function(self.r_type, distance)

            if len(unvisited) > 1:
                unvisited_next = [p for p in unvisited if p != next_point]
                if random.uniform(0, 1) < self.epsilon:
                    next_next_point = random.choice(unvisited_next)
                else:
                    q_values_next = {p: self.q_table[next_point, p] for p in unvisited_next}
                    next_next_point = max(q_values_next, key=q_values_next.get)
                future_q = float(self.q_table[next_point, next_next_point])
            else:
                future_q = 0.0

            last_q = self.q_table[current_point, next_point]
            new_q = last_q + self.alpha * (reward + self.gamma * future_q - last_q)
            self.q_table[current_point, next_point] = new_q

            current_distance += distance
            current_point = next_point
            unvisited.remove(next_point)

        current_distance += float(self.matrix_d[current_point][start_point])
        return current_distance
