# algorithms/double_q.py
import random

import numpy as np

from algorithms.base_trainer import BaseTrainer
from utils.reward_utils import reward_function


class DoubleQTrainer(BaseTrainer):

    def __init__(self, *args, run_index: int = 0, **kwargs) -> None:
        super().__init__(
            *args,
            algorithm_name="double_q",
            results_subdir="double-q",
            run_index=run_index,
            **kwargs,
        )
        self.q1_table = np.zeros((self.n_points, self.n_points))
        self.q2_table = np.zeros((self.n_points, self.n_points))

    def _run_episode(self) -> float:
        start_point = current_point = random.randint(0, self.n_points - 1)
        unvisited = list(range(self.n_points))
        unvisited.remove(current_point)
        current_distance = 0.0

        while unvisited:
            if random.uniform(0, 1) < self.epsilon:
                next_point = random.choice(unvisited)
            else:
                q_values = {
                    p: (self.q1_table[current_point, p] + self.q2_table[current_point, p]) / 2.0
                    for p in unvisited
                }
                next_point = max(q_values, key=q_values.get)

            distance = float(self.matrix_d[current_point][next_point])
            reward = reward_function(self.r_type, distance)

            if random.random() < 0.5:
                best_action = int(np.argmax(self.q1_table[next_point, :]))
                target = reward + self.gamma * self.q2_table[next_point, best_action]
                self.q1_table[current_point, next_point] += self.alpha * (
                    target - self.q1_table[current_point, next_point]
                )
            else:
                best_action = int(np.argmax(self.q2_table[next_point, :]))
                target = reward + self.gamma * self.q1_table[next_point, best_action]
                self.q2_table[current_point, next_point] += self.alpha * (
                    target - self.q2_table[current_point, next_point]
                )

            current_distance += distance
            current_point = next_point
            unvisited.remove(next_point)

        current_distance += float(self.matrix_d[current_point][start_point])
        return current_distance
