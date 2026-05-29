# algorithms/double_q.py
import random
from typing import Tuple, List

import numpy as np

from algorithms.base_trainer import BaseTrainer
from utils.reward_utils import reward_function, epsilon_decay


class DoubleQTrainer(BaseTrainer):

    def __init__(
        self,
        instance: str,
        r_type: str,
        e_type: str,
        matrix_d: np.ndarray,
        n_points: int,
        episodes: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        results_subdir: str = "double-q",
        run_index: int = 0,
        run_timestamp: str = "",
    ) -> None:
        super().__init__(
            instance=instance,
            r_type=r_type,
            e_type=e_type,
            matrix_d=matrix_d,
            n_points=n_points,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            results_subdir=results_subdir,
            algorithm_name="double_q",
            run_index=run_index,
            run_timestamp=run_timestamp,
        )
        self.q1_table = np.zeros((n_points, n_points))
        self.q2_table = np.zeros((n_points, n_points))

    def train(self) -> Tuple[str, str]:
        
        tracker = self._start_tracker()

        try:
            for ep in range(self.episodes):
                current_point = random.randint(0, self.n_points - 1)
                unvisited = list(range(self.n_points))
                unvisited.remove(current_point)
                path: List[int] = [current_point]
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
                    path.append(next_point)
                    current_point = next_point
                    unvisited.remove(next_point)

                last_point = path[-1]
                current_distance += float(self.matrix_d[last_point][path[0]])
                path.append(path[0])

                self.distance_history.append(current_distance)
                if current_distance < self.best_distance:
                    self.best_distance = current_distance
                    self.best_path = path.copy()

                self.epsilon = epsilon_decay(self.e_type, ep, self.episodes,self.epsilon_init)
                self.epsilon_history.append(self.epsilon)

            emissions_data = self._finalize_tracking(tracker)
            return self._save_results(emissions_data)

        except Exception:
            try:
                _ = tracker.stop()
            except Exception:
                pass
            raise
