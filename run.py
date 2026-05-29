#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import numpy as np
import tsplib95

from algorithms.q_learning import QLearningTrainer
from algorithms.sarsa import SarsaTrainer
from algorithms.double_q import DoubleQTrainer

from utils.file_utils import timestamp_tag


def get_instance(filename: str) -> tsplib95.models.StandardProblem:
    """Load a TSPLIB instance from the project's instances directory."""
    base_dir = Path(__file__).resolve().parent
    instance_path = base_dir / "instances" / filename
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    return tsplib95.load(instance_path)

def run_algorithm(
    algorithm: str,
    instances: List[str],
    episodes: int,
    alpha: float,
    repeat: int,
) -> None:
    epsilon_decay_types = ["linear", "convex", "concave", "step", "fixed"]
    reward_types = ["R1","R2","R3"]
    gamma = 0.15
    alpha = 0.75
    epsilon_set = [0.01, 0.05, 0.10]

    run_timestamp = timestamp_tag()

    for rep in range(repeat):
        run_index = rep + 1
        print(f"\n=== Repetition {run_index}/{repeat} ===")

        for instance_name in instances:
            try:
                problem = get_instance(instance_name)
            except FileNotFoundError as exc:
                print(f"Skipping instance {instance_name}: {exc}")
                continue

            nodes = list(problem.get_nodes())
            dist_matrix = np.array(
                [[problem.get_weight(i, j) for j in nodes] for i in nodes], dtype=float
            )
            n_points = problem.dimension

            for epsilon in epsilon_set:
                for e_type in epsilon_decay_types:
                    for r_type in reward_types:
                        print(
                            f"{algorithm.upper()}: instance={instance_name} e_decay={e_type} "
                            f"reward={r_type} gamma={gamma} run_index={run_index}"
                        )

                        if algorithm == "qlearning":
                            trainer = QLearningTrainer(
                                instance=instance_name,
                                r_type=r_type,
                                e_type=e_type,
                                matrix_d=dist_matrix,
                                n_points=n_points,
                                episodes=episodes,
                                alpha=alpha,
                                gamma=gamma,
                                epsilon=epsilon,
                                run_index=run_index,
                                run_timestamp=run_timestamp,
                            )
                            master_ep_path, master_sum_path = trainer.train()
                            print(f"Saved master: {master_ep_path}, {master_sum_path}")

                        elif algorithm == "sarsa":
                            trainer = SarsaTrainer(
                                instance=instance_name,
                                r_type=r_type,
                                e_type=e_type,
                                matrix_d=dist_matrix,
                                n_points=n_points,
                                episodes=episodes,
                                alpha=alpha,
                                gamma=gamma,
                                epsilon=epsilon,
                                run_index=run_index,
                                run_timestamp=run_timestamp,
                            )
                            master_ep_path, master_sum_path = trainer.train()
                            print(f"Saved master: {master_ep_path}, {master_sum_path}")

                        elif algorithm == "doubleq":
                            trainer = DoubleQTrainer(
                                instance=instance_name,
                                r_type=r_type,
                                e_type=e_type,
                                matrix_d=dist_matrix,
                                n_points=n_points,
                                episodes=episodes,
                                alpha=alpha,
                                gamma=gamma,
                                epsilon=epsilon,
                                run_index=run_index,
                                run_timestamp=run_timestamp,
                            )
                            master_ep_path, master_sum_path = trainer.train()
                            print(f"Saved master: {master_ep_path}, {master_sum_path}")

                        else:
                            raise ValueError(f"Unknown algorithm: {algorithm}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TSP RL experiments")
    parser.add_argument(
        "--algorithm",
        choices=["qlearning", "sarsa", "doubleq", "all"],
        default="all",
        help="Select a single algorithm or run all three (qlearning, sarsa, doubleq)",
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--instances",
        nargs="+",
        default=[
            "br17.atsp",
            "berlin52.tsp",
            "eil51.tsp",
            "ftv33.atsp",
            "ftv64.atsp",
            "kroA100.tsp",
            "st70.tsp",
            "tsp225.tsp",
        ],
        help="List of instance filenames located in ./instances",
    )
    parser.add_argument("--repeat", type=int, default=1, help="How many times to repeat the full experiment.")
    args = parser.parse_args()

    if args.algorithm == "qlearning":
        run_algorithm("qlearning", args.instances, args.episodes, args.alpha, args.repeat)
    elif args.algorithm == "sarsa":
        run_algorithm("sarsa", args.instances, args.episodes, args.alpha, args.repeat)
    elif args.algorithm == "doubleq":
        run_algorithm("doubleq", args.instances, args.episodes, args.alpha, args.repeat)
    else:
        run_algorithm("qlearning", args.instances, args.episodes, args.alpha, args.repeat)
        run_algorithm("sarsa", args.instances, args.episodes, args.alpha, args.repeat)
        run_algorithm("doubleq", args.instances, args.episodes, args.alpha, args.repeat)


if __name__ == "__main__":
    main()
