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


TRAINERS = {
    "qlearning": QLearningTrainer,
    "sarsa": SarsaTrainer,
    "doubleq": DoubleQTrainer,
}

REWARD_TYPES = ["R1", "R2", "R3"]
GAMMA = 0.15
EPSILON = 0.01


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
    trainer_cls = TRAINERS[algorithm]
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

            for r_type in REWARD_TYPES:
                print(
                    f"{algorithm.upper()}: instance={instance_name} reward={r_type} "
                    f"alpha={alpha} gamma={GAMMA} epsilon={EPSILON} run_index={run_index}"
                )
                trainer = trainer_cls(
                    instance=instance_name,
                    r_type=r_type,
                    matrix_d=dist_matrix,
                    n_points=n_points,
                    episodes=episodes,
                    alpha=alpha,
                    gamma=GAMMA,
                    epsilon=EPSILON,
                    run_index=run_index,
                    run_timestamp=run_timestamp,
                )
                episodes_path, summary_path = trainer.train()
                print(f"Saved: {episodes_path}, {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TSP RL experiments")
    parser.add_argument(
        "--algorithm",
        choices=[*TRAINERS, "all"],
        default="all",
        help="Select a single algorithm or run all three (qlearning, sarsa, doubleq)",
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.75)
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

    algorithms = list(TRAINERS) if args.algorithm == "all" else [args.algorithm]
    for algorithm in algorithms:
        run_algorithm(algorithm, args.instances, args.episodes, args.alpha, args.repeat)


if __name__ == "__main__":
    main()
