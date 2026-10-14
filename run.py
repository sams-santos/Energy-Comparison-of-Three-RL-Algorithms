#!/usr/bin/env python3
import argparse
import itertools
import sys
from pathlib import Path
from typing import List

import numpy as np
import tsplib95

from algorithms.q_learning import QLearningTrainer
from algorithms.sarsa import SarsaTrainer
from algorithms.double_q import DoubleQTrainer

from utils.error_log import ErrorLog
from utils.file_utils import timestamp_tag
from utils.reward_utils import EPSILON_DECAY_TYPES


TRAINERS = {
    "qlearning": QLearningTrainer,
    "sarsa": SarsaTrainer,
    "doubleq": DoubleQTrainer,
}

REWARD_TYPES = ["R1", "R2", "R3"]
EPSILONS = [0.01, 0.05, 0.10]
GAMMAS = [0.01, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.99]
ALPHAS = [0.01, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.99]
ERROR_LOG_DIR = Path(__file__).resolve().parent / "results" / "error_logs"


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
    e_types: List[str],
    epsilons: List[float],
    gammas: List[float],
    alphas: List[float],
    episodes: int,
    repeat: int,
    seed: int,
    run_timestamp: str,
    error_log: ErrorLog,
) -> None:
    trainer_cls = TRAINERS[algorithm]
    grid = list(itertools.product(e_types, epsilons, gammas, alphas, REWARD_TYPES))

    for rep in range(repeat):
        run_index = rep + 1
        print(f"\n=== Repetition {run_index}/{repeat} ===")

        for instance_name in instances:
            error_log.set_context(algorithm=algorithm, instance=instance_name, run_index=run_index)
            try:
                problem = get_instance(instance_name)
                nodes = list(problem.get_nodes())
                dist_matrix = np.array(
                    [[problem.get_weight(i, j) for j in nodes] for i in nodes], dtype=float
                )
                n_points = problem.dimension
            except Exception as exc:
                print(f"Skipping instance {instance_name}: {exc}")
                error_log.record_exception(exc, "run.py:load_instance")
                error_log.flush()
                continue

            for combo, (e_type, epsilon, gamma, alpha, r_type) in enumerate(grid, start=1):
                print(
                    f"{algorithm.upper()} [{combo}/{len(grid)}]: instance={instance_name} e_decay={e_type} "
                    f"epsilon={epsilon} gamma={gamma} alpha={alpha} reward={r_type} run_index={run_index}"
                )
                context = dict(
                    algorithm=algorithm,
                    instance=instance_name,
                    e_type=e_type,
                    epsilon=epsilon,
                    gamma=gamma,
                    alpha=alpha,
                    r_type=r_type,
                    run_index=run_index,
                )
                error_log.set_context(**context)
                try:
                    trainer = trainer_cls(
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
                        seed=seed + rep,
                    )
                    if trainer.episodes_path.exists():
                        print(f"Already finished, skipping: {trainer.episodes_path}")
                        continue
                    error_log.set_context(trainer=trainer, **context)
                    episodes_path, summary_path = trainer.train()
                    print(f"Saved: {episodes_path}, {summary_path}")
                except Exception as exc:
                    # Log and move on to the next combination; its results are not saved.
                    print(f"ERROR in {algorithm} {instance_name} {context} : {exc}")
                    error_log.record_exception(exc, "run.py:train")
                finally:
                    # Between trainers, i.e. outside any episode measurement.
                    error_log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TSP RL experiments")
    parser.add_argument(
        "--algorithm",
        choices=[*TRAINERS, "all"],
        default="all",
        help="Select a single algorithm or run all three (qlearning, sarsa, doubleq)",
    )
    parser.add_argument("--episodes", type=int, default=10000)
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
    parser.add_argument(
        "--e-types",
        nargs="+",
        choices=EPSILON_DECAY_TYPES,
        default=EPSILON_DECAY_TYPES,
        help="Epsilon decay schedules to run (default: all five)",
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=EPSILONS, help="Initial epsilon values")
    parser.add_argument("--gammas", nargs="+", type=float, default=GAMMAS, help="Discount factor values")
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS, help="Learning rate values")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to repeat the full experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed; repetition k uses seed + k - 1.")
    parser.add_argument(
        "--resume",
        metavar="RUN_TIMESTAMP",
        help="Continue an interrupted execution: reuse its results folders and skip finished combinations.",
    )
    args = parser.parse_args()
    run_timestamp = args.resume or timestamp_tag()
    print(f"Run timestamp: {run_timestamp} (resume with --resume {run_timestamp})")

    # One file per execution, named by its start time, so earlier logs are never overwritten.
    error_log_path = ERROR_LOG_DIR / f"error_log_{timestamp_tag()}.csv"
    error_log = ErrorLog(error_log_path)
    error_log.install()

    algorithms = list(TRAINERS) if args.algorithm == "all" else [args.algorithm]
    try:
        for algorithm in algorithms:
            run_algorithm(
                algorithm,
                args.instances,
                args.e_types,
                args.epsilons,
                args.gammas,
                args.alphas,
                args.episodes,
                args.repeat,
                args.seed,
                run_timestamp,
                error_log,
            )
    except BaseException as exc:  # KeyboardInterrupt, SystemExit, or an error outside a trainer
        error_log.record_exception(exc, "run.py:main")
        raise
    finally:
        error_log.flush()
        if error_log.count:
            print(f"\n{error_log.count} error(s)/warning(s) recorded in {error_log_path}")

    if error_log.count:
        sys.exit(1)


if __name__ == "__main__":
    main()
