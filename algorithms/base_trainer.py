from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from codecarbon import OfflineEmissionsTracker

from utils.file_utils import save_per_episode, save_summary


class BaseTrainer:
    """Base trainer with common utilities for tabular RL on TSP."""

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
        results_subdir: str,
        algorithm_name: str,
        run_index: int = 0,
        run_timestamp: str = "",
    ) -> None:
        self.instance = instance
        self.r_type = r_type
        self.e_type = e_type
        self.matrix_d = matrix_d
        self.n_points = n_points
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.epsilon_history: List[float] = []
        self.run_index = int(run_index)

        self.q_table = np.zeros((n_points, n_points))
        self.best_path: List[int] = []
        self.best_distance: float = float("inf")
        self.distance_history: List[float] = []

        self.timestamp = run_timestamp

        self.results_dir = Path("results") / results_subdir / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        run_tag = f"_l{self.run_index}" if self.run_index else ""
        self.base_name = (
            f"{algorithm_name}_{self.instance}"
            f"_{self.epsilon}_{self.e_type}_{self.r_type}{run_tag}"
        )
        print(f"Initialized trainer: {self.base_name}")

    def _start_tracker(self) -> OfflineEmissionsTracker:
        """Start and return a CodeCarbon OfflineEmissionsTracker saving to results_dir."""
        tracker = OfflineEmissionsTracker(
            project_name=f"{self.base_name}",
            output_dir=self.results_dir,
            output_file=f"{self.base_name}_emissions.csv",
            country_iso_code="",
            country_2letter_iso_code="",
            region = "",
            allow_multiple_runs=True,
            tracking_mode="process",
            rapl_include_dram=True,
            rapl_prefer_psys=True,
        )
        tracker.start()
        return tracker

    def _finalize_tracking(self, tracker: OfflineEmissionsTracker):
        """Stop tracker and return CodeCarbon's final_emissions_data object."""
        _ = tracker.stop()
        return tracker.final_emissions_data

    def _save_results(self, emissions_data: Optional[object]) -> Tuple[str, str]:
        """Save per-episode CSV and summary CSV and return their paths as strings."""
        if not self.distance_history:
            best_episode = -1
        else:
            best_episode = int(min(range(len(self.distance_history)), key=self.distance_history.__getitem__))

        metadata = {
            "run_index": self.run_index,
            "algorithm": self.base_name.split("_")[0],
            "instance": self.instance,
            "r_type": self.r_type,
            "e_type": self.e_type,
            "epsilon": self.epsilon,
            "epsilon_init": self.epsilon_init,
        }
        master_path = save_per_episode(self.results_dir, self.base_name, self.distance_history, metadata, self.epsilon_history)
        summary_row = {
            **metadata,
            "BestEpisode": best_episode,
            "BestDistance": None if best_episode == -1 else self.best_distance,
            "BestPath": "" if best_episode == -1 else " -> ".join(map(str, self.best_path)),
            "Duration_sec": getattr(emissions_data, "duration", None),
            "Emissions_kgCO2": getattr(emissions_data, "emissions", None),
            "EmissionsRate_kgCO2_per_sec": getattr(emissions_data, "emissions_rate", None),
            "CPU_Power_W": getattr(emissions_data, "cpu_power", None),
            "RAM_Power_W": getattr(emissions_data, "ram_power", None),
            "CPU_Energy_Wh": getattr(emissions_data, "cpu_energy", None),
            "RAM_Energy_Wh": getattr(emissions_data, "ram_energy", None),
            "Total_Energy_Wh": getattr(emissions_data, "energy_consumed", None),
        }
        summary_path = save_summary(self.results_dir, self.base_name, summary_row)
        return str(master_path), str(summary_path)

    def train(self) -> Tuple[str, str]:
        """Train method to implement in subclasses."""
        raise NotImplementedError("Subclasses must implement train()")
