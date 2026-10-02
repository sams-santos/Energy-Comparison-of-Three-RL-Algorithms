from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from codecarbon import OfflineEmissionsTracker

from utils.file_utils import save_per_episode, save_summary


ENERGY_FIELDS = {
    "Duration_sec": "duration",
    "Emissions_kgCO2": "emissions",
    "CPU_Energy_kWh": "cpu_energy",
    "RAM_Energy_kWh": "ram_energy",
    "Total_Energy_kWh": "energy_consumed",
}


class BaseTrainer:
    """Base trainer with common utilities for tabular RL on TSP."""

    def __init__(
        self,
        instance: str,
        r_type: str,
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
        self.matrix_d = matrix_d
        self.n_points = n_points
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.run_index = int(run_index)

        self.q_table = np.zeros((n_points, n_points))
        self.distance_history: List[float] = []
        self.energy_history: List[Dict[str, Optional[float]]] = []

        self.results_dir = Path("results") / results_subdir / run_timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        run_tag = f"_l{self.run_index}" if self.run_index else ""
        self.base_name = f"{algorithm_name}_{self.instance}_{self.r_type}{run_tag}"
        print(f"Initialized trainer: {self.base_name}")

    def _start_tracker(self) -> OfflineEmissionsTracker:
        """Start and return a CodeCarbon OfflineEmissionsTracker (results are saved by us, not by CodeCarbon)."""
        tracker = OfflineEmissionsTracker(
            project_name=self.base_name,
            save_to_file=False,
            country_iso_code="",
            country_2letter_iso_code="",
            region="",
            allow_multiple_runs=True,
            tracking_mode="process",
            rapl_include_dram=True,
            rapl_prefer_psys=True,
        )
        tracker.start()
        return tracker

    def _run_episode(self) -> float:
        """Run one training episode and return the tour distance. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_episode()")

    def train(self) -> Tuple[str, str]:
        """Train for all episodes, measuring energy per episode, and save the results."""
        tracker = self._start_tracker()
        try:
            for ep in range(self.episodes):
                tracker.start_task(f"episode_{ep}")
                distance = self._run_episode()
                emissions_data = tracker.stop_task()

                self.distance_history.append(distance)
                self.energy_history.append(
                    {col: getattr(emissions_data, attr, None) for col, attr in ENERGY_FIELDS.items()}
                )
        finally:
            tracker.stop()

        return self._save_results()

    def _summarize_energy(self) -> Dict[str, Optional[float]]:
        """Aggregate per-episode energy into run totals."""
        summary: Dict[str, Optional[float]] = {}
        for col in ENERGY_FIELDS:
            values = [e[col] for e in self.energy_history if e[col] is not None]
            summary[col] = float(np.sum(values)) if values else None

        duration, emissions = summary["Duration_sec"], summary["Emissions_kgCO2"]
        summary["EmissionsRate_kgCO2_per_sec"] = (
            emissions / duration if duration and emissions is not None else None
        )
        return summary

    def _save_results(self) -> Tuple[str, str]:
        """Save per-episode CSV and summary CSV and return their paths as strings."""
        metadata = {
            "run_index": self.run_index,
            "algorithm": self.base_name.split("_")[0],
            "instance": self.instance,
            "r_type": self.r_type,
        }
        episodes_path = save_per_episode(self.results_dir, metadata, self.distance_history, self.energy_history)
        summary_path = save_summary(self.results_dir, {**metadata, **self._summarize_energy()})
        return str(episodes_path), str(summary_path)
