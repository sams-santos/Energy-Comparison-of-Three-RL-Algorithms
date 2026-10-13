import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from codecarbon import OfflineEmissionsTracker
from codecarbon.external.hardware import CPU

from utils.file_utils import save_per_episode, save_summary
from utils.reward_utils import epsilon_decay


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAPL_ENERGY_FILE = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_FIX_HINT = (
    "Grant read access with: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj "
    "(the udev rule /etc/udev/rules.d/99-intel-rapl.rules makes it persistent)."
)

# Per-episode energy columns (summed into the run summary).
ENERGY_FIELDS = {
    "Duration_sec": "duration",
    "Emissions_kgCO2": "emissions",
    "CPU_Energy_kWh": "cpu_energy",
    "RAM_Energy_kWh": "ram_energy",
    "Total_Energy_kWh": "energy_consumed",
}

# CodeCarbon power readings in W (not summed). In task mode CodeCarbon reports a running average
# over every measurement since the tracker started, not the power of the episode alone.
POWER_FIELDS = {
    "CPU_Power_W": "cpu_power",
    "RAM_Power_W": "ram_power",
}

# Average power derived from the measured energy and duration: (column, energy column).
DERIVED_POWER_FIELDS = {
    "CPU_Power_avg_W": "CPU_Energy_kWh",
    "RAM_Power_avg_W": "RAM_Energy_kWh",
}

KWH_TO_J = 3.6e6


def _average_power(energy_kwh: Optional[float], duration_sec: Optional[float]) -> Optional[float]:
    """Average power in W over an interval, or None when it cannot be computed."""
    if energy_kwh is None or not duration_sec:
        return None
    return energy_kwh * KWH_TO_J / duration_sec


# Measurement context, recorded once per run.
CONTEXT_FIELDS = {
    "tracking_mode": "tracking_mode",
}

# Known optimal tour lengths from TSPLIB.
KNOWN_OPTIMA = {
    "br17": 39,
    "berlin52": 7542,
    "eil51": 426,
    "ftv33": 1286,
    "ftv64": 1839,
    "kroA100": 21282,
    "st70": 675,
    "tsp225": 3916,
}


def _check_rapl_readable() -> None:
    """Raise if the RAPL energy counter cannot be read by the current user."""
    if not os.access(RAPL_ENERGY_FILE, os.R_OK):
        raise RuntimeError(f"RAPL energy counter {RAPL_ENERGY_FILE} is not readable. {RAPL_FIX_HINT}")


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
        seed: Optional[int] = None,
    ) -> None:
        self.algorithm_name = algorithm_name
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
        self.run_index = int(run_index)
        self.run_timestamp = run_timestamp
        self.seed = seed

        self.q_table = np.zeros((n_points, n_points))
        self.distance_history: List[float] = []
        self.epsilon_history: List[float] = []
        self.energy_history: List[Dict[str, object]] = []
        self.tracker_data = None
        self.current_episode: Optional[int] = None

        self.results_dir = PROJECT_ROOT / "results" / results_subdir / run_timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        run_tag = f"_l{self.run_index}" if self.run_index else ""
        self.base_name = (
            f"{algorithm_name}_{self.instance}_{self.e_type}_e{self.epsilon_init}"
            f"_g{self.gamma}_a{self.alpha}_{self.r_type}{run_tag}"
        )
        self.episodes_path = self.results_dir / "episodes" / self.instance / self.e_type / f"{self.base_name}.csv"
        print(f"Initialized trainer: {self.base_name}")

    def _start_tracker(self) -> OfflineEmissionsTracker:
        """Start and return a CodeCarbon OfflineEmissionsTracker (results are saved by us, not by CodeCarbon).

        Aborts instead of silently recording TDP/CPU-load estimates when RAPL is not readable.
        """
        _check_rapl_readable()
        tracker = OfflineEmissionsTracker(
            project_name=self.base_name,
            save_to_file=False,
            country_iso_code="IRL",
            country_2letter_iso_code="IE",
            region="leinster",
            allow_multiple_runs=True,
            # RAPL always measures the whole CPU package; "process" mode only adds a full
            # /proc scan on every stop_task (~5 ms), which ends up inside each episode's measurement.
            tracking_mode="machine",
            rapl_include_dram=True,
            rapl_prefer_psys=True,
            # INFO logs several lines per episode; keep warnings visible.
            log_level="warning",
        )
        tracker.start()
        cpu_modes = [hw._mode for hw in tracker._hardware if isinstance(hw, CPU)]
        if cpu_modes != ["intel_rapl"]:
            tracker.stop()
            raise RuntimeError(f"CodeCarbon is not using RAPL (CPU modes: {cpu_modes}). {RAPL_FIX_HINT}")
        return tracker

    def _run_episode(self) -> float:
        """Run one training episode and return the tour distance. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_episode()")

    def train(self) -> Tuple[str, str]:
        """Train for all episodes, measuring energy per episode, and save the results.

        Only _run_episode() runs between start_task() and stop_task(); all bookkeeping
        happens outside that window so it never enters the per-episode measurement.
        """
        if self.seed is not None:
            random.seed(self.seed)
        tracker = self._start_tracker()
        try:
            for ep in range(self.episodes):
                self.current_episode = ep  # read by the error log; set before the measurement starts
                self.epsilon = epsilon_decay(self.e_type, ep, self.episodes, self.epsilon_init)
                tracker.start_task(f"episode_{ep}")
                distance = self._run_episode()
                emissions_data = tracker.stop_task()
                # CodeCarbon reads an unreadable RAPL file as 0 J, which corrupts the episode delta.
                _check_rapl_readable()

                self.distance_history.append(distance)
                self.epsilon_history.append(self.epsilon)
                episode_data = {col: getattr(emissions_data, attr, None) for col, attr in ENERGY_FIELDS.items()}
                episode_data.update({col: getattr(emissions_data, attr, None) for col, attr in POWER_FIELDS.items()})
                for col, energy_col in DERIVED_POWER_FIELDS.items():
                    episode_data[col] = _average_power(episode_data[energy_col], episode_data["Duration_sec"])
                episode_data["Timestamp"] = getattr(emissions_data, "timestamp", None)
                self.energy_history.append(episode_data)
        finally:
            tracker.stop()
            self.tracker_data = getattr(tracker, "final_emissions_data", None)

        return self._save_results()

    def _summarize_energy(self) -> Dict[str, Optional[float]]:
        """Aggregate per-episode energy into run totals."""
        summary: Dict[str, Optional[float]] = {}
        for col in ENERGY_FIELDS:
            values = [e[col] for e in self.energy_history if e[col] is not None]
            summary[col] = float(np.sum(values)) if values else None

        duration, emissions = summary["Duration_sec"], summary["Emissions_kgCO2"]
        total_energy = summary["Total_Energy_kWh"]
        summary["EmissionsRate_kgCO2_per_sec"] = (
            emissions / duration if duration and emissions is not None else None
        )
        summary["carbon_intensity_kgCO2_per_kWh"] = (
            emissions / total_energy if total_energy and emissions is not None else None
        )
        # Power is not additive: report CodeCarbon's run-level value and the energy-weighted average.
        summary.update({col: getattr(self.tracker_data, attr, None) for col, attr in POWER_FIELDS.items()})
        for col, energy_col in DERIVED_POWER_FIELDS.items():
            summary[col] = _average_power(summary[energy_col], duration)
        return summary

    def _summarize_run(self) -> Dict[str, object]:
        """Measurement context."""
        data = self.tracker_data
        summary: Dict[str, object] = {"cpu_measurement": "RAPL"}
        summary.update({col: getattr(data, attr, None) for col, attr in CONTEXT_FIELDS.items()})
        return summary

    def _summarize_distance(self) -> Dict[str, Optional[float]]:
        """Solution-quality statistics over the episodes."""
        distances = np.asarray(self.distance_history, dtype=float)
        optimal = KNOWN_OPTIMA.get(Path(self.instance).stem)
        if distances.size == 0:
            return {"optimal_distance": optimal}

        best_episode = int(np.argmin(distances))
        return {
            "best_distance": float(distances[best_episode]),
            "best_episode": best_episode,
            "optimal_distance": optimal,
        }

    def _save_results(self) -> Tuple[str, str]:
        """Save per-episode CSV and summary CSV and return their paths as strings."""
        metadata = {
            "run_index": self.run_index,
            "algorithm": self.algorithm_name,
            "instance": self.instance,
            "r_type": self.r_type,
            "e_type": self.e_type,
            "epsilon_init": self.epsilon_init,
            "gamma": self.gamma,
            "alpha": self.alpha,
        }
        params = {
            "run_timestamp": self.run_timestamp,
            "seed": self.seed,
            "episodes": self.episodes,
            "n_points": self.n_points,
            # Epsilon used in the last episode, after the decay schedule.
            "epsilon_final": self.epsilon_history[-1] if self.epsilon_history else None,
        }
        # Written before the per-episode file, which marks the combination as finished for --resume.
        summary_path = save_summary(
            self.results_dir,
            {
                **metadata,
                **params,
                **self._summarize_distance(),
                **self._summarize_energy(),
                **self._summarize_run(),
            },
            master_summary_name=f"{self.instance}_master_summary.csv",
        )
        episodes_path = save_per_episode(
            self.episodes_path, metadata, self.distance_history, self.epsilon_history, self.energy_history
        )
        return str(episodes_path), str(summary_path)
