from pathlib import Path
from typing import Dict, List

from datetime import datetime, timezone

import pandas as pd


def timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_df_to_csv(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False)


def save_per_episode(
    results_dir: Path,
    base_name: str,
    distances: List[float],
    metadata: Dict[str, object],
    epsilons: List[float],
) -> Path:
    ensure_dir(results_dir)

    rows = []
    run_idx = metadata.get("run_index", 0)
    instance = metadata.get("instance")

    for i, d in enumerate(distances):
        rows.append(
            {
                "run_index": run_idx,
                "algorithm": metadata.get("algorithm"),
                "instance": instance,
                "r_type": metadata.get("r_type"),
                "e_type": metadata.get("e_type"),
                "epsilon_init": metadata.get("epsilon_init"),
                "epsilon": epsilons[i],
                "episode": i,
                "distance": d,
            }
        )
    
    master_df = pd.DataFrame(rows)

    filename = f"{run_idx}_{instance}_master_episodes.csv"
    master_path = results_dir / filename

    append_df_to_csv(master_df, master_path)
    return master_path


def save_summary(
    results_dir: Path,
    base_name: str,
    summary_row: Dict[str, object],
    master_summary_name: str = "master_summary.csv",
) -> Path:
    ensure_dir(results_dir)
    summary_df = pd.DataFrame([summary_row])
    per_path = results_dir / f"{base_name}_summary.csv"

    master_path = results_dir / master_summary_name
    append_df_to_csv(summary_df, master_path)
    return per_path
