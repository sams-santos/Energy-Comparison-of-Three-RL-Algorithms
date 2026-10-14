from pathlib import Path
from typing import Dict, List, Optional

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
    out_path: Path,
    metadata: Dict[str, object],
    distances: List[float],
    epsilons: List[float],
    energies: List[Dict[str, Optional[float]]],
) -> Path:
    rows = [
        {**metadata, "episode": i, "epsilon": eps, "distance": d, **energy}
        for i, (d, eps, energy) in enumerate(zip(distances, epsilons, energies))
    ]

    # One file per training run (one hyperparameter combination), so files stay at `episodes` rows.
    # Written to a temporary file and renamed, so an interrupted write never leaves a partial file.
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)
    return out_path


def save_summary(
    results_dir: Path,
    summary_row: Dict[str, object],
    master_summary_name: str = "master_summary.csv",
) -> Path:
    master_path = results_dir / master_summary_name
    append_df_to_csv(pd.DataFrame([summary_row]), master_path)
    return master_path
