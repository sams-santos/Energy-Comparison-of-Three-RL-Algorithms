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
    results_dir: Path,
    metadata: Dict[str, object],
    distances: List[float],
    energies: List[Dict[str, Optional[float]]],
) -> Path:
    rows = [
        {**metadata, "episode": i, "distance": d, **energy}
        for i, (d, energy) in enumerate(zip(distances, energies))
    ]

    filename = f"{metadata['run_index']}_{metadata['instance']}_master_episodes.csv"
    master_path = results_dir / filename
    append_df_to_csv(pd.DataFrame(rows), master_path)
    return master_path


def save_summary(
    results_dir: Path,
    summary_row: Dict[str, object],
    master_summary_name: str = "master_summary.csv",
) -> Path:
    master_path = results_dir / master_summary_name
    append_df_to_csv(pd.DataFrame([summary_row]), master_path)
    return master_path
