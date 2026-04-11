import csv
import os
from pathlib import Path

import pandas as pd


class MetricsLogger:
    """Append-only CSV logger. One row per epoch."""

    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self._header_written = os.path.exists(self.path)

    def log(self, epoch: int, metrics: dict):
        row = {"epoch": epoch, **metrics}
        file_exists = os.path.exists(self.path)

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame()
        return pd.read_csv(self.path)
