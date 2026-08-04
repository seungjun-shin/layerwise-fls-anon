from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append dictionaries to a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key) for key in self.fieldnames})

