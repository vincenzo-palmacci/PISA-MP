import os
from pathlib import Path
import pandas as pd

def table():
    files = [
        [key, value, value.exists()]
        for key, value in globals().items()
        if isinstance(value, Path)
    ]
    return pd.DataFrame(files, columns=["variable", "path", "exists"])

# Determine paths
_here = Path(__file__).resolve()
_repo_root = _here.parent.parent

# Data
_data = _repo_root / "data"
raw = _data / "raw"
derived = _data / "derived"

# Add paths to data files and directories here:
pubchem = raw / "pubchem_2024-02-10_19-53-25.csv"

metadata_raw = raw / "metadata"
metadata_derived = derived / "metadata"

metrics = derived / "metrics"
splits = derived / "splits"

# Models development
random_forest = _repo_root / "PisaMP/random_forest"

# Models results
test_random_forest = random_forest / "results/test"


if __name__ == "__main__":
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(table())
