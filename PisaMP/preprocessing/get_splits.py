import numpy as np
import pandas as pd
from tqdm import tqdm

from PisaM.path import derived, splits


def get_splits():
    # Load bioactivity matrix.
    print("Loading data ...")
    bioactivity = pd.read_csv(derived / "matrix.csv").drop("smiles", axis=1)
    bioactivity.columns = bioactivity.columns.astype(int)
    bioactivity.reset_index(inplace=True)

    ### Random kfold split strategy.

    # Shuffle the data.
    shuffled = bioactivity.sample(frac=1, random_state=13)
    folds = np.array_split(shuffled, 6)

    # Save the splits.
    for i, arr in enumerate(folds):
        if i == 5:
            np.save(splits / f"test.npy", arr.index)
            continue
        np.save(splits / f"fold{i}.npy", arr.index)

    return print("\n Splits computed!")

if __name__ == "__main__":
    get_splits()