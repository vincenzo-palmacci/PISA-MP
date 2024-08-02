import numpy as np
import pandas as pd
import os

from PisaM.path import derived, metadata_derived, metrics


def get_delta(bioactivity, method_aid):
    # Divide the bioactivity matrix according to the detection method.
    specific = np.asarray([i for i in method_aid])
    others = np.asarray([i for i in bioactivity.columns if i not in specific])

    bioactivity_specific = bioactivity[specific]
    bioactivity_others = bioactivity[others]
    
    # Compute assay specific atr.
    atr_sample = bioactivity_specific.sum(axis=1) / (len(bioactivity_specific.columns) - bioactivity_specific.isna().sum(axis=1))
    atr_other = bioactivity_others.sum(axis=1) / (len(bioactivity_others.columns) - bioactivity_others.isna().sum(axis=1))
    #atr_other = bioactivity.sum(axis=1) / (len(bioactivity.columns) - bioactivity.isna().sum(axis=1))

    return atr_sample - atr_other


def get_data():
    # Load bioactivity matrix.
    bioactivity = pd.read_csv(derived / "matrix.csv", index_col="smiles")
    bioactivity.columns = bioactivity.columns.astype(int)

    # Load detection methods assay ids.
    detection_methods = {}
    for i in os.listdir(metadata_derived):
        if i.endswith(".npy"):
            method_name = i.split("_")[1].split(".")[0]
            method_aid = np.load(metadata_derived / i)
            method_aid_refined = np.intersect1d(bioactivity.columns, method_aid)

        detection_methods[method_name] = detection_methods.get(method_name, None)
        detection_methods[method_name] = method_aid_refined
    
    bioactivity = bioactivity.loc[bioactivity.sum(axis=1) >= 1]

    return bioactivity, detection_methods


def Main():
    # Load data needed for computing pvalues.
    print("\n Loading data ...")
    bioactivity, detection_methods = get_data()

    atr_values = {i: None for i in detection_methods.keys()}

    for i, j in detection_methods.items():
        results = get_delta(bioactivity, j)
        atr_values[i] = results

    df = pd.DataFrame.from_dict(atr_values)
    df.to_csv(metrics / "delta.csv")

    return print("\n Activity to Tested Ratio DELTA values computed!")

if __name__ == "__main__":
    Main()
