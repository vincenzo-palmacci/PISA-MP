import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os

from PisaM.path import derived, metadata_raw, metadata_derived

def get_detection_methods():
    # Load assays ids.
    pubchem_aid = np.load(derived / "pubchem_aid_refined.npy")

    # Read aid.
    detection_methods = {}
    for i in os.listdir(metadata_raw):
        method_name = i.split(".")[0]
        detection_methods[method_name] = detection_methods.get(method_name, None)

        method_file = metadata_raw / i
        try:
            method_aid = pd.read_csv(method_file)[" aid"].values
        except:
            method_aid = pd.read_csv(method_file)["aid"].values
        detection_methods[method_name] = method_aid

    # Check overlapping compounds.
    for i in detection_methods.keys():
        overlap = np.intersect1d(pubchem_aid, detection_methods[i])
        print(i.split("_")[1], ":", len(overlap))
        np.save(metadata_derived / f"{i}.npy", overlap)

    return print("Overlaps computed!")

if __name__ == "__main__":
    get_detection_methods()
