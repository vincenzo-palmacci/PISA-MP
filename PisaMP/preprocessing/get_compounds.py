import pandas as pd
import numpy as np
from tqdm import tqdm
from PisaM.path import pubchem, derived, metadata_raw

def get_compounds():
    print("\n Loading dataset . . .")
    # Load pubchem from csv file.
    frame = pd.read_csv(pubchem)

    print("\n Selecting assays of interest . . .")
    # Select only assays of interest.
    refined_aid = np.load(derived / "pubchem_aid.npy", allow_pickle=True)

    # Select only assays associated to HTS projects for small molecules.
    hts_smol = pd.read_csv(metadata_raw / "aid_hts_smol.csv")[" aid"].values
    aid_smol = [i for i in refined_aid if i in hts_smol]

    # Slice dataframe according to the selected assays.
    frame = frame.loc[frame["aid"].isin(aid_smol)]
    # Convert activity labels in binary format.
    frame["activity"] = (frame["activity"] == "active").astype(int)

    # Select compounds tested in at least 20%, 50%, 80% of the assays.
    cid_unique = frame["cid"].unique()
    aid_unique = frame["aid"].unique()
    n_assays = len(aid_unique)
    cid_retained_80 = []
    cid_retained_50 = []
    cid_retained_20 = []

    print("\n Selecting compounds of interest . . .")
    for i in tqdm(cid_unique, total=len(cid_unique)):
        # first get all the compounds with the same id.
        sample = frame[frame["cid"] == i]
        # Then count the number of unique compounds tested in the considered assay.
        n_aid_unique = len(sample["aid"].unique())
        # Select compounds tested in at least 80% of the assays.
        if n_aid_unique >= n_assays * 0.8:
            cid_retained_80.append(i)
            cid_retained_50.append(i)
            cid_retained_20.append(i)
            continue
        if n_aid_unique >= n_assays * 0.5:
            cid_retained_50.append(i)
            cid_retained_20.append(i)
            continue
        if n_aid_unique >= n_assays * 0.2:
            cid_retained_20.append(i)

    cid_retained_80 = np.array(cid_retained_80)
    np.save(derived / "cid_80.npy", cid_retained_80)
    cid_retained_50 = np.array(cid_retained_50)
    np.save(derived / "cid_50.npy", cid_retained_50)  
    cid_retained_20 = np.array(cid_retained_20)
    np.save(derived / "cid_20.npy", cid_retained_20)

    return print("Compounds computed!")

if __name__=="__main__":
    get_compounds()