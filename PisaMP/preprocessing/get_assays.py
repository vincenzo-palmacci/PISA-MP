import pandas as pd
import numpy as np
from tqdm import tqdm
from PisaM.path import pubchem, derived, metadata_derived

def get_assays():
    # Load pubchem from csv file.
    print("Loading pubchem data...")
    frame = pd.read_csv(pubchem)
    # Load metadata from csv file.
    metadata = pd.read_csv(metadata_derived / "pubchem_aid_refined_info.csv", index_col=0)

    # Convert activity labels in binary format.
    frame["activity"] = (frame["activity"] == "active").astype(int)

    # Select assays for which at least 10% of the compounds are tested.
    cid_unique = frame["cid"].unique()
    n_compounds = len(cid_unique)
    aid_unique = frame["aid"].unique()
    aid_retained = []
    percentaget_tested = []

    print("Computing assays...")
    for i in tqdm(aid_unique, total=len(aid_unique)):
        # first get all the compounds with the same id.
        sample = frame.loc[np.where(frame["aid"] == i)[0]]
        # Then count the number of unique compounds tested in the considered assay.
        n_cid_unique = len(sample["cid"].unique())
        # Select the assays for which at least 10% of the total number of compounds is tested.
        if n_cid_unique >= n_compounds * 0.1:
            # Drop possible conterscreen assays.
            if "ounter" not in metadata.loc[i]["aidname"]:
                percentaget_tested.append(n_cid_unique/n_compounds * 100)
                aid_retained.append(i)

    aid_retained = np.array(aid_retained)
    np.save(derived / "pubchem_aid_refined.npy", aid_retained)

    percentage_tested = np.array(percentaget_tested)
    np.save(derived / "percentage_tested.npy", percentage_tested)

    print(f"# of assay ids considered: {len(aid_retained)}")
    return print("Assays computed!")

if __name__=="__main__":
    get_assays()