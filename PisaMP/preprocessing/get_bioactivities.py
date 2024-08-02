import pandas as pd
import numpy as np

from PisaM.path import pubchem, derived

def get_bioactivities():
    print("\nLoading data . . .")
    # Load pubchem dataframe.
    frame = pd.read_csv(pubchem)
    # Convert activity labels in binary format.
    frame["activity"] = (frame["activity"] == "active").astype(int)

    # Get assays ids to be considered.
    assays_selected = np.load(derived / "pubchem_aid_refined.npy")

    # Get compounds ids to be considered.
    compounds_selected = np.load(derived / "cid_80.npy")

    print("\nPivoting the table . . .")
    # Slice the dataframe.
    new_frame = frame.loc[frame["aid"].isin(assays_selected)]
    new_frame = new_frame.loc[new_frame["cid"].isin(compounds_selected)]

    # Pivot table.
    pivoted = pd.pivot_table(new_frame, values="activity", index="smiles", columns="aid")
    # Select only assays that have more tham 80% of tested compounds.
    new_assays_selected = np.where((pivoted.isna().sum().sort_values(ascending=False) / pivoted.shape[0]) < 0.2)[0]
    new_assays_selected = [pivoted.columns[i] for i in new_assays_selected]

    pivoted = pivoted[new_assays_selected]
    # Save bioactivity matrix.
    pivoted.to_csv(derived / "matrix.csv")

    # Save new bioassay ids.
    np.save(derived / "pubchem_aid_refined.npy", np.array(new_assays_selected))

    return print("Bioactivity matrix computed!")


if __name__=="__main__":
    get_bioactivities()
