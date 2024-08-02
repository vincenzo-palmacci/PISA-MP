import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import click
import joblib

from imblearn.ensemble import BalancedRandomForestClassifier

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


from PisaM.path import random_forest


# Ignoring warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

"""
Random Forest Classifier: Test models.
"""

def _compute_morgan(smile):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    molecule = Chem.MolFromSmiles(smile)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
    morgan_fp = fpgen.GetFingerprint(molecule)
    
    return morgan_fp

@click.command()
@click.option("-s", "--smiles", required=True, help="Path to the file containing the smiles.")
@click.option("-d", "--dataset", required=True, help="Specify the detection method {fp, biolumnescence, flint, luminescence, chemiluminescence}")
@click.option("-o", "--out", required=True, help="Path to the output file.")
def Main(smiles, dataset, out):
    """
    Random Forest Classifier: Test.
    """
    # Load dataa.
    print("\n Loading data ...")
    data = np.load(smiles, allow_pickle=True)

    # Compute molecules descriptors.
    X = np.array([_compute_morgan(i) for i in tqdm(data)], dtype=float)

    print("\nDataset specs: ")
    print("\t# Compound:", X.shape[0])
    print("\t# features:", X.shape[1])

    # Load trained model.
    model = joblib.load(random_forest / f"trained_models/{dataset}.pkl")
    print(model)

    # Compute predictions.
    predictions = model.predict_proba(X)
    
    # Save predictions.
    predictions_df = pd.DataFrame([smiles, predictions[:,1]]).T
    predictions_df.columns = ["smiles", "probability_interference"]
    predictions_df.to_csv(out, index=False)

    return print("Predictions saved")


if __name__=="__main__":
    Main()
