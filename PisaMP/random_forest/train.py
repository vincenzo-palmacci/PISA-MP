import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import joblib
import os

from imblearn.ensemble import BalancedRandomForestClassifier

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from PisaM.path import derived, metrics, splits


# Ignoring warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

"""
Random Forest Classifier: Train models.
"""

def _compute_morgan(smile):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    molecule = Chem.MolFromSmiles(smile)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
    morgan_fp = fpgen.GetFingerprint(molecule)
    
    return morgan_fp

def _get_labels(dataset):
    """
    Compute binary interference labels.
    """
    labels = pd.read_csv(metrics / "delta.csv", index_col=0)[dataset].values

    return labels


def _get_threshold(labels):
    return labels.mean(), labels.std()


def make_rfc(hyperparameters):
    """
    Instanciate Random Forest with specified parameters.
    """
    model = BalancedRandomForestClassifier(
                n_estimators = hyperparameters["n_estimators"],
                max_depth = hyperparameters["max_depth"],
                min_samples_split = hyperparameters["min_samples_split"],
                max_features = hyperparameters["max_features"],
                bootstrap = True,
                n_jobs = 32,
                verbose = 32,
                )

    return model


@click.command()
@click.option("-d", "--dataset", required=True, help="Specify the detection method {fp, biolumnescence, flint, luminescence, chemiluminescence}")
def Main(dataset):
    """
    Random Forest Classifier: Training.
    """
    # Load dataa.
    print("\n Loading data ...")
    data = pd.read_csv(derived / "matrix.csv", index_col="smiles")

    # Load training indexes.
    folds_indices = [np.load(splits / i).tolist() for i in os.listdir(splits) if i != "test.npy"]
    train_idx = []
    for idx in folds_indices:
         train_idx += idx
    
    # Compute molecules descriptors.
    smiles = data.iloc[train_idx].index.tolist() # get samples
    X = np.array([_compute_morgan(i) for i in tqdm(smiles)], dtype=float)
    # Get labels.
    labels = _get_labels(dataset)[train_idx]
    cut = _get_threshold(labels)
    y = (labels > cut[0] + 1 *cut[1]).astype(int)
    
    print("\nDataset specs: ")
    print("\t# Compound:", X.shape[0])
    print("\t# features:", X.shape[1])

    # Load the optimized random forest parameters.
    params = np.load(f"results/validation/{dataset}.npy", allow_pickle=True)[()]

    # Define the model.
    model = make_rfc(params)
    print(f"\nInitilized Model: {model}")

    # Train and validate.
    model.fit(X, y)

    # Save trained model.
    joblib.dump(model, f"trained_models/{dataset}.pkl") 

    return print("Model saved")


if __name__=="__main__":
    Main()
