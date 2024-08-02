import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import click
import joblib
import os

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, roc_auc_score, average_precision_score, balanced_accuracy_score

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


from PisaM.path import derived, metrics, splits


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


def classification_report(true, preds):
    """
    Make classification report.
    """
    classification_metrics = {"recall":0, "precision":0, "mcc":0, "auroc":0, "aupr":0, "bacc":0}
    
    binary = [1 if i >= 0.5 else 0 for i in preds]

    classification_metrics["recall"] = recall_score(true, binary)
    classification_metrics["precision"] = precision_score(true, binary)
    classification_metrics["mcc"] = matthews_corrcoef(true, binary)
    classification_metrics["auroc"] = roc_auc_score(true, preds)
    classification_metrics["aupr"] = average_precision_score(true, preds)
    classification_metrics["bacc"] = balanced_accuracy_score(true, binary)
    
    return classification_metrics


def _get_labels(dataset):
    """
    Compute binary interference labels.
    """
    labels = pd.read_csv(metrics / "delta.csv", index_col=0)[dataset].values

    return labels


def _get_threshold(labels):
    return labels.mean(), labels.std()


@click.command()
@click.option("-d", "--dataset", required=True, help="Specify the detection method {fp, biolumnescence, flint, luminescence, chemiluminescence}")
def Main(dataset):
    """
    Random Forest Classifier: Test.
    """
    # Load dataa.
    print("\n Loading data ...")
    data = pd.read_csv(derived / "matrix.csv", index_col="smiles")

    # Load training and test indexes.
    folds_indices = [np.load(splits / i).tolist() for i in os.listdir(splits) if i != "test.npy"]
    train_idx = []
    for idx in folds_indices:
         train_idx += idx

    test_idx = np.load(splits / "test.npy").tolist()

    # Compute molecules descriptors.
    smiles = data.iloc[test_idx].index.tolist() # get samples
    X = np.array([_compute_morgan(i) for i in tqdm(smiles)], dtype=float)
    # Get labels.
    labels = _get_labels(dataset)
    train_labels = labels[train_idx] # Training labels needed to compute labeling threshold.
    cut = _get_threshold(train_labels)
    test_labels = labels[test_idx]
    y = (test_labels > cut[0] + 1 *cut[1]).astype(int)

    print("\nDataset specs: ")
    print("\t# Compound:", X.shape[0])
    print("\t# features:", X.shape[1])

    # Load trained model.
    model = joblib.load(f"trained_models/{dataset}.pkl")
    print(model)

    # Compute predictions.
    predictions = model.predict_proba(X)[:, 1]

    results = classification_report(y, predictions)
    print(results)

    # Save predictions.
    np.save(f"results/test/{dataset}.npy", predictions)

    return print("Predictions saved")


if __name__=="__main__":
    Main()
