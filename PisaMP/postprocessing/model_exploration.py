import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import click

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from lazypredict.Supervised import LazyClassifier

# Disable rdkit logger
RDLogger.DisableLog('rdApp.*')

def compute_morgan(smile, radius=2):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    try:
        molecule = Chem.MolFromSmiles(smile)
    except:
        return None
    if molecule is None:
        return None
    fp_object = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=2048)
    morgan_fp = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp_object, morgan_fp)
    return morgan_fp

def compute_physchem(smile, missingVal=None):
    """
    Calculate the full list of descriptors for a molecule
    missingVal is used if the descriptor cannot be calculated
    """
    res = []
    for nm, fn in Descriptors._descList:
        try:
            mol = Chem.MolFromSmiles(smile)
            val = fn(mol)
        except:
            import traceback
            traceback.print_exc()
            val = missingVal
        res.append(val)
    return res

def _get_labels(dataset):
    """
    Compute binary interference labels.
    """
    labels = pd.read_csv("../../data/derived/metrics/delta.csv", index_col=0)[dataset].values
    return labels

def _get_threshold(labels):
    return labels.mean(), labels.std()

@click.command()
@click.option("--fps", default="morgan", help="Fingerprints to use for the model.")
def Main(fps):
    #### Load data and generate descriptors
    data = pd.read_csv("../../data/derived/matrix.csv", index_col="smiles")

    # Load training indexes
    folds_indices = [np.load(f"../../data/derived/splits/{i}").tolist() for i in os.listdir("../../data/derived/splits") if i != "test.npy"]
    train_idx = []
    for idx in folds_indices:
        train_idx += idx

    # Load test data
    test_idx = np.load("../../data/derived/splits/test.npy").tolist()

    # Compute morgan fingerprints
    if fps == "morgan":
        fps_train = [compute_morgan(data.index[i]) for i in tqdm(train_idx)]
        fps_test = [compute_morgan(data.index[i]) for i in tqdm(test_idx)]

    # Compute physchem descriptors
    else:
        fps_train = [compute_physchem(data.index[i]) for i in tqdm(train_idx)]
        fps_test = [compute_physchem(data.index[i]) for i in tqdm(test_idx)]

    # Compute labels
    labels_train = {}
    labels_test = {}
    for i in ["bioluminescence", "chemiluminescence", "flint", "fp", "luminescence"]:
        l_train = _get_labels(i)[train_idx]
        l_test = _get_labels(i)[test_idx]
        cut = _get_threshold(l_train)
        labels_train[i] = (l_train > cut[0] + 1 * cut[1]).astype(int)
        labels_test[i] = (l_test > cut[0] + 1 * cut[1]).astype(int)

    #### Test Morgan fingerprints
    performances = {}
    X_train = np.array([np.array(list(j)) for j in fps_train])
    X_test = np.array([np.array(list(j)) for j in fps_test])
    for i in labels_train.keys():
        print(i)
        print("Training...")
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        y_train_ = labels_train[i]
        y_test_ = labels_test[i]
        models, predictions = clf.fit(X_train, X_test, y_train_, y_test_)
        performances[i] = models

        print(models)

        # Save performances to a file
        models.to_csv(f"model_exploration/{i}_{fps}.csv")

if __name__ == "__main__":
    Main()
