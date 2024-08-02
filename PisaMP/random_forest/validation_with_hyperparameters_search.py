import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import os

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import optuna

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from PisaM.path import derived, metrics, splits

# Ignoring warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

"""
Random Forest Classifier: Validation with hyperparameters search.
"""


def _compute_morgan(smile):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    molecule = Chem.MolFromSmiles(smile)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
    morgan_fp = fpgen.GetFingerprint(molecule)
    
    return morgan_fp


def _suggest_hyperparameters(trial:optuna.trial.Trial) -> list:
    """
    Suggest hyperparameters for optuna search.
    """
    # Suggest number of trees.
    n_estimators = trial.suggest_int("n_estimators", low=32, high=500, step=128)
    # Suggest trees depth.
    max_depth = trial.suggest_int("max_depth", low=16, high=1024, step=16)
    # Suggest minimum samples split.
    min_samples_split = trial.suggest_int("min_samples_split", low=2, high=5, step=1)
    # Suggest maximum number of features per split.
    max_features = trial.suggest_int("max_features", low=2, high=2048, step=1)

    print(f"Suggested hyperparameters: \n{(trial.params)}")
    return trial.params


def _get_labels(dataset):
    """
    Compute binary interference labels.
    """
    labels = pd.read_csv(metrics / "delta.csv", index_col=0)[dataset].values
    return labels


def _get_threshold(labels):
    return labels.mean(), labels.std()


def objective(trial:optuna.trial.Trial, X, y) -> float:
    """
    Search for optimal set of hyperparameters via cross validation.
    """
    # Get suggested hyperparameters.
    hyperparameters = _suggest_hyperparameters(trial)
    print(hyperparameters)

    # Load splits indexes
    folds_indices = [np.load(splits / i, allow_pickle=True).tolist() for i in os.listdir(splits) if i != "test.npy"]

    scores = []
    for idx in range(len(folds_indices)):
        # Get training and test indexes.
        test_idx = folds_indices[idx]
        train_idx = []
        for tmp in range(len(folds_indices)):
            if tmp != idx:
                train_idx += folds_indices[tmp]
        
        # Get training and validation sets.
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Binarize labels.
        cut = _get_threshold(y_train)
        y_train = (y_train > cut[0] + 1 *cut[1]).astype(int)
        y_test = (y_test > cut[0] + 1 *cut[1]).astype(int)

        # Instanciate the random forest classifier with the suggested hyperparameters.
        model = BalancedRandomForestClassifier(
                    n_estimators = hyperparameters["n_estimators"],
                    max_depth = hyperparameters["max_depth"],
                    min_samples_split = hyperparameters["min_samples_split"],
                    max_features = hyperparameters["max_features"],
                    bootstrap = True,
                    n_jobs = 16,
                    )

        # Fit training data.
        model.fit(X_train, y_train)
        # Compute prediction.
        y_pred = model.predict(X_test)
        # Compute performances.
        metric = matthews_corrcoef(y_test, y_pred)
        scores.append(metric)

    return np.mean(scores)


@click.command()
@click.option("-d", "--dataset", required=True, help="Specify the detection method {fp, biolumnescence, flint, luminescence, chemiluminescence}")
def Main(dataset):
    """
    Random Forest Classifier: validation with hyperparameters search.
    """
    # Load dataa.
    print("\n Loading data ...")
    data = pd.read_csv(derived / "matrix.csv", index_col="smiles")
    smiles = data.index.tolist() # get samples
    # Get labels.
    labels = _get_labels(dataset)

    # Compute molecules descriptors.
    fingerprints = np.array([_compute_morgan(i) for i in tqdm(smiles)], dtype=float)
    
    print("\nDataset specs: ")
    print("\t# Compound:", fingerprints.shape[0])
    print("\t# features:", fingerprints.shape[1])

    # Run hyperparameters search with optuna.
    study_name = f"{dataset}.run"
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=optuna.pruners.HyperbandPruner(), sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, fingerprints, labels), n_trials=50)

    best_trial = study.best_trial.params

    # Save best hyperparameters.
    np.save(os.path.join("results/validation", f"{dataset}.npy"), best_trial)

    return print("Model validated!")


if __name__=="__main__":
    Main()
