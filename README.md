# PISA-M(P): Prediction of Interference with Specific assay Methods

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Issues](https://img.shields.io/github/issues/vincenzo-palmacci/PISA-MP)

## Overview

This repository hosts the code for PISA-M, a machine learning classifier designed to predict compounds likely to interfere with various assay technologies, including fluorescence intensity, bioluminescence, luminescence, chemiluminescence, and fluorescence polarization.

PISA-M utilizes the BalancedRandomForest classifier from the imbalanced-learn suite, achieving a balanced accuracy greater than 0.75 across all assay technologies. The models are rigorously trained, validated, and tested using a comprehensive set of bioactivity data derived from the PubChem Bioassay database.

## Features

- Fast and accurate predictions
- Easy to integrate into existing projects
- Supports multiple input formats

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the necessary dependencies, follow these steps:

1. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```

2. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Running Predictions

You can easily run predictions for your compounds using the command line interface. Below is a simple example to get you started.

### Required Arguments

- **`--smiles`**: A list of SMILES strings for which you want to predict the interference probability.
- **`--assay`**: The type of assay for which you want to predict the interference probability: flint, bioluminescence, chemiluminescence, fp, luminescence.
- **`--out`**: The name of the output file where predictions will be saved.

### Example

To run predictions, use the following command:

```bash
python predict.py --smiles smiles.npy --dataset bioluminescence --out predictions.csv
```

In this example:
- `smiles.npy` is a file containing your list of SMILES strings.
- `bioluminescence` is the type of assay.
- `predictions.csv` is the file where the prediction results will be saved.

Make sure to replace the argument values with those specific to your dataset and desired output file.

## Acknowledgments

This project was developed in collaboration with the COMP3D group at the University of Vienna. Special thanks to the team for their invaluable support and contributions.

The authors gratefully acknowledge the support from the European Commission's Horizon 2020 Framework Programme (AIDD; grant no. 956832).
