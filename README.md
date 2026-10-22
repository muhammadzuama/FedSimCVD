## Proposed Method and Implementation

This repository contains the proposed aggregation method, model architecture, baseline implementations, and dataset preprocessing pipeline for federated learning experiments.

- [`aggregation4.py`](https://github.com/muhammadzuama/FedSimCVD/blob/main/aggregation4.py) — Implementation of the proposed aggregation method, which assigns aggregation weights based on pairwise client similarity to improve collaboration in federated learning.

- [`fedpersonalized/fullClient.ipynb`](https://github.com/muhammadzuama/FedSimCVD/blob/main/fedpersonalized/fullClient.ipynb) — Implementation of **FedPer (Federated Personalization)**, where shared model components are aggregated across clients while personalized components remain local to each client.

- [`model.py`](https://github.com/muhammadzuama/FedSimCVD/blob/main/model.py) — Defines the neural network architecture used in the experiments, including the feature extractor and classification head.

- [`baselinefed/fl_basic4.py`](https://github.com/muhammadzuama/FedSimCVD/blob/main/baselinefed/FullClient10round.ipynb) — Implementation of the full-parameter federated learning baseline, in which the entire model is aggregated across participating clients.

## Dataset and Preprocessing

- [`dataset/`](https://github.com/muhammadzuama/FedSimCVD/tree/main/dataset) — Contains the dataset-related files and preprocessing pipeline used to prepare the data for federated learning experiments.
