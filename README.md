# NeurIPS 2025 Open Polymer Prediction

My notebooks for the [NeurIPS 2025 Open Polymer Prediction](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) Kaggle competition. The task is to predict five polymer properties from a repeat-unit SMILES string: glass transition temperature (Tg), fractional free volume (FFV), thermal conductivity (Tc), density, and radius of gyration (Rg). The labels are very sparse. Of the 7,973 training polymers, only 511 have a Tg label, 737 a Tc label, 613 a density label and 614 an Rg label. FFV is the only well-covered target (7,030). So most of the work went into getting reliable per-target models out of a few hundred labels each.

## Approach

All the models are per-target and trained on masked labels, so each property uses only the rows where it is measured.

**Features.** About 2,600 descriptors per polymer after filtering:
- 2D Mordred descriptors.
- RDKit basics.
- A small physics-motivated block: sp3 fraction, heteroatom ratio, halogen, N, O and S counts, aromatic rings.
- Morgan radius-2 counts and radius-3 presence bits.
- MACCS keys.

The polymer wildcard `*` is replaced so RDKit can parse every SMILES (100% parse rate on train and test). Each block is cleaned, median-imputed and cached to parquet. Later phases add Murcko-scaffold grouping, a 768-dimensional frozen chemical language model embedding, and 2D molecular graphs for a GNN.

**Validation.** Folds are grouped by canonical SMILES (GroupKFold, 5 folds), so duplicate polymers never appear on both sides of a split. Phase 5 groups by Murcko scaffold instead, which is stricter.

**Feature selection.** A variance filter and a |r| > 0.995 correlation filter come first. Then LightGBM gain pruning keeps the union of the top 600 features per target, which takes 2,641 features down to 938.

**Models.** LightGBM and CatBoost for each target, averaged over 3 seeds. Phase 4 also has a Tanimoto-kernel SVR on the binary fingerprint bits and a per-target non-negative least-squares blend over out-of-fold predictions. Phase 5 adds post-hoc linear calibration and physical-range clipping (`artifacts/oof/lgb_calibration.json`), plus a stacking cell for a PyTorch Geometric GNN.

## Results

Out-of-fold MAE from `phase4.ipynb`, 5-fold GroupKFold, mean over seeds 111, 222 and 333:

| Target | Labelled rows | LightGBM | CatBoost |
|---|---|---|---|
| Tg (°C) | 511 | 49.21 | 51.62 |
| FFV | 7,030 | 0.00592 | 0.00589 |
| Tc | 737 | 0.0255 | 0.0255 |
| Density (g/cm³) | 613 | 0.0252 | 0.0283 |
| Rg (Å) | 614 | 1.547 | 1.594 |

LightGBM is better than or level with CatBoost on four of the five targets. FFV, which has the most labels, is the exception. The gain-pruning step was fit on all labelled rows before the CV split, so these numbers are slightly optimistic. Per-fold predictions are in `outputs/predictions/` (`oof_lgb.csv`, `oof_cat.csv`).

## Repo layout

```
phase1.ipynb        first end-to-end baseline: Mordred + Morgan + RDKit (2,444 features), LightGBM per target, competition wMAE
phase2.ipynb        feature blocks rebuilt with physics priors; label-sparsity audit; external datasets collected and sorted
phase4.ipynb        integrated pipeline: prefilters, gain pruning, GroupKFold, LightGBM + CatBoost x 3 seeds, SVR + NNLS blend
phase5.ipynb        local hybrid: scaffold folds, LM embeddings, LightGBM with saved per-fold models, calibration, GNN stacking
hybridpipeline.ipynb  research-mode version of phase 5 (resumable caches, scaffold folds, PyG GNN K-fold); code only, not executed here
track2.ipynb        modular rewrite of the phase 1 baseline
track3.ipynb        experiment on precomputed feature blocks (see notes)
neurips-preparedataset-predict.ipynb   Kaggle-side inference: offline installs, feature prep, per-target AutoGluon predictors, submission
artifacts/          fold definitions, LightGBM per-target manifests (CV MAE, params, guardrails), calibration
outputs/predictions/  OOF and test predictions, selected feature names, fold sizes
catboost_info/      CatBoost training logs
```

There is no separate phase 3 notebook. `phase4.ipynb` is the integrated version (its header reads "Phase1 x Phase3").

## Running

Python 3.10+, with `rdkit`, `mordred`, `lightgbm`, `catboost`, `scikit-learn`, `pandas`, `pyarrow` and `tqdm`. `phase5` and `hybridpipeline` also need `torch` and `torch_geometric`. Put the competition files in `competition/` (`train.csv`, `test.csv`, `sample_submission.csv`) and run the notebooks from the repo root. `phase1` and `track2` read the CSVs from the working directory instead. The first Mordred pass takes about 6–7 minutes for the training set. After that the features are read from the parquet cache.

## Data

The data is not in the repo.

- Competition data, including the supplementary `dataset1`, `dataset3` and `dataset4` files: the [competition page](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data) on Kaggle.
- `phase2` sorts a set of external polymer tables into `external/`. These are the PI1M release (`PI1M_v2.csv`), the Tg tables that ship with the LieConv-Tg code release (`PolyAskInG_3_Tg_greater than_300.csv` / `_400.csv`), and the `Bicerano_bigsmiles.csv` and `JCIM_sup_bigsmiles.csv` Tg sets. They are third-party data, so get them from their original sources.

## Notes

- `track3.ipynb` merges a physics feature block taken from `train.csv`, and that block includes the target columns. Its CV scores are therefore not valid, and I don't report them.
- The saved run of `phase4.ipynb` stops after the tree models. The SVR and NNLS blend cells are in the notebook but weren't executed in this copy.
