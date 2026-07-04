# BMP Bioenergy Reproducibility Package

Version 1.0.0 accompanying the manuscript "An Explainable Machine Learning Framework for Hypothesis Generation in Biochemical Methane Potential Prediction."

## Archive and Release

Final Zenodo DOI: https://doi.org/10.5281/zenodo.21081788

GitHub release: https://github.com/Fashdan/BMP_Bioenergy_Reproducibility/releases/tag/v1.0.0

## Quick Check

```powershell
python run_all.py --quick-check --output-dir reproduction_quick_check
```

## Full Regeneration

```powershell
python run_all.py --output-dir reproduction_output
```

The workflow starts from the Lallement et al. source table, reconstructs the modeling table, applies the DM >= 15% and completeness filters, and regenerates the reported validation, prediction-error, target-sensitivity, explainability, uncertainty, and row-level transparency outputs.

## Journal Supplementary Data Files

The journal-uploaded supplementary data files are:

- `Supplementary_Dataset_Row_Level_OOF_Predictions.csv`
- `Supplementary_Dataset_Data_Dictionary.csv`
- `Supplementary_Dataset_README.pdf`

This repository contains the two CSV files above under `supplementary_data_files/`. It also contains `supplementary_data_files/Supplementary_Dataset_README.txt`, the repository-readable counterpart of the journal README PDF. The PDF is supplied separately to the journal and is not expected inside `supplementary_data_files/`; the TXT counterpart is not a fourth journal-uploaded supplementary data file.

Fold assignments, individual repeated out-of-fold predictions, selected hyperparameters, GridSearchCV candidate results, held-out SHAP outputs, scripts, environment files, and generated detailed outputs are repository-only and are stored under `results/`, `scripts/`, and the environment files in this repository. `PUBLIC_REPOSITORY_FINAL_MANIFEST.csv` and `PUBLIC_REPOSITORY_SHA256SUMS.txt` are non-self-referential integrity files: neither file lists itself or the other integrity file.


## Source Data Citation

Lallement A, Peyrelasse C, Lagnet C, Barakat A, Schraauwers B, Maunas S, Monlau F. (2023). A Detailed Database of the Chemical Properties and Methane Potential of Biomasses Covering a Large Range of Common Agricultural Biogas Plant Feedstocks. Waste, 1(1), 195-227. https://doi.org/10.3390/waste1010014
