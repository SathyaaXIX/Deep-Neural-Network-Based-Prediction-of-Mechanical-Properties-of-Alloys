# DNN Alloy Mechanical Property Prediction

Deep Neural Network for predicting alloy mechanical properties (0.2% Proof Stress and Tensile Strength) from chemical composition and temperature. Includes comparisons against Linear Regression and Random Forest baselines, k-fold cross-validation, and feature importance analysis.

Project 23 | IMI Project

## Project Structure

```
alloy-dnn-prediction/
├── data/           # Place pmo.csv here
├── src/
│   └── train_dnn.py    # Main training/evaluation script
├── models/         # Saved model files (.h5) generated at runtime
├── outputs/        # Generated plots (.png) generated at runtime
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Place `pmo.csv` in the `data/` folder (or update the CSV path inside `src/train_dnn.py`).
2. Run the script:

```bash
python src/train_dnn.py
```

Originally written as a Google Colab notebook script; the upload/mount cells for Colab (`google.colab.files`, Google Drive) are left commented at the top of the script for reference.

## Outputs

Running the script generates:

- `target_distributions.png`, `correlation_heatmap.png`, `temp_vs_properties.png` — EDA plots
- `training_history.png` — DNN training curves
- `dnn_actual_vs_predicted.png`, `dnn_residuals.png` — DNN evaluation plots
- `model_comparison.png` — DNN vs. Linear Regression vs. Random Forest
- `feature_importance.png` — Random Forest feature importance proxy
- `kfold_cv_results.png` — 5-fold cross-validation results
- `best_dnn_alloy.h5`, `dnn_alloy_final.h5` — saved model weights

By default these are written to the working directory; move them into `models/` and `outputs/` as preferred.

## Model

Multi-output DNN (Input → 128 → 256 → 128 → 64 → Output) trained with Adam, early stopping, and learning-rate reduction on plateau. Predicts both target properties jointly.

## License

MIT
