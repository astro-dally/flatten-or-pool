# Results

Outputs extracted from the two notebooks. Runs 1 and 2 of the ablation come from earlier executions of `cv-project-h1.ipynb` (as reported in the presentation); run 3 is the saved version of the notebook.

## Tables

| File | Contents |
|:--|:--|
| `tables/reproduction_vs_paper.csv` | Per-class sensitivity and specificity, Wong et al. (2025) vs. our reproduction |
| `tables/leakage_audit.csv` | Accuracy on all, duplicated and unseen test images |
| `tables/flatten_vs_gap_three_runs.csv` | Flatten vs. GAP metrics for all three runs |
| `tables/run3_ablation_table.csv` | Full metric table for run 3 |
| `tables/run3_prediction_check.csv` | Pre-registered predictions vs. run-3 outcome |
| `tables/run3_misclassifications.csv` | Error counts by true and predicted class, run 3 |

## Training histories

Per-epoch loss and accuracy (epochs 1–50 are stage 1, 51–100 are stage 2). Validation values are those logged during training on 672 of the 800 validation images.

| File | Run |
|:--|:--|
| `histories/reproduction_history.csv` | Reproduction (Flatten) |
| `histories/run3_history_flatten.csv` | Ablation run 3, Flatten |
| `histories/run3_history_gap.csv` | Ablation run 3, GAP |

## Figures

| File | Contents |
|:--|:--|
| `figures/augmentation.png` | Offline augmentation example |
| `figures/reproduction_learning_curves.png` | Reproduction training curves |
| `figures/reproduction_confusion_matrix.png` | Reproduction confusion matrix |
| `figures/reproduction_spot_check.png` | One random test prediction per class |
| `figures/flatten_vs_gap_three_runs.png` | Test accuracy and validation-loss change across runs |
| `figures/run3_learning_curves.png` | Flatten vs. GAP learning curves, run 3 |
| `figures/run3_confusion_flatten_vs_gap.png` | Confusion matrices, run 3 |
| `figures/missed_gliomas.png` | Missed gliomas by predicted class, runs 2 and 3 |
| `figures/run3_misclassified_flatten.png` | Most confident Flatten errors |
| `figures/run3_misclassified_gap.png` | Most confident GAP errors |
| `figures/run3_gradcam.png` | Grad-CAM, Flatten vs. GAP |
