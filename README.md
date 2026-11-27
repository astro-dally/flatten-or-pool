# Flatten or Pool?

**Reproducing VGG16 transfer learning for brain tumour MRI classification, and testing Global Average Pooling as a remedy for overfitting**

Dally R (231043) · Kalash Kumari Thakur (230136)\
Newton School of Technology, Rishihood University

[Paper (PDF)](paper/FlattenOrPool_IEEE_Paper.pdf) · [Reproduction notebook](https://www.kaggle.com/code/astrodally/cv-project) · [Ablation notebook](https://www.kaggle.com/code/astrodally/cv-project-h1)

## Summary

Wong *et al.* (2025) report 99.24% accuracy on four-class brain tumour MRI classification (glioma, meningioma, pituitary tumour, no tumour) using an ImageNet-pretrained VGG16 with a Flatten–Dense head. We reimplemented their pipeline and obtained **95.25%**. We also found that **202 of the 800 test images are exact duplicates of training images**.

We then tested a pre-registered hypothesis: replacing the 6.42 M-parameter Flatten head with a 0.13 M-parameter Global Average Pooling (GAP) head should reduce overfitting. We changed only the head and ran the experiment three times. GAP consistently reduced validation loss while the VGG16 backbone was frozen. Once fine-tuning began, both heads overfit equally, and final accuracy changed by +0.50, −3.13 and −0.50 points.

**Conclusion:** fine-tuning, not the classifier head, drives the overfitting.

## Results

**Reproduction**

| | Accuracy |
|:--|--:|
| Wong et al. (2025), reported | 99.24% |
| Ours, full test set (n = 800) | 95.25% |
| Ours, leaked test images (n = 202) | 99.50% |
| Ours, unseen test images (n = 598) | 93.81% |

Glioma is the weakest class (84.5% recall); most missed gliomas are predicted as meningioma.

**Flatten vs. GAP (three identical runs)**

| Metric | Run 1 | Run 2 | Run 3 |
|:--|:--:|:--:|:--:|
| Test accuracy, Flatten → GAP (%) | 94.50 → 95.00 | 95.38 → 92.25 | 95.13 → 94.63 |
| Glioma recall, Flatten → GAP (%) | 81.5 → 83.0 | 85.0 → 72.5 | 82.5 → 81.5 |
| Final validation loss, GAP vs. Flatten | −15% | +46% | +3.8% |
| Stage-1 validation loss, GAP vs. Flatten | **−42%** | **−47%** | **−33%** |

None of the five pre-registered predictions held in all three runs. The only effect that replicated was the stage-1 reduction in validation loss.

**Why the effect disappears:** unfreezing VGG16 blocks 4–5 for fine-tuning adds 12.98 M trainable weights. GAP's advantage in trainable parameters then drops from 48.5× (6.42 M vs. 0.13 M) to 1.48× (19.40 M vs. 13.11 M). The runs also diverged despite a fixed seed, most likely because of GPU non-determinism, producing a 2.75-point spread for GAP.

## Repository

```
├── paper/
│   ├── FlattenOrPool_IEEE_Paper.pdf
│   └── latex/                  # LaTeX source and figures
├── notebooks/
│   ├── cv-project.ipynb        # reproduction of Wong et al. (2025)
│   └── cv-project-h1.ipynb     # Flatten vs. GAP ablation and failure analysis
└── results/                    # ablation table, training histories, figures
```

## Method

```
MRI → grayscale 224×224 → offline augmentation ×3 → VGG16 (ImageNet) → Flatten | GAP → Dense(256) → Softmax(4)
```

- **Stage 1:** backbone frozen, Adam, lr 1e-3, 50 epochs.
- **Stage 2:** VGG16 blocks 4–5 unfrozen, Adam, lr 1e-4, 50 epochs.
- Batch size 32, 179 steps per epoch, seed 42.
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,200 images, 4 classes), split into 16,800 training images (after augmentation), 800 validation and 800 test.
- **Leakage audit:** 64-bit difference hashes, with exact matches between training and test images flagged.

## Running the code

The notebooks are built for Kaggle and take about 2–2.5 hours on a T4 GPU.

1. Open the [reproduction](https://www.kaggle.com/code/astrodally/cv-project) or [ablation](https://www.kaggle.com/code/astrodally/cv-project-h1) notebook and click **Copy & Edit**.
2. Attach the Brain Tumor MRI Dataset, and set the accelerator to **GPU T4 ×2**.
3. Run all cells. Results are written to `work/results/`.

To run locally:

```bash
pip install tensorflow tf_keras kagglehub seaborn scikit-learn pillow dash
```

Then set `DATA_ROOT = None` in the configuration cell; the dataset downloads automatically via `kagglehub`.

The trained models (240 MB and 164 MB) exceed GitHub's file-size limit. Download them from the Output tab of the [ablation notebook](https://www.kaggle.com/code/astrodally/cv-project-h1).

Environment: TensorFlow 2.20.0, tf_keras 2.20.1, 2× NVIDIA Tesla T4.

## Limitations

The dataset has no patient identifiers, so the split is per image and slices from the same patient may cross splits. Three runs per model show the spread but are too few for confidence intervals. Our dataset release (7,200 images) differs from the paper's (7,023). This is a methods study; the models are not validated for clinical use.

## Citation

```bibtex
@misc{dally2026flattenorpool,
  title  = {Flatten or Pool? Reproducing VGG16 Transfer Learning for Brain Tumour
            MRI Classification and Testing Global Average Pooling as a Remedy for Overfitting},
  author = {Dally R and Kalash Kumari Thakur},
  year   = {2026},
  note   = {Newton School of Technology, Rishihood University}
}
```

Base paper: Y. Wong, E. L. M. Su, C. F. Yeong, W. Holderbaum, and C. Yang, "Brain tumor classification using MRI images and deep learning techniques," *PLoS ONE*, 20(5): e0322624, 2025.
