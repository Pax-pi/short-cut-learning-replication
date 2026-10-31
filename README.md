# Shortcut Learning Replication

Independent replication of selected experiments from:

Le, P. Q., Schlötterer, J., & Seifert, C.
"An XAI-Based Analysis of Shortcut Learning in Neural Networks." [Publisher](https://doi.org/10.1007/978-3-032-08327-2_20) [arXiv](https://doi.org/10.48550/arXiv.2504.15664)

## Stage 1: Waterbirds / ResNet-18 baseline

This stage reconstructs the Waterbirds / ResNet-18 baseline using the
original Waterbirds-95 dataset.

Evaluation metrics:

- **AVG:** Overall accuracy (correct predictions / total samples).
- **WGA:** The minimum accuracy across the four `(label, background)` groups.
- **GAP:** AVG − WGA.

| Waterbirds / ResNet-18 | AVG | WGA | GAP |
|---|---:|---:|---:|
| Paper, Table 2 | 83% ± 1 pp | 46% ± 2 pp | 37 pp |
| This reconstruction, 5 runs | 81.49% ± 0.56 pp | 37.17% ± 2.52 pp | 44.33 pp |

Here, `pp` denotes percentage points. The paper reports the mean and standard deviation across five runs; its standard-deviation convention is not specified. For this reconstruction, AVG and WGA are reported as the mean ± standard deviation across five seeds (`np.std` with `ddof=0`). GAP is the difference between the unrounded AVG and WGA means; no standard deviation is reported for GAP.

## Implementation decisions

The paper specifies an ImageNet-1K-pretrained ResNet-18 fine-tuned on Waterbirds for 100 epochs using SGD, a learning rate of 0.001, weight decay of $10^{-4}$, and a batch size of 32. This reconstruction follows those settings and evaluates the model on the original test split.

The paper does not fully specify the image preprocessing, data augmentation, remaining optimizer settings, or checkpoint selection rule. The following choices define this reconstruction:

| Component | Choice in this implementation |
|---|---|
| Image preprocessing | Resize to 256 × 256, center crop to 224 × 224, convert to a tensor, and normalize using ImageNet mean and standard deviation for all three splits |
| Data augmentation | No random augmentation |
| Optimization | Cross-entropy loss and SGD without momentum or a learning-rate scheduler |
| Checkpoint selection | Select the epoch with the lowest validation loss, then evaluate that checkpoint on the test split |

These choices are documented because the paper's description does not determine a unique implementation. Their possible contribution to the difference in worst-group accuracy has not yet been established.

## Results provenance and limitations

The reported results come from five 100-epoch runs with seeds 42–46 on Google Colab with a CUDA GPU. The results JSON was retained, but the model checkpoints and a snapshot of the runtime environment were not. Based on the experiment timeline and subsequent Git changes, the runs are believed to correspond to commit `2ec32be732232c51f52236964fb0e139fabb00c2`.

Explicit DataLoader worker seeding was added after these runs. The reported metrics therefore describe the earlier code version. An end-to-end run from a fresh environment using the current code has not yet been verified.

## Running Stage 1

### Data preparation

This project uses the original Waterbirds-95 dataset, `waterbird_complete95_forest2water2`, described in the [Group DRO data instructions](https://github.com/kohpangwei/group_DRO#waterbirds). The dataset is not included in this repository.

Obtain and extract the dataset so that `metadata.csv` is located at:

```text
data/waterbird_complete95_forest2water2/metadata.csv
```

The image paths listed in `metadata.csv` must also exist under that directory. The current availability of the original download endpoints has not been verified, so dataset acquisition is currently a manual step.

### Run the experiment

From the repository root, in a Python environment with PyTorch, torchvision, pandas, NumPy, and Pillow installed:

```bash
python run_experiment.py
```

This runs five seeds (42–46), evaluates each selected checkpoint on the test split, and writes metrics to `results.json`. An existing `results.json` is overwritten after each seed and again when the final summary is added; if interrupted, it may contain only the completed seeds and no `final` entry. The file `best_model.pth` is overwritten by subsequent seeds; it does not preserve five separate checkpoints.

These instructions describe the current code. An end-to-end run from a fresh environment has not yet been verified.

[Per-seed results from the October 6, 2026 Colab run](results/stage1_colab_five_runs_2026-10-06.json).