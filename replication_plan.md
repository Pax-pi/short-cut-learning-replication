# Replication Plan

## Stage 1

Replicate the Waterbirds / ResNet-18 shortcut-learning result from Le et al.

### Experimental Setup

Dataset: original Waterbirds-95 (`waterbird_complete95_forest2water2`)

Model: ImageNet-pretrained ResNet-18

Training:
- Epochs: 100
- Optimizer: SGD
- Learning rate: 0.001
- Weight decay: 1e-4
- Batch size: 32
- Runs: 5

### Evaluation

Waterbirds contains two target classes:
- `y = 0`: landbird
- `y = 1`: waterbird

and two background types:
- `place = 0`: land
- `place = 1`: water

Evaluate accuracy separately for the four class-background groups:

| Group | Bird | Background |
|---|---|---|
| G0 | landbird | land |
| G1 | landbird | water |
| G2 | waterbird | land |
| G3 | waterbird | water |

Report:
- G0–G3 accuracy
- Average accuracy (AVG)
- Worst-group accuracy (WGA)
- AVG-WGA gap (GAP)

### Replication Criterion

Exact numerical reproduction is not required.

The result is considered replicated if:
- overall accuracy remains relatively high;
- the conflicting bird/background groups perform substantially worse than the aligned groups;
- AVG, WGA, and GAP are in the same general range as the original result.