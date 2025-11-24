# Homework 8 Answers

## Question 1

**Answer**: `nn.BCEWithLogitsLoss()`

**Reasoning**: We are doing binary classification with a single output neuron. `BCEWithLogitsLoss` is more numerically stable than `Sigmoid` + `BCELoss`.

## Question 2

**Answer**: 20073473

**Calculation**:

- **Conv2d**: `(3 * 3 * 3) * 32 + 32 = 896`
- **Linear 1**:
  - Input size after flattening: `32 * 99 * 99 = 313632`
  - Weights: `313632 * 64 = 20072448`
  - Bias: `64`
  - Total: `20072512`
- **Linear 2**: `64 * 1 + 1 = 65`
- **Total**: `896 + 20072512 + 65 = 20073473`

## Question 3

**Answer**: 0.84

**Reasoning**:

- Calculated Median Training Accuracy: `0.8175`
- Options: 0.05, 0.12, 0.40, 0.84
- Closest option is **0.84**.

## Question 4

**Answer**: 0.171

**Reasoning**:

- Calculated Standard Deviation of Training Loss: `0.1676`
- Options: 0.007, 0.078, 0.171, 1.710
- Closest option is **0.171**.

## Question 5

**Answer**: 0.88

**Reasoning**:

- Calculated Mean Test Loss (Augmented): `0.5898`
- Options: 0.008, 0.08, 0.88, 8.88
- Closest option is **0.88**.

## Question 6

**Answer**: 0.68

**Reasoning**:

- Calculated Average Test Accuracy (Last 5 Epochs, Augmented): `0.6965`
- Options: 0.08, 0.28, 0.68, 0.98
- Closest option is **0.68**.
