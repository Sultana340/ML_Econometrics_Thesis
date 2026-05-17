
# Experiment Log

This experiment log summarizes the five simulation experiments conducted in the thesis. The experiments evaluate the finite-sample behavior of AGMM in high-dimensional nonparametric instrumental variable settings under varying instrument strength, structural-function complexity, dimensional design, estimator choice, and adversarial training stability.

## Common Technical Setup

- **Framework:** Nonparametric instrumental variable estimation under endogeneity
- **Main estimator:** AGMM
- **Data design:** Controlled simulation using MNIST-based high-dimensional representations
- **Sample size:** `n = 2000`
- **Monte Carlo replications:** `MC = 5`
- **Training environment:** CPU-based implementation
- **Training horizon:** `200` epochs
- **Batch size:** `100`
- **Dropout probability:** `0.1`
- **Hidden dimension:** `200`
- **Optimizer:** Optimistic Adam
- **Main evaluation metrics:**
  - Early-stopping MSE
  - Final \(R^2\)

The high-dimensional designs are defined as follows:

- `z_image`: the instrument \(Z\) is high-dimensional, while the treatment \(X\) is scalar.
- `x_image`: the treatment \(X\) is high-dimensional, while the instrument \(Z\) is scalar.

---

## Experiment 1: IV Strength and Structural-Function Complexity

### Objective

This experiment evaluates how AGMM performs when instrument strength and structural-function complexity vary.

### Design

- **DGP:** `z_image`
- **Estimator:** AGMM
- **IV strength values:** \(\pi \in \{0.3, 0.6, 0.9\}\)
- **Structural functions:**
  - Linear: \(h_0(x)=x\)
  - Nonsmooth nonlinear: \(h_0(x)=|x|\)
  - Smooth nonlinear: \(h_0(x)=\sin(x)\)

### Technical Details

- Sample size: `n = 2000`
- Monte Carlo replications: `5`
- Epochs: `200`
- Batch size: `100`
- Optimizer: Optimistic Adam
- Main metrics: early-stopping MSE and final \(R^2\)

### Result Line

The results show that stronger instruments improve structural recovery, producing lower early-stopping MSE and higher final \(R^2\). Nonlinear structural functions, especially \(\sin(x)\), are more difficult to recover than the linear benchmark.

### Purpose

This experiment establishes the baseline simulation evidence. It examines whether stronger instruments improve structural recovery and whether nonlinear structural functions are harder to estimate than the linear benchmark.

---

## Experiment 2: Location of High Dimensionality

### Objective

This experiment studies whether AGMM behaves differently when high dimensionality enters through the instrument space or through the treatment space.

### Design

- **Estimator:** AGMM
- **Compared designs:**
  - `z_image`: high-dimensional instrument \(Z\), scalar treatment \(X\)
  - `x_image`: high-dimensional treatment \(X\), scalar instrument \(Z\)

### Technical Details

- Sample size: `n = 2000`
- Monte Carlo replications: `5`
- Epochs: `200`
- Batch size: `100`
- Same neural architecture and optimizer settings as Experiment 1

### Result Line

The `z_image` design produces more stable structural recovery, while the `x_image` design performs poorly and can generate strongly negative final \(R^2\) values. This indicates that AGMM is more reliable when high dimensionality enters through the instrument space rather than the treatment space.

### Purpose

This experiment identifies whether AGMM is more reliable when the high-dimensional information enters through the instrument rather than through the treatment. The results also justify the later focus on the `z_image` design.

---

## Experiment 3: Kernel-Based Variants vs. Baseline AGMM

### Objective

This experiment compares baseline AGMM with kernel-based AGMM variants to evaluate whether kernelization improves structural recovery or training stability.

### Design

- **DGP:** `z_image`
- **IV strength:** \(\pi = 0.6\)
- **Structural functions:**
  - \(h_0(x)=|x|\)
  - \(h_0(x)=\sin(x)\)

### Estimators

- AGMM
- KernelLayerMMDGMM
- CentroidMMDGMM

### Technical Details

- Sample size: `n = 2000`
- Monte Carlo replications: `5`
- Epochs: `200`
- Kernel function: Gaussian kernel
- Kernel features: \(g_{\mathrm{features}}\), commonly set around `10`
- Centroid choices: \(n_{\mathrm{centers}}\), commonly tested around `25`
- Main metrics:
  - Early-stopping MSE
  - Final \(R^2\)
  - Run-wise Monte Carlo stability

### Result Line

KernelLayerMMDGMM performs competitively in some settings, especially for \( |x| \), but its gains over AGMM are not uniform. CentroidMMDGMM is highly unstable and produces poor final \(R^2\) values, so it should be interpreted as a weak robustness variant rather than an improvement over baseline AGMM.

### Purpose

This experiment assesses whether kernel-based variants provide systematic improvements over baseline AGMM. It also clarifies whether these variants should be treated as reliable improvements or as robustness checks whose performance depends on architecture and regularization.

---

## Experiment 4: AGMM vs. Neural IV Benchmarks

### Objective

This experiment compares AGMM with alternative neural instrumental-variable estimators.

### Design

- **DGP:** `z_image`
- **Structural functions:**
  - \(h_0(x)=|x|\)
  - \(h_0(x)=\sin(x)\)

### Estimators

- AGMM
- DeepIV
- DeepGMM

### Technical Details

- Sample size: `n = 2000`
- Monte Carlo replications: typically `3` to `5`, depending on computational feasibility
- CPU-based implementation
- The comparison is restricted mainly to the `z_image` design because DeepIV and DeepGMM are more compatible with scalar or low-dimensional treatment settings.

### Result Line

AGMM achieves stronger structural recovery than the neural IV benchmarks in the controlled `z_image` design, producing lower MSE and more favorable final \(R^2\) values than DeepIV and DeepGMM under the tested nonlinear structural functions.

### Purpose

This experiment evaluates whether AGMM provides stronger structural recovery than existing neural IV benchmarks in a controlled high-dimensional instrument setting.

---

## Experiment 5: Stability and Convergence Diagnostics

### Objective

This experiment examines whether the reported AGMM results are stable across Monte Carlo replications and whether the adversarial training procedure exhibits reasonable convergence behavior.

### Design

The experiment evaluates three diagnostic components:

1. Monte Carlo stability using the cumulative standard error of final \(R^2\)
2. Run-wise early-stopping MSE across Monte Carlo replications
3. Moment-loss convergence across training epochs

### Estimators

- Main convergence analysis: AGMM
- Run-wise stability comparison:
  - AGMM
  - KernelLayerMMDGMM
  - CentroidMMDGMM

### Technical Details

- Monte Carlo replications: `5`
- Training horizon: `200` epochs
- Main diagnostic metrics:
  - Cumulative standard error of final \(R^2\)
  - Run-wise early-stopping MSE
  - Averaged absolute moment loss across epochs

### Result Line

The diagnostics show that AGMM's results are not driven by isolated Monte Carlo replications. The cumulative standard error decreases as replications increase, and the moment-loss curves stabilize before `200` epochs, supporting the chosen Monte Carlo and training-budget settings under CPU constraints.

### Purpose

This experiment supports the reliability of the simulation results by checking that the findings are not driven by isolated Monte Carlo replications or by an insufficient training horizon. It also justifies the use of five Monte Carlo replications and a 200-epoch training budget under CPU constraints.

