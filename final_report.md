# Diabetes 30-Day Readmission Risk Stratification

## Machine Learning + Random Matrix Theory (RMT) + Fairness Analysis

---

## Executive Summary

Using the UCI “Diabetes 130-US hospitals” dataset, we built baseline ML models and layered in RMT diagnostics and fairness evaluation. Best test AUROC was **0.6862 (XGBoost)**. The apparent **88.9% accuracy at a global threshold of τ=0.4765** is almost entirely due to class imbalance (positives = 11.2%); at that operating point the model has **TPR 1.6%** and **TNR 99.89%**—i.e., it predicts almost everyone as non-readmitted.  
RMT revealed an extremely ill-conditioned feature space (effective rank ≈ 24.7 out of 2,232; condition number → ∞).  
A follow-up mini-model applied **RMT-guided reduction (2232→25)** and **subgroup-aware thresholds**, cutting age accuracy gaps by **83.7%** (with an AUROC tradeoff).

---

## Data & Preprocessing

- **Encounters:** 101,766  
- **Outcome prevalence:** 11.2% (<30 day readmission)  
- **Splits:** 60/20/20 stratified (Train/Valid/Test = 61,059 / 20,353 / 20,354)  
- **Features:** 47 raw → 2,232 processed after one-hot encoding of categorical fields (notably diag_1/2/3 ICD-9 codes, payer_code, medical_specialty) + standardized numerics

---

## Baseline Models & Accuracy Decomposition

**Test AUROC (95% CI):**  
- Logistic Regression: **0.6391 [0.626–0.651]**  
- XGBoost: **0.6862 [0.674–0.698]**  
- MLP: **0.5115 [0.499–0.524]**  

**Operating point (global threshold τ=0.4765, chosen on validation):**  
- Accuracy: **88.93%**  
- Sensitivity (TPR): **1.63%**  
- Specificity (TNR): **99.89%**  

**Confusion-matrix scale (≈20,354 test samples, 11.2% positives):**  
- Positives ≈ 2,280 → TP ≈ 37, FN ≈ 2,243  
- Negatives ≈ 18,074 → TN ≈ 18,054, FP ≈ 20  

**Why accuracy looks high:** With 88.8% negatives, predicting almost everyone as negative yields ~88.8% accuracy by default. The model’s τ=0.4765 point is effectively that—high accuracy driven by class imbalance, with near-zero recall. For clinical triage, this offers little utility.

---

## Random Matrix Theory (RMT): Math & Findings

### A. Marchenko–Pastur (MP) Law (noise baseline)

Let **X ∈ R^{n×p}** be standardized features; sample covariance **S = (1/n) XᵀX**.  
Define γ = p/n. If X were pure noise, the eigenvalues {λᵢ} of S concentrate in:

**λ± = (1 ± √γ)²**  

- Here, n=61,059, p=2,232 ⇒ γ≈0.0366 ⇒ MP bounds ≈ [0.654, 1.419].  
- Interpretation:  
  - λ ∈ [λ−, λ+] = bulk noise  
  - λ > λ+ = signal spikes (directions with variance beyond noise)  
  - λ ≈ 0 = redundancy/collinearity  

**Observed:** overwhelming bulk within bounds, many near zero; 2,225 outside bounds but only a small subset act as stable “signal” directions (see effective rank).

---

### B. Ill-conditioning, Effective Rank, and Eigenspectrum

- **Condition number:** κ = λmax / λmin. We observed λmin → 0 ⇒ κ→∞.  
  → Exact or near-exact linear dependencies (classic with one-hot blocks).  

- **Effective rank:**  
  r_eff = (Σᵢ λᵢ)² / Σᵢ λᵢ² ≈ 24.7  
  Despite p=2232, only ~25 directions carry most usable variance.

- **Eigenspectrum:** long tail of near-zero eigenvalues + small number above MP upper edge ⇒ strong redundancy, a few coherent directions of signal.

---

### C. Subgroup Spectra & Fairness

- Age bands show different effective ranks (e.g., ~9 for 20–30 vs ~24 for 90–100).  
- Implies different geometry in feature space by subgroup.  
- Helps explain age-dependent sensitivity gaps at a single global threshold.

---

## Scaling via Power Law (Beyond Reporting AUC@N)

We tested performance as data grew:  
- 30% (≈20k): 0.6647  
- 70% (≈41k): 0.6701  
- 100% (≈61k): 0.6862  

In many ML regimes, error follows a power law:  

**Error(N) ≈ A·N^{−α} + C** (equivalently, 1−AUC).  

- Attempting to fit (A, α, C) requires multiple clear points.  
- Curve was flat (+0.54% AUC for >2× data), insufficient to fit a stable α.  
- Meaning: scaling plateau—more data won’t meaningfully help given current features.  
- Bottleneck is representation/geometry, not sample size.

---

## Fairness: Definitions, Results, and Interpretation

### What we measure
- **Accuracy parity:** range of accuracies across groups (smaller = fairer)  
- **Sensitivity parity:** range of TPR across groups  
- **Specificity parity:** range of TNR across groups  

---

### Big model (global τ=0.4765)
- **Age:** accuracy 87.2%→100.0% ⇒ gap 12.8 pts; sensitivity 0%→16.3%  
  → Unfair across age, especially underserving older groups (low TPR).  
- **Gender:** gap 0.8 pts → excellent.  
- **Race:** gap ~4.6 pts → excellent overall, with small-n instability in tiny groups.  

**Interpretation:** A single global threshold overlays different score distributions. Age groups with flatter/shifted distributions (consistent with their spectral differences) get very low recall.

---

### Mini-model (RMT reduction + subgroup-aware thresholds)
- **RMT-guided PCA:** 2232 → 25 comps (≈89× compression).  
- **AUROC tradeoff:** Logistic 0.6391→0.6126 (−4.15%); XGBoost 0.6755→0.6056 (−10.34%).  
- **Subgroup-aware thresholds:**  
  - Age accuracy range: 0.505 → 0.082 (+83.7% improvement)  
  - Gender range: 0.021 → 0.007 (+66.1%)  
  - Race range: 0.116 → 0.051 (+55.9%)  

**Interpretation:** Choosing per-group thresholds aligns operating points with each group’s score distribution, closing performance gaps substantially. This is a pragmatic fairness fix.  

---

### What “good” looks like
- Accuracy gaps <5% across major groups: **excellent**  
- 5–10%: **acceptable/monitor**  
- >10%: **concerning**  

After subgroup thresholding, your largest gap (age) is ~8.2 pts → **acceptable, and vastly better than baseline**.

---

## What Changed & Why It Matters

- **Accuracy truth:** 88.9% accuracy is a class-imbalance artifact.  
- **RMT audit:** Feature space is effectively low-rank, ill-conditioned, heterogeneous by subgroup. Explains instability + fairness gaps.  
- **Mini-model:** Demonstrated geometry-aware compression + subgroup thresholding reduce demographic gaps, even if AUROC drops.  

---

## Limitations

- Reduced model AUROC = info loss with naïve PCA; better compression methods may help.  
- Subgroup thresholds improve parity in accuracy but may trade off precision/recall.  
- Calibration not reported (important for threshold stability).  
- Temporal drift + coding practices not modeled.

---

## What’s Next (guided by RMT & scaling)

- **Representation over raw one-hot:** Use supervised low-rank transforms or embeddings.  
- **Cost-aware thresholds:** Pick τ to meet recall or cost constraints; report PR-AUC.  
- **Calibration & parity:** Calibrate probabilities (Platt/isotonic) and check parity across age.  
- **Spectral monitoring:** Track effective rank and eigen-spike patterns; link to fairness gaps.  
- **Stable ensembles:** Use ensembles with explicit colsample/subsample tuning for collinearity.

---

## Conclusion

The dataset’s geometry (low effective rank, high collinearity, subgroup spectral differences) limits performance and drives fairness gaps.  
The big model achieves superficially high accuracy only because of class imbalance, while missing almost all true readmissions.  

RMT provided the diagnostic lens to see why, and the mini-model showed two practical remedies—lower-rank representations and subgroup-aware thresholding—that materially improve fairness.  

Further gains will come from better representations, calibrated/cost-aware thresholds, and ongoing spectral/fairness monitoring.
