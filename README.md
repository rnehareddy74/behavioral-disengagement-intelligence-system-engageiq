# Behavioral Disengagement Intelligence System (BDIS)

A production-style behavioral health engagement analytics pipeline which is  built to answer not just *who* will disengage, but *when*, *why*, and *what to do about it*.

---

## Why This Project Exists

Most churn prediction systems stop at a risk score. A care team gets a sorted list of high-risk patients and decides what to do manually. That is not a system  that is a sorted spreadsheet.

This project builds the full loop:

```
Who will disengage?     → Risk Scoring        (Step 2)
When will they leave?   → Survival Analysis   (Step 3)
Why are they at risk?   → SHAP Explainability (Step 4)
What type are they?     → Segmentation        (Step 5)
What should we do?      → Intervention Engine (Step 6)
```

---

## How to Run

```bash
pip install -r requirements.txt



# Step 1 — full pipeline
jupyter notebook full_pipeline_real.ipynb
```

Run order matters. We have  `patient_data.csv`. The pipeline reads from it.

---

## File Structure

```

full_pipeline_real.ipynb     → full end-to-end pipeline
patient_data.csv           → single source of truth
patient_final.csv          → full pipeline output
clinician_worklist.csv     → high priority patients for human outreach
nudge_queue.csv            → medium risk patients for automated messaging
requirements.txt           → dependencies
```

---

## Pipeline — Step by Step 

### Step 1 — Load Data

Reads `patient_data.csv` — synthetic behavioral signals across 600 patients.

Data types are verified immediately. Integer columns (`missed_checkins`, `days_since_last_login`, `session_duration_min`, `inactivity_gap_days`) 
**8 features:**

| Feature | Type | Clinical rationale |
|---|---|---|
| engagement_frequency | float | Core adherence — how often are they showing up |
| response_latency_hrs | float | Responsiveness — are they still paying attention |
| inactivity_gap_days | int | Longest absence streak — worst-case signal |
| behavioral_decay | float | Trend slope — snapshot misses a declining patient |
| session_duration_min | int | Quality of engagement, not just frequency |
| missed_checkins | int | Direct non-adherence signal |
| days_since_last_login | int | Recency — how long since last seen |
| message_open_rate | float | Are outreach messages even reaching them |

---

### Step 2 — Train / Test Split

80/20 split with `stratify=y` — ensures the churn rate is identical in both train and test sets. Without stratification, a random split could put most churners in training and leave the test set with too few positives to evaluate meaningfully.

---

### Step 3 — Model Comparison and Auto-Selection

**5 models compared:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM.

**Why compare at all:**
Model selection should be driven by evidence, not preference. Every model is trained, evaluated, and ranked. The code picks the winner automatically based on cross-validated AUC.

**Why CV AUC over test AUC:**
Test AUC depends on one random split and can be misleading. Cross-validated AUC averages across 5 folds is more robust, less sensitive to which patients happened to land in the test set.

**Why Random Forest wins:**
 Why Random Forest wins on this data: Random Forest achieved the highest CV AUC across 5 folds is  the most robust measure of generalisation. On this dataset with noisy labels and overlapping behavioral signals, Random Forest's ensemble of uncorrelated trees handles variance better than boosting methods. Gradient Boosting, XGBoost, and LightGBM converged to similar performance — when models are this close, the winner is determined by how well each handles the noise structure of the specific dataset, not by a general rule.
Why not keep GBM despite similar performance: The code auto-selects by CV AUC. Random Forest won that comparison on this data. In production with a larger dataset, GBM or LightGBM would likely pull ahead — boosting methods generally scale better. But on 600 noisy patients, Random Forest is the honest winner.

**Why not XGBoost or LightGBM:**
All three perform similarly on datasets of this size. XGBoost adds L1/L2 regularisation and is  more useful on very high-dimensional sparse data. LightGBM is significantly faster on 100k+ rows and is irrelevant at 600 patients. GBM is sklearn-native, no extra dependency, and works directly with SHAP TreeExplainer.

**Why not LSTM:**
LSTMs require dense, regular time sequences. Real patient engagement data is sparse, irregular, and full of gaps. GBM handles this natively. LSTMs and GBMs are complementary tools . LSTM for sequence modeling, GBM for structured risk scoring. They are not competitors.

**Expected performance:**
- AUC: 0.74–0.78 — realistic for noisy behavioral data
- Accuracy: 0.76–0.83 — intentionally not perfect
- If accuracy exceeds 0.90 the model is likely overfitting

---

### Step 4 — Survival Analysis

**Why survival analysis on top of risk scoring:**
Risk score answers *who* will disengage. Survival analysis answers *when*. Two patients can both have a 0.70 risk score where one is  projected to disengage in 5 days, another in 6 weeks. They should not receive the same response.

**In real data:**
```python
duration_days  = last_active_date - enrollment_date
event_observed = 1  # confirmed dropout
                0  # still active — censored
```
Censored patients are not ignored — KM uses partial information correctly: "this patient survived at least until today."

**Kaplan-Meier curves:**
Non-parametric survival curves by risk tier. The median survival time is where the curve crosses 50% , half the group has disengaged by that day. If the curve never crosses 50%, most patients are still engaged — that is a good outcome, not an error.

**Assumption — proportional hazards:**
The ratio of dropout rates between groups stays constant over time.  If KM curves cross, the assumption is violated. In real data, Schoenfeld residuals test confirms this formally.

**Log-rank test:**
KM curves can look different by chance. The log-rank test asks whether the difference is statistically real.

- H₀: Survival curves of High-risk and Low-risk patients are identical
- H₁: Curves are different — one group drops out significantly faster
- p < 0.05 → reject H₀ → risk tiers genuinely separate dropout timing

Second assumption: censoring is independent — patients still active dropped out of observation for reasons unrelated to their risk level.

**Cox Proportional Hazards:**
Quantifies which features drive dropout timing. Outputs hazard ratios per feature.

| HR value | Meaning |
|---|---|
| HR = 1.0 | No effect on dropout timing |
| HR = 1.8 | One unit increase → dropout 1.8× faster |
| HR = 0.7 | One unit increase → protective, slows dropout |

**Results from this project:**
Only 2 features were statistically significant in Cox:
- `response_latency_hrs` (p < 0.005, HR = 1.37) — slower responses = faster dropout
- `message_open_rate` (p < 0.005, HR = 0.77) — higher open rate = protective

`inactivity_gap_days` was not significant in Cox (p = 0.36) despite being important in the risk score. This is an honest finding — inactivity predicts *whether* someone will churn but not necessarily *how fast*. The two models are measuring different things.

**Concordance Index = 0.71:**
Survival model equivalent of AUC. Measures how often the model correctly ranks which patient drops out first — not just yes/no, but ordering. 0.5 = random, 1.0 = perfect. Realistic range: 0.65–0.75.

---

### Step 5 — SHAP Explainability

**Why SHAP:**
A risk score without explanation is a black box. Clinicians will not act on scores they cannot interpret. SHAP provides per-patient feature attributions — not just which features matter globally, but which feature is driving *this specific patient's* risk the most and in which direction.

**Why TreeExplainer:**
Exact (not approximate) SHAP values for tree-based models. KernelSHAP is model-agnostic but orders of magnitude slower and approximate. TreeExplainer is the right tool here.

**What SHAP gives that feature importance does not:**
Feature importance tells you globally which features matter. SHAP tells you for each patient the direction and magnitude of every feature's contribution. Two patients with the same risk score can have completely different drivers.

**Outputs:**
- Global summary bar chart — which features matter most across the cohort
- Beeswarm plot — direction and magnitude per patient per feature
- Waterfall plot — full breakdown for the highest-risk patient
- Dependence plot — how SHAP value for `inactivity_gap_days` shifts across its range
- Top driver per patient — surfaced in the clinician worklist so coordinators know exactly what to address

---

### Step 6 — Behavioral Segmentation

**Why segment at all:**
Risk tier tells you urgency. Segment tells you type. A High-risk patient who was stable for months and suddenly went silent needs a different approach than one who has been erratically engaging for weeks. Score alone is insufficient for intervention design.

**Algorithm:** KMeans on StandardScaler-transformed features. PCA is used only for 2D visualization — not part of the clustering model.

**Choosing k — elbow method + silhouette scores:**

| k | Silhouette |
|---|---|
| 2 | 0.3834 |
| 3 | 0.2971 |
| 4 | 0.2151 |
| 5 | 0.1648 |

Silhouette score favoured k=2, but two clusters simply replicates the churn label — engaged vs not engaged — which adds no value beyond what the risk score already tells us. k=3 was chosen as the best balance between statistical validity and clinical utility. The three segments — Stable, Decaying, and Erratic — each map to a distinct intervention strategy: no action, automated nudge, and human outreach respectively. The goal of segmentation is actionability, not just statistical optimality.

**Why not k=4:**
PCA visualisation showed two of the four clusters overlapping heavily. Silhouette confirmed k=4 produces poorly defined clusters. High-Risk Inactive patients did not form a separate cluster — their signals overlapped with Erratic. Forcing k=4 would create a meaningless cluster. The Erratic segment captures both unpredictable engagers and dormant patients — the intervention logic handles them identically.

**Three archetypes:**

| Segment | Profile | Intervention |
|---|---|---|
| Stable | High engagement, low decay, fast response | No action |
| Decaying | Gradual decline across all signals | Automated nudge — catch early |
| Erratic | High variance, unpredictable | Human outreach — score alone misses volatility |

---

### Step 7 — Intervention Engine

**Why not just use risk tier:**
Pure threshold routing misses two important cases:
1. An Erratic patient with a Low score today — their history of volatility warrants early contact regardless
2. A Decaying patient whose score has not crossed the High threshold yet but whose slope is steep enough to cross it before the next review

**Routing logic:**

| Condition | Action | Reason |
|---|---|---|
| Segment = Erratic | Human Outreach | Volatility is the signal — score is a snapshot |
| Segment = Decaying AND decay < −0.35 | Human Outreach | Steep slope — will cross threshold soon |
| Segment = Decaying | Automated Nudge | Gradual decline — early re-engagement |
| Risk Tier = High | Human Outreach | Score threshold exceeded |
| Risk Tier = Medium | Automated Nudge | Moderate risk — automated re-engagement |
| Risk Tier = Low | No Action | Passive monitoring |

**Why Erratic → Human Outreach regardless of score:**
An erratic patient who looks fine today has a history of sudden disappearance. A risk score is a snapshot — it misses the variance in engagement history. The pattern itself is the red flag, not just the current score.

**Outputs:**
- Clinician worklist — high priority patients sorted by risk score with top SHAP driver surfaced
- Automated nudge queue — medium risk patients queued for re-engagement messaging
- What-if simulator — change any feature value, see risk score update instantly
- Sensitivity curve — shows how risk changes across the full range of any feature for any patient

**What this is NOT:**
This is not a causal uplift model. It cannot measure whether the intervention will actually cause re-engagement — only that the patient is at risk and an action is warranted. True causal inference requires randomised intervention data and uplift modeling. That is the next frontier.

---


## Honest Limitations

**No causal inference:** Predicts risk and recommends actions. Cannot measure whether interventions actually work. Requires randomised trial data and uplift modeling.

**Synthetic data:** Generated to mimic real behavioral health engagement patterns. In production, duration comes from real timestamps and event flags from confirmed dropout records.

**Static model:** Trained on a snapshot. Real systems need periodic retraining as behavioral patterns shift over time.

**Cross-channel behavior not modeled:** Patients interact via app, email, phone, and in-person. This system sees one channel. A unified cross-channel behavioral representation is an unsolved problem at scale.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.44.0
lifelines>=0.27.0
matplotlib>=3.7.0
```
##  Future outlook 
**Causual Inference**
**Add Sequence modeling using LSTM**
