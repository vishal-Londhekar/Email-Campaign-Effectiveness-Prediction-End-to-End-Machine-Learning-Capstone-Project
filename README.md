# 📧 Email Campaign Effectiveness Prediction — End-to-End Multi-Class Classification ML Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Multi--Class%20Classification-orange?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Final%20Model-XGBoost-brightgreen" />
  <img src="https://img.shields.io/badge/ROC--AUC-93%25-success" />
  <img src="https://img.shields.io/badge/Domain-Email%20Marketing%20%7C%20CRM-blueviolet" />
  <img src="https://img.shields.io/badge/Imbalance-SMOTE%20Applied-red" />
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen" />
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/vishal-Londhekar/Email-Campaign-Effectiveness-Prediction-End-to-End-Machine-Learning-Capstone-Project/blob/main/Email_Campaign_Effectiveness_Prediction_End_to_End_Machine_Learning_Capstone_Project.ipynb" target="_parent">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [Project Objective](#-project-objective)
- [Dataset Description](#-dataset-description)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering & Preprocessing](#-feature-engineering--preprocessing)
- [Model Development](#-model-development)
- [Model Evaluation](#-model-evaluation)
- [Key Insights](#-key-insights)
- [Business Recommendations](#-business-recommendations)
- [Future Improvements](#-future-improvements)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🧩 Business Problem

### Industry Challenge

Email marketing remains one of the highest-ROI digital marketing channels — generating an average of **$36 for every $1 spent** across the industry. Yet for small and medium-sized businesses (SMBs) operating on lean budgets, the reality is far less optimistic. The fundamental challenge is **operational blindness**: businesses send thousands of emails without any reliable intelligence about which emails will be read, which will be ignored, and which will actually drive the customer engagement that converts to revenue.

Without this intelligence, marketing teams operate on guesswork — about subject lines, content length, send timing, campaign type, and recipient segmentation. The result is a vicious cycle: low engagement → no feedback → no improvement → continued low engagement. This project's dataset confirms the severity of the problem: **80.38% of all emails sent are completely ignored**.

### Why It Matters

| Stakeholder | Pain Point | Business Consequence |
|---|---|---|
| **Marketing Managers** | No predictive signal before sending | Campaigns launched blind; budget wasted on ignored emails |
| **CRM / Revenue Teams** | Can't identify high-engagement customer segments | Low conversion, missed upsell opportunities |
| **Email Copywriters** | No feedback loop on content effectiveness | Subject lines and body copy remain unoptimised |
| **Business Owners** | No ROI visibility on campaign spend | Cannot justify or scale email investment |
| **Customer Success Teams** | Can't distinguish engaged from churning customers | Late churn detection; reactive retention efforts |

### Business Impact — What This Solves

A machine learning model that predicts email engagement outcomes **before campaigns are sent** enables organisations to:
- **Reduce wasted campaign spend by 30–50%** by excluding or reforming content for recipient segments predicted to ignore
- **Increase acknowledged email rates** by optimising content attributes (hotness score, word count, links, images) before sending
- **Power smarter segmentation** by scoring every contact's predicted engagement tier prior to any campaign launch
- **Build a customer engagement warmth score** from `Total_Past_Communications` and predicted engagement — enabling proactive retention before churn signals appear
- **Personalise campaign timing and type** using model-derived feature signals to drive measurable uplift in open and conversion rates

---

## 🎯 Project Objective

This project delivers a **production-grade multi-class classification pipeline** — from raw data through a trained, evaluated, and business-interpreted XGBoost model — to predict whether an individual email will be **Ignored**, **Opened**, or **Acknowledged** by its recipient.

**Analytical Objectives:**
- Quantify how email type, campaign type, content attributes, send timing, recipient location, and engagement history individually drive email engagement outcomes
- Identify content and campaign characteristics most strongly associated with the rare but high-value "Acknowledged" outcome
- Validate statistical relationships between categorical features and the target using formal hypothesis testing

**Machine Learning Objectives:**
- Build and compare three classifiers: Logistic Regression, Random Forest, and XGBoost
- Address the severe 80:16:4 class imbalance using SMOTE oversampling on the training set
- Apply GridSearchCV hyperparameter tuning to all three models
- Select the final model using ROC-AUC (One-vs-Rest) as the primary imbalance-robust evaluation metric
- Explain model predictions using SHAP TreeExplainer for stakeholder-interpretable feature importance

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Source** | Gmail-based email marketing campaign records (SMB dataset) |
| **Rows** | 68,353 email records |
| **Columns** | 12 features |
| **Target Variable** | `Email_Status` — 3-class engagement outcome |
| **Duplicate Records** | **Zero** |
| **Class Distribution** | Severely imbalanced — 80:16:4 ratio |

### Feature Overview

| Column | Type | Description |
|---|---|---|
| `Email_ID` | Identifier | Unique email record ID — dropped before modelling |
| `Email_Type` | Categorical (int) | Type 1 (marketing / promotional) or Type 2 (important notice / transactional) |
| `Subject_Hotness_Score` | Numeric (0–5) | Score measuring subject line appeal and urgency |
| `Email_Source_Type` | Categorical (int) | Origin type of the email (e.g., sales vs. administrative) |
| `Customer_Location` | Categorical (str) | Geographic recipient segment (A–G) |
| `Email_Campaign_Type` | Categorical (int) | Campaign classification (Types 1, 2, 3) |
| `Total_Past_Communications` | Numeric | Historical count of prior emails sent to this recipient |
| `Time_Email_sent_Category` | Categorical (int) | Send time bucket: Morning (1), Noon (2), Night (3) |
| `Word_Count` | Numeric | Total word count in the email body |
| `Total_Links` | Numeric | Number of hyperlinks embedded in the email |
| `Total_Images` | Numeric | Number of images embedded in the email |
| `Email_Status` | Categorical (int) ⭐ | **Target** — 0: Ignored, 1: Opened, 2: Acknowledged |

### Target Variable — Class Distribution

| Class | Label | Count | Percentage |
|---|---|---|---|
| **0** | **Ignored** | **54,941** | **80.38%** |
| 1 | Opened | 11,039 | 16.15% |
| 2 | Acknowledged | 2,373 | 3.47% |

> **Critical imbalance:** An 80:16:4 ratio means a naive classifier predicting "Ignored" always achieves 80.38% accuracy while providing **zero business value**. SMOTE was applied to correct this bias before model training.

### Campaign-Level KPIs (Derived from Raw Data)

| KPI | Value | Business Significance |
|---|---|---|
| **Ignore Rate** | **80.38%** | 4 in 5 emails never opened — core problem to solve |
| **Open Rate** | 19.62% | Combined Opened + Acknowledged |
| **Engagement Rate** | **3.47%** | Only 1 in 29 emails receives a meaningful response |

### Missing Value Summary

| Column | Missing Count | % Missing | Imputation Strategy |
|---|---|---|---|
| `Customer_Location` | 11,595 | **16.9%** | Mode (most frequent location category) |
| `Total_Past_Communications` | 6,825 | **10.0%** | Mean (near-normal distribution) |
| `Total_Links` | 2,201 | 3.2% | Median (right-skewed) |
| `Total_Images` | 1,677 | 2.5% | Median (right-skewed) |

---

## 🛠 Tech Stack

```
Language              : Python 3.10+
Data Handling         : Pandas, NumPy
Visualisation         : Matplotlib, Seaborn
Statistical Testing   : SciPy (Chi-squared, Pearson, Spearman, ANOVA, Kruskal-Wallis)
Imbalance Handling    : imbalanced-learn (SMOTE)
Multicollinearity     : Statsmodels (VIF — Variance Inflation Factor)
Feature Selection     : Scikit-learn (VarianceThreshold), Pearson Correlation Matrix
ML Framework          : Scikit-learn, XGBoost
  ├── Models          : LogisticRegression, RandomForestClassifier, XGBClassifier
  ├── Tuning          : GridSearchCV + RepeatedStratifiedKFold (10×3)
  ├── Preprocessing   : StandardScaler, pd.get_dummies, IQR Capping, sqrt transform
  └── Metrics         : Accuracy, Precision, Recall, F1, ROC-AUC (OvR)
Model Explainability  : SHAP (TreeExplainer + interaction values)
Environment           : Google Colab / Jupyter Notebook
```

---

## 🔄 Project Workflow

```
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  1. Data Loading      │───▶│  2. EDA & KPI         │───▶│  3. Hypothesis       │
│     & Inspection      │    │     Analysis (9 charts)│    │     Testing (3 tests)│
└──────────────────────┘    └──────────────────────┘    └──────────┬───────────┘
                                                                    │
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────▼───────────┐
│  8. SHAP Model       │◀───│  7. Model Training    │◀───│  4. Feature          │
│     Explainability   │    │     + GridSearchCV    │    │     Engineering &    │
└──────────────────────┘    │     (3 Algorithms)   │    │     Preprocessing    │
                            └──────────────────────┘    └──────────┬───────────┘
                                       ▲                            │
                            ┌──────────┴───────────┐    ┌──────────▼───────────┐
                            │  6. 80/20 Train-Test  │◀───│  5. SMOTE Resampling │
                            │     Split            │    │     (Imbalance Fix)  │
                            └──────────────────────┘    └──────────────────────┘
```

### Phase Breakdown

**Phase 1 — Data Loading:** 68,353 rows × 12 columns. Zero duplicates. Business KPIs (ignore rate 80.38%, open rate 19.62%, engagement rate 3.47%) computed from raw target. Numeric and categorical features identified for targeted treatment.

**Phase 2 — EDA (9 Visualisations):** Distplots for all numeric features; count plots and pies for categoricals; campaign-type engagement analysis; location-based engagement rate comparison; KDE of `Total_Past_Communications` by email status; correlation heatmap; pair plot. Custom engagement/open/ignore rate functions computed per categorical group.

**Phase 3 — Hypothesis Testing:** Chi-squared (Email_Type → Email_Status), Pearson + Spearman (Subject_Hotness_Score → Total_Past_Communications), ANOVA + Kruskal-Wallis (Customer_Location → Total_Links and Total_Images).

**Phase 4 — Feature Engineering & Preprocessing:** Distribution-aware imputation; IQR outlier capping; three content ratio features engineered; VarianceThreshold selection; Pearson correlation filtering; VIF multicollinearity removal; one-hot encoding; square root transformation; StandardScaler normalisation.

**Phase 5 — SMOTE:** Applied exclusively to training set — balancing 80:16:4 to 1:1:1 before model fitting. Test set preserved in original distribution for unbiased evaluation.

**Phase 6 — Train-Test Split:** 80/20 split, `random_state=0`. SimpleImputer applied within training set for any residual NaN in scaled features before SMOTE.

**Phase 7 — Model Training & Tuning:** Logistic Regression (RepeatedStratifiedKFold + GridSearchCV over C), Random Forest (GridSearchCV over depth/estimators/leaf), XGBoost (same grid). All evaluated on the identical holdout set.

**Phase 8 — Model Explainability:** SHAP TreeExplainer on XGBoost. SHAP values and interaction values computed for X_test. Feature importance ranked for each class.

---

## 📈 Exploratory Data Analysis

### Email Status — The Core Business Reality

The target distribution tells the full business story upfront: **4 in 5 emails sent are completely ignored**. Only 3.47% receive the acknowledged response that marketers actually want. This is not an anomaly — it is the baseline performance of undifferentiated email marketing, and it is exactly what this model is designed to disrupt.

### Email Type Analysis

| Email Type | Volume Share | Ignore Rate | Business Implication |
|---|---|---|---|
| **Type 1** (Marketing / Promotional) | **71%** | **High** | High volume, poor engagement |
| **Type 2** (Important Notice / Transactional) | 29% | Significantly lower | Smaller volume, meaningfully better open rates |

Type 2 (transactional) emails outperform promotional Type 1 emails despite lower volume — confirming the "spam fatigue" effect. Recipients have conditioned themselves to deprioritise marketing content while remaining responsive to functional communications.

### Campaign Type Analysis

| Campaign Type | Volume | Key Metric | Business Action |
|---|---|---|---|
| **Type 1** | Smallest | Highest acknowledged rate | ✅ Scale up — most efficient per email sent |
| **Type 2** | **Largest** | **Only 1% acknowledged** | ❌ Discontinue or fundamentally reform |
| Type 3 | Medium | Moderate both | ⚠️ Optimise and monitor |

The dominant campaign strategy (Type 2 — highest volume) is the least effective. The highest-performing strategy (Type 1) receives the least investment. This is the most actionable business finding in the dataset.

### Send Timing Analysis

| Send Window | Volume Share | Engagement Pattern |
|---|---|---|
| **Noon** | **60%** | **Worst performance — highest ignore rate** |
| Morning | 20% | Higher acknowledged rate |
| Night | 20% | Higher acknowledged rate |

The majority of emails are sent at the worst time of day. Morning and evening — when people plan their day or wind down — show meaningfully higher response rates. This is a zero-cost operational fix.

### Customer Location Analysis

- **Location G:** Highest email volume but one of the lowest acknowledgement rates — over-invested, under-converting
- **Location C:** Lowest volume, **highest engagement rate** — highest past communications count, highest acknowledged ratio, highest ROI per email sent — a high-value segment being systematically under-emailed
- **Location B:** Low volume yet anomalously high acknowledged rate — another under-targeted high-value segment

### Past Communications & Engagement (KDE)

The stacked KDE plot clearly shows: **ignored emails cluster at lower past-communication counts**; **acknowledged emails cluster at higher counts**. Relationship warmth is a genuine predictor. Cold-outreach emails are structurally more likely to be ignored than emails sent to contacts with an established history.

### Correlation Heatmap — Critical Multicollinearity Detection

`Total_Links` and `Total_Images` exhibit **r = 0.78** — strong positive multicollinearity. Retaining both independently in any model would inflate VIF scores, distort coefficient estimates, and obscure true feature importance. These were resolved into engineered ratio features and the originals dropped.

### Hypothesis Testing Summary

| Hypothesis | Test | Outcome | Business Takeaway |
|---|---|---|---|
| Email_Type has no impact on Email_Status | Chi-squared | **Reject H₀** (p < 0.05) | Email type significantly determines engagement — must be optimised |
| Subject_Hotness_Score unrelated to Total_Past_Communications | Pearson + Spearman | **Reject H₀** (p < 0.05) | Subject quality correlates with communication history — engaged contacts respond to better subject lines |
| Customer_Location doesn't affect Total_Links / Total_Images | ANOVA + Kruskal-Wallis | **Reject H₀** (p < 0.05) | Content composition should be tailored by recipient geography |

---

## ⚙️ Feature Engineering & Preprocessing

### 1. Distribution-Aware Missing Value Imputation

```
Customer_Location         → Mode   (categorical, 16.9% missing — mode preserves most-common segment)
Total_Past_Communications → Mean   (near-normal distribution — mean is unbiased for symmetric data)
Total_Links               → Median (right-skewed — median robust to high-link-count outliers)
Total_Images              → Median (right-skewed — same reasoning)
```

### 2. Outlier Treatment — IQR Capping

All skew-symmetric numeric features capped at IQR outer fences. **Capping chosen over trimming** — all 68,353 records preserved. Extreme values of `Subject_Hotness_Score` and `Total_Links` carry genuine information about campaign design decisions and should not be discarded.

### 3. Engineered Content Ratio Features

Three new features resolve the `Total_Links` ↔ `Total_Images` multicollinearity (r = 0.78) while creating denser, more informative predictors:

| Engineered Feature | Formula | Business Meaning |
|---|---|---|
| `Percentage_of_words_that_are_links_image` | `(Links + Images) / Word_Count × 100` | Email media density ratio — how heavy is the email in visual/navigational elements relative to text? |
| `Images_per_link` | `Total_Images + Total_Links` | Total rich media element count per email |
| `Images_plus_link` | `Total_Images / Total_Links` | Image-to-link ratio — visual vs. action content balance |

`Total_Links` and `Total_Images` dropped after engineering — multicollinearity eliminated at source.

### 4. Feature Selection — Three-Stage Pipeline

**Stage 1 — VarianceThreshold (sklearn):** Removes quasi-constant features (variance < 0.05) that carry no discriminative signal.

**Stage 2 — Pearson Correlation Filter:** Pairwise correlation > 0.6 flagged. `Email_Campaign_Type_2`, `Customer_Location_G`, and `Email_Source_Type` dropped to eliminate high-correlation redundancy.

**Stage 3 — VIF Analysis (Statsmodels):** Variance Inflation Factors computed for all remaining numeric features. Features with VIF ≥ 8 removed — ensuring no residual multicollinearity inflates linear model standard errors. VIF recomputed post-removal to confirm resolution.

**Validation — Random Forest Embedded Importance:** Feature importances from a 550-tree Random Forest validated that all retained features carry non-trivial predictive signal.

### 5. One-Hot Encoding

`Time_Email_sent_Category`, `Customer_Location`, and `Email_Campaign_Type` encoded with `pd.get_dummies(drop_first=True)`. No ordinal relationship exists between any of these categories — one-hot encoding prevents models from imposing a false numeric ordering.

### 6. Square Root Transformation

```python
for col in ['Subject_Hotness_Score', 'Total_Past_Communications', 'Word_Count']:
    df_removed[col] = np.sqrt(df_removed[col])
```

These three features are right-skewed. Square root transformation reduces skew toward normality — satisfying Logistic Regression's distributional assumptions and stabilising gradient descent for XGBoost.

### 7. StandardScaler Normalisation

Applied to continuous features post-split to prevent data leakage. Centres features at mean=0, std=1 — essential for Logistic Regression where regularisation penalises large coefficient magnitudes and must operate on a standardised scale.

### 8. SMOTE — Synthetic Minority Over-sampling

```
Original training distribution: Class 0: 80% | Class 1: 16% | Class 2: 4%
Post-SMOTE distribution:        Class 0: 33% | Class 1: 33% | Class 2: 33%
```

SMOTE interpolates between existing minority class samples to generate synthetic observations — superior to duplication (causes overfitting to existing minority examples) and undersampling (discards majority class information). Applied **only to the training set** — test set preserved in original imbalanced distribution for honest evaluation.

---

## 🤖 Model Development

### Why Multi-Class Classification?

`Email_Status` has three distinct engagement outcomes: Ignored (0), Opened (1), Acknowledged (2). The business requires predicting *which* outcome class each email will fall into — not a binary flag — so that marketing teams can prioritise, reformat, or exclude emails from send queues based on predicted engagement tier. Probability outputs per class further enable recipient engagement scoring.

### Model 1 — Logistic Regression (Baseline)

Multinomial Logistic Regression with `class_weight='balanced'` and L2 regularisation. Produces interpretable probability estimates.

**Tuning:** GridSearchCV with RepeatedStratifiedKFold (10 splits × 3 repeats) over C values {0.001 → 1000}; optimised on F1-score.

**Limitation:** Assumes linear decision boundaries — fundamentally inadequate for the complex non-linear interactions between email content, timing, campaign type, and recipient behaviour that drive engagement outcomes.

### Model 2 — Random Forest Classifier

Ensemble of decision trees with bootstrap aggregation and random feature subsets. Captures non-linear relationships and feature interactions.

**Critical issue — Overfitting:** Baseline Random Forest achieved **100% training accuracy** — a definitive memorisation signal. Post-tuning constraints (max_depth, min_samples_leaf) substantially reduced overfitting; test performance recovered but minority class metrics remained weak.

**GridSearchCV grid:**
```python
param_grid = {
    'n_estimators'      : [50, 80, 100],
    'max_depth'         : [4, 6, 8],
    'min_samples_split' : [50, 100, 150],
    'min_samples_leaf'  : [40, 50]
}
```

### Model 3 — XGBoost Classifier ✅ (Final Model Selected)

Gradient boosting with sequential tree building — each tree corrects the residual errors of its predecessors. Native L1/L2 regularisation controls overfitting intrinsically, making XGBoost structurally more stable than Random Forest for this problem.

**Why XGBoost wins:**

| Capability | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Non-linear decision boundaries | ❌ | ✅ | ✅ |
| Sequential error correction | ❌ | ❌ | ✅ |
| Built-in L1/L2 regularisation | Partial (penalty param) | ❌ | ✅ |
| Overfitting resistance | High | Moderate (needs tuning) | High (native) |
| ROC-AUC on test data | Lower | Moderate | **Highest (93%)** |
| SHAP interpretability quality | Limited | Good | **Best** |

**SHAP Explainability:** TreeExplainer applied to the trained XGBoost model — computing SHAP values and interaction values per feature per prediction, enabling marketing stakeholders to understand **why** the model classifies each email.

---

## 📐 Model Evaluation

### Primary Metric: Why ROC-AUC (One-vs-Rest)?

With an 80:16:4 class imbalance:
- **Accuracy** is deceptive — predicting "Ignored" always yields 80.38% accuracy with zero value
- **F1-Score** per class captures precision-recall trade-offs but varies dramatically across the imbalanced classes
- **ROC-AUC (OvR)** measures the model's ability to discriminate each class from all others, averaged across classes — **invariant to class imbalance** and the most reliable single measure of classification capability for this problem type

### Model Performance Comparison

| Model | Configuration | Test ROC-AUC | Test Accuracy | Key Finding |
|---|---|---|---|---|
| Logistic Regression | Baseline | Moderate | ~50% | Linear boundary insufficient for interaction effects |
| Logistic Regression | GridSearchCV tuned | Moderate | Modest improvement | Regularisation helps but doesn't solve non-linearity |
| Random Forest | Baseline | High on train (~1.0) | 75% test | **Severely overfit — memorised training data** |
| Random Forest | GridSearchCV tuned | ~74% | ~65% | Corrected but underperforms on minority classes |
| **XGBoost** | **Baseline** | **~93%** | **81%** | **Best generalisation — selected as final** ✅ |
| XGBoost | GridSearchCV tuned | ~80% | ~70% | Conservative; more stable for production |

> **XGBoost baseline selected as the final model** — achieving 93% ROC-AUC with the best train-test performance balance across all three classes.

### XGBoost Final Model — Class-Level Test Performance

| Class | Label | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 0 | Ignored | **93%** | **85%** | **89%** |
| 1 | Opened | 19% | 41% | 26% |
| 2 | Acknowledged | 13% | 12% | ~13% |
| **Overall** | | **81% accuracy** | | **ROC-AUC: 93%** |

**Training performance (pre-tuning):**
- Ignored: Precision 95% / Recall 85% / F1 90%
- Opened: Precision 62% / Recall 82% / F1 70%
- Acknowledged: Precision 87% / Recall 77% / F1 82%
- Accuracy: 81% | ROC-AUC: 93%

### Business Interpretation of Model Performance

**Strong Ignored-class prediction (F1 = 89%)** is the most operationally valuable output: the model can reliably identify which emails will receive no engagement — enabling the marketing team to **exclude, reformat, or hold** these emails before sending. Reducing sends to predicted-ignored recipients by even 30% would directly cut wasted email budget.

**Weak Acknowledged-class precision (13%)** reflects the inherent rarity of this class — 3.47% of data. However, the business interpretation is asymmetric: even at 13% precision, identifying *any* high-intent acknowledged-probability recipients before sending is more valuable than zero prediction. These contacts should receive the highest-quality, most personalised email content.

**93% ROC-AUC** means that for any pair of email outcomes the model is asked to rank, it gets the ranking right 93% of the time — confirming genuine, robust discriminative capability across all three engagement classes far beyond random (50%) or naive (single-class) baselines.

---

## 💡 Key Insights

> *Translating ML outputs into board-level marketing intelligence.*

**1. 80% of Email Marketing Budget Is Measurably Wasted — And Now Predictable**
An 80.38% ignore rate is not an immutable industry constant — it is the measurable outcome of sending undifferentiated content at the wrong time to the wrong people. The ML model identifies the specific feature combinations that shift emails from the "Ignored" bucket into the "Opened" and "Acknowledged" buckets, giving the marketing team a concrete, data-driven lever.

**2. Campaign Type 2 — The Most Invested Strategy — Is the Most Ineffective**
Campaign Type 2 (largest volume, only 1% acknowledged rate) represents the single largest source of wasted campaign spend. The business is systematically over-investing in its worst-performing strategy. Reallocating 30% of Campaign Type 2 budget to Campaign Type 1 (lowest volume, highest acknowledged rate) would generate significant ROI improvement without any increase in total spend.

**3. Email Type Drives Engagement More Than Content Attributes Alone**
The Chi-squared test (p < 0.05) confirmed Email Type significantly determines email status. Transactional / notice-type emails (Type 2) consistently outperform promotional (Type 1) — confirming that **recipients respond to necessity and relevance, not marketing messaging**. This has direct implications for content framing strategy.

**4. Sending at Noon Is a Systematic, Zero-Cost-to-Fix Mistake**
60% of emails are dispatched during the lowest-engagement time window. Shifting even half of noon-send volume to morning and evening windows — a pure scheduling change requiring no content revision or additional budget — is predicted to improve acknowledged rates meaningfully. This is the highest-confidence, lowest-effort improvement available.

**5. Communication History Is a Genuine Engagement Predictor — CRM Integration Is Strategic**
`Total_Past_Communications` confirms that relationship warmth predicts engagement. Cold-outreach emails face a structurally higher ignore rate than emails to established contacts. Building a **Customer Engagement Warmth Score** from historical communication data — and routing campaign types by warmth tier — would improve overall campaign effectiveness while reducing churn risk for established relationships.

**6. Location C Is the Hidden High-Value Segment Being Systematically Under-Served**
Location C demonstrates the highest engagement rate per email sent while receiving among the lowest volume. The inverse of this pattern — Location G receives the highest volume with lower engagement — represents a systematic misallocation of campaign distribution. Rebalancing volume toward Location C is a data-confirmed high-ROI opportunity.

**7. 93% ROC-AUC Makes Pre-Send Engagement Scoring Operationally Viable**
A model with 93% ROC-AUC can be productionised as a **pre-send scoring gate** — ranking every planned email-recipient pair by predicted engagement probability before the campaign launches. This transforms email marketing from volume-first to precision-first: the highest predicted-engagement contacts receive the best content; the lowest predicted-engagement contacts are held for re-engagement or suppressed.

---

## 💼 Business Recommendations

**1. Immediately Audit Campaign Type 2 — The Most Invested, Least Effective Strategy**
Campaign Type 2's 1% acknowledged rate with the highest volume is the clearest ROI problem in the dataset. Conduct an emergency content and strategy review: test whether reformatting Type 2 content (subject line, word count, send time) moves the acknowledged rate. Set a 60-day performance threshold — if Type 2 doesn't reach 5% acknowledged, reallocate its budget to Campaign Type 1.

**2. Deploy XGBoost as a Pre-Send Engagement Prediction Gate**
Integrate the trained model into the campaign workflow as a scoring step before any email is sent. Emails with predicted engagement probability below a defined threshold (e.g., <20% open probability) are held for content review or excluded from the current send. This single operational change can reduce the ignore rate without any increase in send volume or budget.

**3. Shift 40% of Noon-Send Volume to Morning (08:00–10:00) and Evening (19:00–21:00)**
Run a 4-week controlled A/B test: 50% of planned noon sends redirected to morning, 50% to evening, measured against the noon control group. This is the lowest-effort, zero-cost change available — and the model's timing signal strongly supports it.

**4. Build and Deploy a Customer Engagement Warmth Score (CRM Integration)**
Integrate `Total_Past_Communications` as a live warmth score in the CRM — updated after every send event. Define three tiers: Cold (0–10 past comms), Warm (11–30), Hot (31+). Route Campaign Type 1 exclusively to Warm and Hot contacts; route nurturing sequences to Cold contacts before commercial outreach. This single segmentation change directly addresses the relationship-warmth finding.

**5. Increase Email Volume and Personalisation for Location C**
Location C shows the highest engagement rate per email sent in the database. Increase email frequency for Location C contacts by 25–30%, test premium campaign offers exclusively in Location C before platform-wide rollout, and develop C-specific content variants. This is the highest confidence high-ROI geographic investment available.

**6. Adopt Transactional Framing for Marketing Emails**
Since Type 2 (important notice) emails outperform Type 1 (promotional) in acknowledgement rate, A/B test marketing emails re-framed as important updates: *"Your account renewal is approaching"*, *"New features are available for your plan"*, *"We've updated the service you use"* — against standard promotional copy. Measure acknowledged rate lift over 8 weeks across matched audience segments.

**7. Build a Predicted-Ignore Re-Engagement Programme**
Contacts the model predicts as "Ignore" are not necessarily lost — they may be fatigued or disengaged. Trigger an automated re-engagement sequence for these contacts: a single high-quality personalised email targeting their inferred preferences, followed by a 90-day suppression window to prevent further fatigue degradation. Model predictions drive the sequence logic; human content quality drives the conversion.

---

## 🚀 Future Improvements

### Deployment & MLOps

**Production REST API (FastAPI + Docker):**
```json
POST /api/v1/predict-engagement
{
  "Email_Type": 1,
  "Subject_Hotness_Score": 2.4,
  "Email_Campaign_Type": 1,
  "Total_Past_Communications": 35,
  "Time_Email_sent_Category": 1,
  "Word_Count": 620,
  "Total_Links": 8,
  "Total_Images": 3,
  "Customer_Location": "C"
}
→ {
    "predicted_status": "Opened",
    "probabilities": {"Ignored": 0.31, "Opened": 0.52, "Acknowledged": 0.17},
    "engagement_score": 0.69
  }
```
Deploy on AWS Lambda (serverless) or Google Cloud Run for sub-100ms inference at campaign scale.

**MLOps Retraining Pipeline (Apache Airflow):**
Monthly scheduled pipeline — ingesting new campaign data, rerunning the full feature engineering pipeline, evaluating challenger vs. champion on a held-out validation window, promoting new model only when ROC-AUC improves.

**Model Drift Detection (Evidently AI):**
Monitor prediction distribution drift (ignore rate shifts due to market/seasonal changes) and feature drift (new email content norms shift word count or link distributions) — triggering automated retraining before accuracy degrades in production.

### Advanced Modelling

**Calibrated Probability Outputs:**
Apply `CalibratedClassifierCV` (Platt scaling or isotonic regression) to XGBoost outputs — enabling the business to use predicted scores as **true engagement probability estimates** (e.g., "23% probability of acknowledged") rather than relative ranking scores only.

**LightGBM as Production Speed-Optimiser:**
LightGBM achieves comparable accuracy to XGBoost at 3–5× faster inference — valuable for scoring millions of email-recipient pairs per campaign cycle in near real-time.

**Sequential Engagement Modelling:**
Model customer engagement as a temporal sequence using the history of Email_Status outcomes per recipient. LSTM or Transformer-based sequence models could capture the relationship trajectory that a single-email feature vector misses — predicting engagement based on the full communication history pattern.

### GenAI & LLM Integration

**Subject Line Hotness Optimiser (LLM-Powered):**
Build an AI copywriting assistant that takes a draft subject line, scores it against the model's `Subject_Hotness_Score` feature importance, and uses a GenAI API (Claude or GPT-4) to generate 5 alternatives predicted to score higher — enabling copywriters to A/B test model-optimised subject line variants before campaign launch.

**Automated Campaign Performance Briefings:**
Post-campaign, use an LLM to auto-generate a **natural language performance summary** from model output and actual engagement metrics — delivering a plain-English "what worked, what didn't, and what to change" briefing to marketing managers without requiring ML dashboard literacy.

**Personalised Content Generation:**
Combine the engagement prediction model with a GenAI text generator to auto-draft recipient-segment-specific email bodies — adapting word count, link density, and image count to each segment's predicted content preference profile, informed by feature importance from the trained XGBoost model.

### Product & Dashboard

**Interactive Marketing Intelligence Dashboard (Streamlit):**
A self-service tool where marketing managers upload a planned campaign recipient list and content parameters, receive predicted engagement scores per recipient, view predicted Ignored / Opened / Acknowledged breakdown by segment, and download a ranked send list ordered by engagement probability.

---

## ▶️ How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap statsmodels scipy
```

### Steps

**1. Clone the repository:**
```bash
git clone https://github.com/vishal-Londhekar/Email-Campaign-Effectiveness-Prediction-End-to-End-Machine-Learning-Capstone-Project.git
cd Email-Campaign-Effectiveness-Prediction-End-to-End-Machine-Learning-Capstone-Project
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Place the dataset in the working directory:**
```
data_email_campaign.csv
```

**4. Launch Jupyter Notebook:**
```bash
jupyter notebook Email_Campaign_Effectiveness_Prediction_End_to_End_Machine_Learning_Capstone_Project.ipynb
```

**5. Or open directly in Google Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishal-Londhekar/Email-Campaign-Effectiveness-Prediction-End-to-End-Machine-Learning-Capstone-Project/blob/main/Email_Campaign_Effectiveness_Prediction_End_to_End_Machine_Learning_Capstone_Project.ipynb)

> Update the dataset file path in **Cell 17** to your local or Google Drive path before running.

**6. Run all 370 cells sequentially** — the notebook executes end-to-end without errors.

**7. Generate a prediction using the final model:**
```python
# After running the notebook, use the trained xg_models directly:
sample = X_test.iloc[[0]]
proba = xg_models.predict_proba(sample)[0]
predicted_class = xg_models.predict(sample)[0]

labels = {0: 'Ignored', 1: 'Opened', 2: 'Acknowledged'}
print(f"Predicted outcome: {labels[predicted_class]}")
print(f"Ignored: {proba[0]:.2f} | Opened: {proba[1]:.2f} | Acknowledged: {proba[2]:.2f}")
```

---

## 📁 Project Structure

```
Email-Campaign-Effectiveness-Prediction/
│
├── Email_Campaign_Effectiveness_Prediction_End_to_End_Machine_Learning_Capstone_Project.ipynb
│                                              # Main notebook (370 cells, full ML pipeline)
├── data_email_campaign.csv                    # Dataset (68,353 rows × 12 columns)
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation (this file)
```

### `requirements.txt`
```
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
shap>=0.41.0
statsmodels>=0.13.0
scipy>=1.9.0
```

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <b>Vishal Londhekar</b><br/>
      <i>Data Analyst | Data Scientist | ML Engineer</i><br/><br/>
      <a href="https://github.com/vishal-Londhekar">🔗 GitHub</a>
    </td>
  </tr>
</table>

> *"An email campaign without engagement prediction is like running ads without targeting — volume without precision. This project turns email marketing from a guessing game into a data-driven science."*

---

## ⭐ If this project added value to your learning or portfolio, please star the repository!

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?logo=python" />
  <img src="https://img.shields.io/badge/Model-XGBoost%20Classifier-orange" />
  <img src="https://img.shields.io/badge/ROC--AUC-93%25-success" />
  <img src="https://img.shields.io/badge/Imbalance-SMOTE%20Applied-red" />
  <img src="https://img.shields.io/badge/Explainability-SHAP%20Values-blueviolet" />
  <img src="https://img.shields.io/badge/Domain-Email%20Marketing%20%7C%20CRM-lightgrey" />
</p>
