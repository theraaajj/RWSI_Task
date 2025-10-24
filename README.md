# RWSI - Conversion Prediction

## Project Overview

This project analyzes the "Retail Web Session Intelligence" (RWSI) dataset, which simulates user interactions on an e-commerce platform. The primary goal is to understand what user behaviors and contextual factors drive a session to result in a purchase.

The core business problem is to build a predictive model that can accurately identify high-intent shoppers, allowing the business to differentiate them from casual browsers and take targeted actions.

Business Goal: Predict MonetaryConversion (Yes/No).

## Project Workflow & Key Findings

This project follows a structured data science workflow. Here is a summary of the steps taken and the insights gathered.

### 1. Exploratory Data Analysis (EDA)

We began with a deep dive into the data to understand its structure and identify key challenges.

Key Finding 1: Severe Class Imbalance

The target variable MonetaryConversion is highly imbalanced.

Conversion ('Yes'): 1908 (15.47%)

No Conversion ('No'): 10422 (84.53%)

Inference: This imbalance is the most critical challenge. "Accuracy" will be a misleading metric. Our success must be measured by Precision and Recall, and our models must use a strategy (like class_weight='balanced') to handle this.

Key Finding 2: Skewed Data & Significant Outliers

Analysis of df.describe() showed that many numerical features (e.g., ItemBrowseTime, InfoSectionTime, and even count-based columns like ItemBrowseCount) are heavily right-skewed.

We observed std values significantly larger than mean values and max values (e.g., 63,000+ seconds for ItemBrowseTime) that are extreme outliers.

Inference: This proves that a simple StandardScaler (which relies on the mean and std) would be an ineffective and incorrect choice for these features.

### 2. Data Preprocessing & Feature Engineering

Based on our EDA, we designed a robust preprocessing pipeline to clean and transform the data, making it suitable for modeling.

**A. Missing Value Imputation**

We filled missing values using a robust strategy:

Numerical Features (e.g., AdClicks, ItemBrowseTime): Imputed with the Median, which is resistant to the outliers we found.

Categorical Features (e.g., MarketZone, UserCategory): Imputed with the Mode (most frequent value).

**B. Encoding Strategy**

We analyzed all categorical features and determined they are nominal (no inherent rank).

Technique: One-Hot Encoding (OHE) was chosen as the most accurate technique.

Why: It creates independent binary columns for each category (e.g., MarketZone_Europe), preventing the model from learning a false mathematical relationship (e.g., that Europe > Asia).

Features Encoded: VisitMonth, WebClientCode, MarketZone, UserCategory, DeviceCategory, TrafficSourceCode, and IsWeekendVisit.

**C. Scaling Strategy (Advanced)**

Based on our df.describe() analysis, we adopted a precise, two-part scaling strategy after the train-test split:

RobustScaler: Applied to all skewed features (both "time" and "count" columns) that were heavily affected by outliers.

StandardScaler: Applied to all non-skewed features (e.g., ratios like ExitRateFirstPage and scores like PageEngagementScore).


## 📈 Model Development and Iteration

This project involved a systematic process of building and tuning several classification models to find the best performer for this imbalanced dataset. The primary business goal was to maximize the identification of **Conversions (Class 1)**, making **Recall** and **F1-Score** the most important evaluation metrics.

---

### 1. The Imbalance Problem

The dataset is highly imbalanced, with "No Conversion" (Class 0) samples significantly outnumbering "Conversion" (Class 1) samples.

**Class 0 (No Conversion):** 2084 samples
**Class 1 (Conversion):** 382 samples
**Imbalance Ratio:** Approximately **5.45 to 1**

This imbalance causes most standard models to become "lazy," achieving high accuracy by simply predicting the majority class, but failing to identify the rare positive cases we care about.

---

### 2. Modeling Strategy

We tested five models, iteratively improving our approach to handle the class imbalance:

1.  **Model 1: Baseline Logistic Regression**
    **Result:** High accuracy (87.6%) but terrible performance on the minority class.
    **Conversion F1-Score: 0.46** / **Recall: 0.34**
    **Conclusion:** This model was not useful as it missed 66% of all conversions.

2.  **Model 2: Tuned Logistic Regression**
    **Tuning:** Used `GridSearchCV` with the `class_weight='balanced'` parameter.
    **Result:** A massive improvement. The model sacrificed a little accuracy to find far more conversions.
    **Conversion F1-Score: 0.60** / **Recall: 0.68**
    **Conclusion:** Proved that handling class imbalance was the key to success.

3.  **Model 3: Baseline Random Forest**
    **Result:** Similar to the baseline logistic regression. It had very high precision (0.73) but poor recall.
    **Conversion F1-Score: 0.58** / **Recall: 0.48**
    **Conclusion:** A more complex model doesn't automatically solve the imbalance problem.

4.  **Model 4: Tuned Random Forest**
    **Tuning:** Used `GridSearchCV`, again including `class_weight='balanced'`.
    **Result:** A very strong, well-balanced model. It became our new top performer.
    **Conversion F1-Score: 0.63** / **Recall: 0.63**
    **Conclusion:** A significant improvement over the tuned Logistic Regression in F1-score.

5.  **Model 5: Tuned XGBoost**
    **Tuning:** Used `GridSearchCV` with the `scale_pos_weight` parameter, setting it to the imbalance ratio (~5.45).
    **Result:** This model produced the best results of the entire project.
    **Conversion F1-Score: 0.65** / **Recall: 0.74**
    **Conclusion:** The champion model. It finds the most conversions (74%) while maintaining the best F1-score.

---

### 3. Final Model Comparison (Class 1: Conversion)

This table summarizes the performance of all models on the task of finding positive conversions.

| Model | Accuracy | Conversion Precision | Conversion Recall | Conversion F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| 1. Initial LogReg | 87.6% | 0.71 | 0.34 | 0.46 |
| 2. Tuned LogReg | 86.1% | 0.54 | 0.68 | 0.60 |
| 3. Initial RF | 89.3% | **0.73** | 0.48 | 0.58 |
| 4. Tuned RF | 88.7% | 0.64 | 0.63 | 0.63 |
| **5. Tuned XGBoost** | 87.7% | 0.58 | **0.74** | **0.65** |

---

## 🏆 Conclusion

The **Tuned XGBoost (Model 5)** was selected as the final and best-performing model.

While its overall accuracy is not the highest, it decisively outperforms all other models on the metrics that matter for this business problem. It successfully identifies **74% of all conversions (True Positives)**, missing the fewest opportunities (only 99 False Negatives), and provides the best overall balance between precision and recall, as shown by its **F1-Score of 0.65**.
