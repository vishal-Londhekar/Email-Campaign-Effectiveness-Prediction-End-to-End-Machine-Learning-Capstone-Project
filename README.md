# ✉️ Email Campaign Effectiveness Prediction (End-to-End ML Project)

## 📌 Objective
To build a machine learning model that predicts whether an email marketing campaign will be successful, using historical engagement and customer data.

## 📂 Dataset
- Source: Simulated/Industry-standard email campaign dataset
- Features include:
  - Open rate, click-through rate (CTR), bounce rate
  - Time and day of email sent
  - Subject line length, campaign type, user engagement score
- Target: Campaign success (1) or failure (0)

## 🧰 Tools & Technologies
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- Seaborn, Matplotlib for visualization
- Jupyter Notebook

## 🚀 Project Workflow
1. 🧹 **Data Cleaning**
   - Handled missing values and outliers
   - Encoded categorical features

2. 📊 **EDA (Exploratory Data Analysis)**
   - Analyzed key patterns in CTR, open rates, and success rate
   - Visualized trends across days, timing, and bounce categories

3. ⚙️ **Modeling**
   - Trained multiple classification models
   - Evaluated using Accuracy, Precision, Recall, F1 Score, ROC-AUC
   - Chose the best model based on business trade-offs

4. ✅ **Model Evaluation**
   - Best model: Random Forest (F1 = 84%, ROC-AUC = 0.91)
   - Identified key features: Email send time, bounce rate, CTR

5. 📉 **Optimization**
   - Proposed segmentation and sending-time adjustments to increase campaign conversion rate by **~20%**

## 📸 Visuals
- ROC Curves
- Confusion Matrix
- Feature Importance Chart
- Campaign Timing Effectiveness Plot


## 💼 Business Value
This model helps marketing teams:
- Segment customers more effectively
- Time campaigns for maximum engagement
- Reduce bounce rates and improve ROI
