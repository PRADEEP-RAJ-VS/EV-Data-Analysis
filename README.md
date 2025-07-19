---

# 🔍 Automotive Consumer Data Analysis for Optimizing Marketing Strategies

This project analyzes electric vehicle (EV) sales data to uncover patterns in **purchase behavior**, **demand forecasting**, **targeted marketing**, and **inventory optimization** using **machine learning**, **clustering**, and **deep learning models**.

---

## 📌 Key Objectives

* 🧠 **Predict EV purchase likelihood** using advanced ML models
* 🎯 **Segment customers** for personalized marketing
* 🧾 **Forecast demand** and recommend optimal inventory per region
* 💰 **Estimate expected revenue** based on predicted behavior
* 📊 Visualize KPIs using Seaborn and Plotly

---

## 🧰 Tools & Technologies

* **Language**: Python
* **ML Libraries**: Scikit-learn, XGBoost, LightGBM, CatBoost
* **Deep Learning**: TensorFlow (MLP model)
* **Clustering**: HDBSCAN, KMeans, PCA
* **Visualization**: Plotly, Seaborn, Matplotlib
* **Explainability**: SHAP

---

## 📁 Dataset Features

* `Revenue`, `Units_Sold`, `Discount_Percentage`, `Battery_Capacity_kWh`
* Categorical: `Region`, `Brand`, `Vehicle_Type`, `Fast_Charging_Option`
* Engineered:

  * `Revenue_per_Unit`
  * `Discount_Benefit`
  * `Battery_per_Revenue`
  * `High_Discount`, `Revenue_Trend`, etc.

---

## 🔬 Feature Engineering

```python
train_df['Revenue_per_Unit'] = train_df['Revenue'] / train_df['Units_Sold']
train_df['Discount_Benefit'] = train_df['Discount_Percentage'] * train_df['Revenue_per_Unit']
train_df['Battery_per_Revenue'] = train_df['Battery_Capacity_kWh'] / (train_df['Revenue'] + 1)
train_df['High_Discount'] = (train_df['Discount_Percentage'] > train_df['Discount_Percentage'].mean()).astype(int)
```

---

## 📊 Exploratory Data Analysis (EDA)

* **Monthly EV Sales Trends**
* **Region vs Vehicle Type Sales**
* **Fast Charging Impact**
* **Segment-wise Discount Impact**
* **Correlation Matrix & Distributions**

---

## 📈 Clustering & Segmentation

### ✅ HDBSCAN + PCA

* Unsupervised clustering of consumer profiles
* Visualized with Plotly
* Best `min_cluster_size` selected using silhouette score

### ✅ KMeans (3 Clusters)

* Segmentation based on:

  * `Units_Sold`
  * `Discount_Percentage`

---

## 🤖 Models Used

### 🎯 Classification: Purchased or Not?

1. **XGBoost Classifier**

   * Feature importance + hyperparameter tuning via `RandomizedSearchCV`

2. **MLP Neural Network**

   * Dense layers + Dropout with early stopping
   * Visualized training vs validation accuracy

3. **Stacked Ensemble**

   * Base learners: CatBoost, LightGBM, HistGradientBoosting
   * Meta learner: Logistic Regression
   * Evaluated with:

     * ROC Curves
     * Calibration Curve
     * Confusion Matrix

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-Score
* Confusion Matrix
* ROC AUC
* SHAP Plots (Feature Importance)

---

## 📦 Business Insights

### 📍 Region-wise Performance

* Purchase probability and expected revenue calculated for each region

### 🛒 Inventory Recommendations

* Predicted demand per segment & region using:

  ```python
  expected_demand = purchase_prob × units_sold
  ```

### 💡 Discount Strategy

* Visual analysis of purchase rate by discount brackets

---

## 🧠 Explainability with SHAP

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## 📌 Key Visuals

* Monthly EV Sales Trend
* Clustered Consumer Segments
* Purchase Rate by Brand / Region
* Discount vs Purchase Rate
* Expected Revenue & Demand Maps
* ROC & Calibration Curves
* SHAP Feature Importance

---

## 📈 How to Run

1. Clone the repo
2. Place `train.csv` in the working directory
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:

   ```bash
   python untitled32.py
   ```

---

## 📜 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👨‍💻 Author

**Pradeep Raj V S**
B.Tech (Information Technology) – Final Year
Focused on real-world ML applications for intelligent marketing

---
