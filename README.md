# 🚚 ML-Based Late Delivery Risk Prediction
### Global Supply Chain Operations | APL Logistics (KWE Group)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ML](https://img.shields.io/badge/Model-Random%20Forest-green)
![Accuracy](https://img.shields.io/badge/Accuracy-77%25-brightgreen)

---

## 📌 Project Overview

This project develops a **machine learning–based predictive system** to identify late delivery risk in global supply chain operations **before shipments are dispatched** — enabling APL Logistics to shift from reactive delay management to proactive risk intelligence.

> **Key Result:** Random Forest model achieves **77% accuracy** and **69% recall** on 180,519 real order records, flagging 11,027 high-risk shipments for proactive intervention.

## 🌐 Live Demo

> **Dashboard Link:** [Click here to view live dashboard](https://YOUR-APP-LINK.streamlit.app)
> *(Link will be updated after deployment)*

---

## 🎯 Problem Statement

APL Logistics faces:
- **54.8%** of all orders experience late delivery
- No early warning system — delays discovered only after they occur
- High remediation costs from last-minute fixes
- SLA breaches causing customer churn and financial penalties

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| Total Records | 180,519 orders |
| Total Features | 40 columns |
| Target Variable | `Late_delivery_risk` (0 = On-Time, 1 = Late) |
| Late Orders | 98,977 (54.8%) |
| On-Time Orders | 81,542 (45.2%) |
| Missing Values | 11 cells (negligible) |

---

## 🔑 Key Findings

1. **First Class Paradox** — First Class shipping has **95.3% late delivery rate** vs Standard Class at 38.1%
2. **Scheduling is Root Cause** — Scheduled shipping days is the #1 predictor (6.36% importance)
3. **Geography is Minor Factor** — All 22 regions cluster within 48.8%–58.0% late rate
4. **Balanced Dataset** — 54.8% vs 45.2% split eliminates need for SMOTE

---

## 🤖 Models Trained

| Model | Accuracy | Recall | F1 Score |
|-------|----------|--------|----------|
| Logistic Regression (Baseline) | 71% | 54% | 67% |
| XGBoost | 74% | 64% | 73% |
| **Random Forest ✅ (Selected)** | **77%** | **69%** | **77%** |

---

## 🏗️ Project Structure

```
APL_Logistics_Project/
├── app.py                          # Streamlit dashboard
├── late_delivery_prediction.ipynb  # ML pipeline notebook
├── APL_Logistics.csv               # Dataset (180,519 orders)
├── rf_model.pkl                    # Trained Random Forest model
├── scaler.pkl                      # StandardScaler
└── README.md                       # This file
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/APL_Logistics_Project.git
cd APL_Logistics_Project
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 📱 Dashboard Features

| Page | Description |
|------|-------------|
| 🏠 **Overview** | KPI metrics, Shipping Mode chart, Region analysis |
| 📊 **Risk Analysis** | Feature importance, Model comparison, Key insights |
| 🔍 **Order Prediction** | Real-time risk scoring for new orders |
| 🚨 **Action Panel** | High-risk order queue with risk threshold filter |

**Sidebar Filters:**
- Shipping Mode selector
- Risk Threshold slider (0–100%)

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML Libraries:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Environment:** Jupyter Notebook (VS Code)

---

## 📦 Deliverables

- ✅ ML Pipeline (Jupyter Notebook)
- ✅ Trained Model (`rf_model.pkl`)
- ✅ Streamlit Dashboard (4 modules)
- ✅ Research Paper
- ✅ Executive Summary

---

## 👤 Author

**Atharva Sharma**
Data Analytics Intern
Unified Mentor | APL Logistics (KWE Group)
July 2026

---

## 📄 License

This project was developed as part of an internship with Unified Mentor for APL Logistics (KWE Group). Dataset is proprietary and provided by the organization.
