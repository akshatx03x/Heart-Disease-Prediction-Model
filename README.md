# Cardiovascular Clinical Risk Assessment Tool

A physician-referenced decision-support web application that evaluates the risk of coronary artery disease using a calibrated $k$-Nearest Neighbors ($k$-NN) classification pipeline trained on multi-dimensional clinical indicators.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## Overview

The tool processes 11 physiological inputs (mapped to 15 one-hot encoded and standardized features) to estimate patient risk of coronary disease. The interface is organized according to standard clinical examination workflows:

1. **Patient Baseline**: Age, biological sex.
2. **Hemodynamics & Blood Chemistry**: Resting blood pressure (AHA/ACC criteria), serum cholesterol (NCEP guidelines), and fasting blood glucose.
3. **Stress Response & Electrocardiography**: Chest pain presentation, resting ECG deviations, peak exertional heart rate, exercise-induced angina, and ST-segment depression (Oldpeak) and slope.

---

## Machine Learning Pipeline

- **Algorithm**: $k$-Nearest Neighbors ($k=9$, distance-weighted $w = 1/d$).
- **Preprocessing**: `StandardScaler` fitted across continuous and one-hot categorical dimensions.
- **Reference Dataset**: UCI Heart Disease Repository (918 clinical patient records).
- **Validation Metrics**:
  - Test Accuracy: **88.6%**
  - F1-Score: **0.90** (High-risk class) / **0.87** (Low-risk class)
  - 5-Fold Cross-Validation Mean: **83.6%**

---

## Getting Started

### 1. Prerequisites
Ensure Python 3.10 or higher is installed.

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/akshatx03x/Heart-Disease-Prediction-Model.git
cd Heart-Disease-Prediction-Model
pip install -r requirements.txt
```

### 3. Launch Application
Run the Streamlit app:

```bash
streamlit run app.py
```
*(If `streamlit` is not in your system PATH, use `python -m streamlit run app.py`).*

---

## Clinical Notice

This application is developed as an educational demonstration of applied machine learning in clinical informatics. It is not a certified diagnostic device and does not substitute for qualified clinical evaluation, laboratory tests, or diagnostic medical imaging.
