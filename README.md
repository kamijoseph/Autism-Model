# Autism Prediction using Machine Learning  

## 📌 Overview  
This project applies supervised machine learning to predict the likelihood of **Autism Spectrum Disorder (ASD)** using behavioral and demographic features.  

The dataset was sourced from [Kaggle](https://www.kaggle.com/). The pipeline covers **EDA, preprocessing, class imbalance handling, model training, hyperparameter tuning, evaluation, and model persistence**.  

The final deployed model is a **Random Forest Classifier**, tuned via randomized hyperparameter search and balanced using **SMOTE oversampling**.  

---

## 📂 Project Structure  
```bash

```
---

## 🔍 Workflow
### 1. Exploratory Data Analysis (EDA)

- Categorical features: visualized using countplots.

- Numerical features: explored using boxplots for spread and outlier detection.

- Target distribution: dataset was imbalanced, requiring oversampling.

### 2. Preprocessing

- Encoding: categorical features transformed using LabelEncoder (scikit-learn).

- Resampling: imbalance handled using SMOTE (imblearn.over_sampling.SMOTE).

### 3. Modeling

- Tree-based models were selected due to their interpretability and performance on tabular data:
    - Decision Tree Classifier

    - Random Forest Classifier

    - XGBoost Classifier


# Autism Prediction using Machine Learning

This project explores the use of machine learning techniques to predict autism spectrum disorder (ASD) from survey-based data available on [Kaggle](https://www.kaggle.com/). The workflow spans exploratory data analysis (EDA), preprocessing, model training, hyperparameter tuning, and evaluation. The goal was to build a reliable classifier capable of handling an imbalanced dataset.

---

## Requirements

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn
  - xgboost
  - pickle
---

## 📁 Project Structure
```bash
Autism-Model/
│── dataset/
│   └── autism.csv
│
│── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── encoders.pkl
│
│── plots/
│   ├── categorical_feature_countplots.png
│   ├── numerical_feature_boxplots.png
│   └── ...
│
│── notebook/
│   └── main.ipynb
│
│── .gitignore
│── requirements.txt
│── README.md
````

## 🚀 Features

- 🎚️ **Interactive Sliders and Manual Inputs** for eight diabetes related medical features

- 🔍 **Hybrid Input Mode:** Use either sidebar sliders or text inputs

- 📈 **Real-Time** Prediction using SVM model

- 📊 **Probability Scores** per class (Diabetic / Not Diabetic)

- ⚠️ **Medical Disclaimer Message** for responsible and proper application

## ⚙️ Installation
### 🔐 Prerequisites
- Python ≥ 3.10

- Conda (recommended)

- Git

### 📦 Setup Guide
1. Clone this repository
```bash
git clone https://github.com/kamijoseph/diabetes_predictor.git
cd diabetes_predictor
```
2. Create a new Conda environment
```bash
conda create -n diabetes-predictor python=3.12
```
3. Activate the environment
```bash
conda activate diabetes-predictor
```
4. Install dependencies
```bash
conda install --file requirements.txt
```
5. Run streamlit application
```bash
cd app
streamlit run main.py
```

---

## 🙋‍♂️ Questions or Feedback?

Feel free to open an issue or reach out if you have suggestions, questions, or ideas to improve this project.