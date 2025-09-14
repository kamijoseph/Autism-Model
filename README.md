# Autism Prediction using Machine Learning  

## 📌 Overview  
This project applies supervised machine learning to predict the likelihood of **Autism Spectrum Disorder (ASD)** based on behavioral and demographic features.  

The dataset was sourced from [Kaggle](https://www.kaggle.com/). The pipeline covers:  
- **EDA** (exploratory data analysis)  
- **Preprocessing** (encoding + resampling)  
- **Model training & hyperparameter tuning**  
- **Evaluation & persistence**  

The final deployed model is a **Random Forest Classifier**, tuned via randomized hyperparameter search and balanced with **SMOTE oversampling**.  

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
│   └── ...
│
│── notebook/
│   └── main.ipynb
│
│── .gitignore
│── requirements.txt
│── README.md
```

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

## ⚙️ Installation
### 🔐 Prerequisites
- Python ≥ 3.10

- Conda (recommended)

- Git

### 📦 Setup Guide
1. Clone this repository
```bash
git clone https://github.com/kamijoseph/Autism-Model.git
cd Autism-Model
```
2. Create a new Conda environment
```bash
conda create -n Autism_Model python=3.12
```
3. Activate the environment
```bash
conda activate Autism-Model
```
4. Install dependencies
```bash
conda install --file requirements.txt
```
5. Run the Notebook
```bash
cd notebooks
jupyter-notebook main.ipynb
```

---
## Run Inference With Saved Model
### import pickle
```bash
import pickle
```

### Load model
```bash
with open("../models/RandomForest.sav", "rb") as f:
    model = pickle.load(f)
```

### Load encoders
```bash
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
```

### Preprocess new data, then predict
```bash
y_pred = model.predict(new_data)
```

## 🙋‍♂️ Questions or Feedback?
Feel free to open an issue or reach out if you have suggestions, questions, or ideas to improve this project.