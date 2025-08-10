# Titanic Survival Prediction

This project analyzes the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) and builds a machine learning model to predict passenger survival.

## 📌 Project Overview
The Titanic dataset is one of the most well-known beginner datasets in data science.  
It contains demographic and travel information for passengers aboard the Titanic, with the goal of predicting survival based on features such as:
- **Pclass** (ticket class)
- **Sex**
- **Age**
- **SibSp** (number of siblings/spouses aboard)
- **Parch** (number of parents/children aboard)
- **Fare**
- **Embarked** (port of embarkation)

## 📂 Files
- `titanic.ipynb` — Jupyter Notebook containing the full analysis, feature engineering, model training, and evaluation.
- `train.csv` — Training dataset (from Kaggle).
- `test.csv` — Test dataset (from Kaggle).
- `gender_submission.csv` — Example submission file (from Kaggle).

## 🔍 Main Steps
1. **Data Loading** — Import training and test datasets.
2. **Exploratory Data Analysis (EDA)** — Understand data types, missing values, and feature distributions.
3. **Data Cleaning** — Handle missing values, encode categorical features, and create new features.
4. **Feature Engineering** — Generate new variables from existing ones (e.g., `Fare_Per_Person`).
5. **Model Training** — Train classification models to predict survival.
6. **Prediction & Submission** — Generate predictions for the Kaggle competition.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/WEILLLL889/Titanic---Machine-Learning-from-Disaster.git
 
