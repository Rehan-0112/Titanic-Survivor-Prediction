# 🚢 Titanic Survivor Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster, built using classic data preprocessing, feature engineering, and a Random Forest Classifier.

This project was made as a part of my journey learning ML — combining what I picked up from pandas/NumPy practice with concepts from supervised learning (classification, hyperparameter tuning, etc).

🔗 https://titanic-survivor-prediction-kcxejdy2vs9lrqy7s6f4kw.streamlit.app/

---

## 📌 Project Overview

The Titanic dataset is one of the most popular beginner-friendly ML problems (yes, the Kaggle classic 😄), and this project walks through a complete pipeline:

- Data cleaning & missing value imputation
- Feature engineering from raw columns (Name, Ticket, Cabin, etc.)
- Encoding categorical variables
- Scaling features
- Training a Random Forest Classifier
- Hyperparameter tuning using `RandomizedSearchCV` + `GridSearchCV`

The goal was not just to get a good accuracy score, but to actually understand **why** each preprocessing step matters.

---

## 🗂️ Dataset

The dataset used is the standard Titanic dataset (`train.csv` and `test.csv`), containing details like passenger class, age, sex, fare, cabin, siblings/spouses aboard, etc.

```
data/
├── train.csv
└── test.csv
```

The `data/` folder is included in this repo, so the notebook should run out of the box without needing to download anything separately.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – data manipulation
- **Matplotlib** – visualization
- **Scikit-learn** – modeling & hyperparameter tuning
- **Jupyter Notebook**
- **Streamlit** – for the web app where you can actually test predictions
- **Joblib** – to save/load the trained model, scaler, and encoders

---

## 🔍 What I Did (Step by Step)

1. **Loaded & combined** train + test data so preprocessing stays consistent across both
2. **Handled missing values**:
   - `Age` → filled using median age per passenger class
   - `Fare` → filled using median fare of similar passengers
   - `Embarked` → filled based on passengers with similar class/fare
   - `Cabin` → filled with placeholder `'M'`, reduced to just the deck letter
3. **Feature engineering**:
   - Extracted `Title` from passenger `Name` (Mr, Mrs, Miss, Master, etc.)
   - Binned `Age` into age groups
   - Calculated `tkt_count` (number of people sharing a ticket)
   - Created `Fare_per_Ticket` and binned it
   - Created `Num_Family` = SibSp + Parch + 1
4. **Encoding**:
   - Label Encoding for initial transformation
   - One-Hot Encoding for `Sex`, `Embarked`, and `Titles`
5. **Scaling** features using `MinMaxScaler`
6. **Model training**:
   - Baseline `RandomForestClassifier`
   - Tuned using `RandomizedSearchCV` followed by a focused `GridSearchCV`

---

## 📊 Results

After hyperparameter tuning with `RandomizedSearchCV` followed by a focused `GridSearchCV`, the final Random Forest model achieved:

| Metric | Score |
|---|---|
| Train Accuracy | 83.43% |
| Test Accuracy | 82.68% |

**Best hyperparameters found:**
```
criterion: gini
n_estimators: 210
max_depth: 44
max_features: sqrt
max_samples: 0.603
min_samples_split: 15
```

The close gap between train and test accuracy suggests the model generalizes well without significant overfitting.

---

## 🖥️ Web App

I also wrapped this up into a small Streamlit app so the model isn't just sitting in a notebook — you can actually punch in passenger details (class, age, fare, etc.) and get a survival prediction with probability.

Behind the scenes it loads the saved model, scaler, and label encoders (`model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_columns.pkl` in the `models/` folder) and runs the same preprocessing steps used during training, so the predictions match what the notebook would give.

To run it locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
It'll open in your browser at `localhost:8501`.

---

## 🚀 How to Run

1. Clone this repo
   ```bash
   git clone https://github.com/Rehan-0112/Titanic-Survivor-Prediction.git
   cd Titanic-Survivor-Prediction
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook in VS Code
   - Open the folder in VS Code
   - Open `Titanic Survivor Prediction.ipynb`
   - Select a Python kernel (top right) and run the cells
4. Or just run the web app directly (see Web App section below) if you don't want to touch the notebook

---

## 🧠 What I Learned

- How much missing-value handling can affect model performance
- Why feature engineering (like extracting titles, family size, fare-per-ticket) often matters more than the model itself
- The difference between Label Encoding and One-Hot Encoding, and when to use each
- How `RandomizedSearchCV` and `GridSearchCV` work together for efficient hyperparameter tuning
- That deploying a model is its own challenge — had to make sure the Streamlit app's preprocessing exactly matched the notebook's, otherwise predictions would be wrong even with the same model

---

## 🔮 Future Improvements

- Try other models (XGBoost, Logistic Regression) and compare performance
- Use cross-validation more extensively to validate tuning results
- Explore more advanced feature engineering (e.g., surname-based family grouping)
- Add input validation on the Streamlit app so it handles edge cases better

---

## 👤 Author

**Rehan**
B.Tech Data Science (S.Y.) Student

Feel free to connect or drop suggestions — always open to feedback as I'm still learning! 🙂
