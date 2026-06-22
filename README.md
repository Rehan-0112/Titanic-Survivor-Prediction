# 🚢 Titanic Survivor Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster, built using classic data preprocessing, feature engineering, and a Random Forest Classifier.

This project was made as a part of my journey learning ML — combining what I picked up from pandas/NumPy practice with concepts from supervised learning (classification, hyperparameter tuning, etc).

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

The final Random Forest model, after hyperparameter tuning, gave a solid improvement over the baseline. (Add your final accuracy score here once you note it down!)

---

## 🚀 How to Run

1. Clone this repo
   ```bash
   git clone https://github.com/<your-username>/Titanic_Survivor_Prediction.git
   cd Titanic_Survivor_Prediction
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```
3. Run the notebook
   ```bash
   jupyter notebook Titanic_Survivor_Prediction.ipynb
   ```

---

## 🧠 What I Learned

- How much missing-value handling can affect model performance
- Why feature engineering (like extracting titles, family size, fare-per-ticket) often matters more than the model itself
- The difference between Label Encoding and One-Hot Encoding, and when to use each
- How `RandomizedSearchCV` and `GridSearchCV` work together for efficient hyperparameter tuning

---

## 🔮 Future Improvements

- Try other models (XGBoost, Logistic Regression) and compare performance
- Use cross-validation more extensively to validate tuning results
- Explore more advanced feature engineering (e.g., surname-based family grouping)
- Deploy the model with a simple Streamlit/Flask interface

---

## 👤 Author

**Rehan**
B.Tech Data Science (S.Y.) Student

Feel free to connect or drop suggestions — always open to feedback as I'm still learning! 🙂
