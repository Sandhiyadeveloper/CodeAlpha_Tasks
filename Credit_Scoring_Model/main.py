import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Credit_Scoring_Dataset.csv")

print("Dataset Shape:", data.shape)
print(data.head())

data.drop("CUST_ID", axis=1, inplace=True)

print("\nMissing Values:")
print(data.isnull().sum())

X = data.drop("DEFAULT", axis=1)
y = data["DEFAULT"]

X["Debt_to_Savings"] = X["DEBT"] / (X["SAVINGS"] + 1)
X["Income_After_Debt"] = X["INCOME"] - X["DEBT"]

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

results = {}
for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)

        prediction = model.predict(X_test_scaled)
        probability = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)
        probability = model.predict_proba(
            X_test
        )[:, 1]


    results[name] = [
        accuracy_score(y_test, prediction),
        precision_score(y_test, prediction),
        recall_score(y_test, prediction),
        f1_score(y_test, prediction),
        roc_auc_score(y_test, probability)
    ]

    print("\n", "="*50)
    print(name)
    print("="*50)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        prediction
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        prediction
    ))

result_df = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)
print("\nModel Performance Comparison")
print(result_df)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": data.drop(
        ["DEFAULT", "CUST_ID"],
        axis=1
    ).columns.tolist()
    + [
        "Debt_to_Savings",
        "Income_After_Debt"
    ],
    "Importance": rf.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nTop 10 Important Features")
print(importance.head(10))
plt.figure(figsize=(10,6))
plt.barh(
    importance["Feature"].head(10),
    importance["Importance"].head(10)
)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title(
    "Top 10 Features Affecting Credit Risk"
)
plt.gca().invert_yaxis()
plt.show()