# ======================================================
# Disease Prediction from Medical Data
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ======================================================
# 1. Load Dataset
# ======================================================

data = pd.read_csv(
    "heart_disease.csv"
)

print("Dataset Shape:", data.shape)
print(data.head())


# ======================================================
# 2. Data Preprocessing
# ======================================================

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Separate features and target
X = data.drop(
    "Heart Disease Status",
    axis=1
)

y = data[
    "Heart Disease Status"
]


# Encode target (No=0, Yes=1)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


# Encode categorical features
categorical_columns = X.select_dtypes(
    include="object"
).columns


feature_encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col].astype(str)
    )

    feature_encoders[col] = encoder


# Handle missing values
imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X)


# ======================================================
# 3. Feature Scaling
# ======================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ======================================================
# 4. Train Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================================================
# 5. Model Training
# ======================================================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Support Vector Machine":
        SVC(
            probability=True
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


results = {}


for name, model in models.items():

    print("\n" + "="*50)
    print(name)
    print("="*50)

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )


    results[name] = [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]


    print(
        classification_report(
            y_test,
            y_pred
        )
    )


# ======================================================
# 6. Compare Model Performance
# ======================================================

result_table = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
).T


print("\nModel Comparison")
print(result_table)


# ======================================================
# 7. Train Final Random Forest Model
# ======================================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)


# Feature Importance
feature_names = data.drop(
    "Heart Disease Status",
    axis=1
).columns


importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 10 Important Features")
print(
    importance.head(10)
)


# ======================================================
# 8. Plot Feature Importance
# ======================================================

plt.figure(
    figsize=(10,6)
)

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top Features Affecting Heart Disease"
)

plt.show()


# ======================================================
# 9. Save Final Model
# ======================================================

import pickle


with open(
    "heart_disease_model.pkl",
    "wb"
) as file:

    pickle.dump(
        rf,
        file
    )


print("\nModel Saved Successfully!")