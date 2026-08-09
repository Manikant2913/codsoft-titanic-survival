import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load the dataset
df = pd.read_csv(
    "/Users/manikanthaprakashbadin/CODSOFT_INTERNSHIP/"
    "Task1_Titanic_Survival/Titanic-Dataset.csv"
)


# 2. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# 3. Convert categorical variables into numbers
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)


# 4. Select features
features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S"
]

X = df[features]
y = df["Survived"]


# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Titanic Survival Prediction")
print("=" * 30)
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# 10. Display Confusion Matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)

plt.title("Titanic Survival - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()