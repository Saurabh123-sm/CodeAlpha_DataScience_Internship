"""CodeAlpha Data Science Internship - Task 1: Iris Flower Classification"""
import os
import matplotlib
matplotlib.use("Agg")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from any location
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Load data
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.describe(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")

# 2. Exploratory visualisation
sns.pairplot(df.drop(columns="target"), hue="species")
plt.savefig("pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

# 3. Train / test split
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Compare models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
results = []
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append((name, cv, acc))
res = pd.DataFrame(results, columns=["Model", "CV Accuracy", "Test Accuracy"])
print(res.round(4).to_string(index=False), "\n")

# 5. Evaluate best model
best_name = res.sort_values(["Test Accuracy", "CV Accuracy"], ascending=False).iloc[0]["Model"]
best = models[best_name]
pred = best.predict(X_test)
print(f"Best model: {best_name}")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=iris.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=iris.target_names)
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

res.to_csv("model_comparison.csv", index=False)
