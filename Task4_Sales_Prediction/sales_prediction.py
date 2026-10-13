"""CodeAlpha Data Science Internship - Task 4: Sales Prediction using Python"""
import os
import matplotlib
matplotlib.use("Agg")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from any location
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# 1. Load & clean
df = pd.read_csv("advertising.csv", index_col=0)
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.describe().round(2), "\n")
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicates:", int(df.duplicated().sum()), "\n")
df = df.drop_duplicates().dropna()

# 2. Exploration
sns.pairplot(df, kind="reg", plot_kws={"line_kws": {"color": "red"}})
plt.savefig("pairplot.png", dpi=110, bbox_inches="tight"); plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=110, bbox_inches="tight"); plt.close()
print("Correlation with Sales:\n", df.corr()["Sales"].drop("Sales").round(3), "\n")

# 3. Split
X, y = df[["TV", "Radio", "Newspaper"]], df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
rows = []
for name, m in models.items():
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring="r2").mean()
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    rows.append((name, cv, r2_score(y_test, p), mean_absolute_error(y_test, p),
                 np.sqrt(mean_squared_error(y_test, p))))
res = pd.DataFrame(rows, columns=["Model", "CV R2", "Test R2", "MAE", "RMSE"])
print(res.round(4).to_string(index=False), "\n")
res.to_csv("model_comparison.csv", index=False)

# 5. Best model
best_name = res.sort_values("Test R2", ascending=False).iloc[0]["Model"]
best = models[best_name]
pred = best.predict(X_test)
print("Best model:", best_name)

plt.figure(figsize=(5.5, 5))
plt.scatter(y_test, pred, alpha=0.7)
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted - {best_name}")
plt.savefig("actual_vs_predicted.png", dpi=110, bbox_inches="tight"); plt.close()

# 6. Business insight: effect of ad spend (linear model coefficients)
lr = LinearRegression().fit(X, y)
coef = pd.Series(lr.coef_, index=X.columns)
print("\nSales change per +1 unit ad spend (others fixed):\n", coef.round(4))
plt.figure(figsize=(5, 4))
coef.plot(kind="bar", color=["#4c72b0", "#55a868", "#c44e52"])
plt.ylabel("Sales impact per unit spend"); plt.title("Impact of Advertising Channel")
plt.xticks(rotation=0)
plt.savefig("channel_impact.png", dpi=110, bbox_inches="tight"); plt.close()

if hasattr(best, "feature_importances_"):
    print("\nFeature importances:\n",
          pd.Series(best.feature_importances_, index=X.columns).round(3))
