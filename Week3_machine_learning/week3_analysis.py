import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- 1. LOAD DATA ----------
df = pd.read_excel("Maths.csv")
print("Shape of dataset:", df.shape)
print(df.head())

# ---------- 2. BASIC EDA ----------
print("\nMissing values per column:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# ---------- 3. OUTLIER DETECTION (using boxplot on final grade G3) ----------
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["G3"])
plt.title("Outlier Check: Final Grade (G3)")
plt.savefig("outlier_G3.png")
plt.close()
print("\nSaved outlier_G3.png")

# ---------- 4. ENCODE CATEGORICAL (YES/NO AND TEXT) COLUMNS ----------
df_encoded = df.copy()
text_columns = df_encoded.select_dtypes(exclude=["number"]).columns.tolist()
print("\nText columns being encoded:", text_columns)

for col in text_columns:
    df_encoded[col] = df_encoded[col].astype("category").cat.codes

# ---------- 5. CORRELATION HEATMAP ----------
plt.figure(figsize=(16, 12))
sns.heatmap(df_encoded.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap - All Features")
plt.savefig("correlation_heatmap.png")
plt.close()
print("Saved correlation_heatmap.png")

# ---------- 6. FEATURES AND TARGET ----------
X = df_encoded.drop("G3", axis=1)
y = df_encoded["G3"]

# ---------- 7. TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 8. TRAIN LINEAR REGRESSION MODEL ----------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- 9. PREDICT AND EVALUATE ----------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n----- MODEL RESULTS -----")
print(f"R2 Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")

# ---------- 10. ACTUAL VS PREDICTED PLOT ----------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grade")
plt.savefig("actual_vs_predicted.png")
plt.close()
print("Saved actual_vs_predicted.png")

print("\nAll done! Check your folder for 3 saved PNG images.")