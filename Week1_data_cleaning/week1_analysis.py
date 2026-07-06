import pandas as pd

# ---- STEP 1: LOAD THE DATASET ----
df = pd.read_csv("StudentsPerformance.csv")

# ---- STEP 2: EXPLORE THE DATA FIRST ----
print("First 5 rows of data:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nBasic info about the dataset:")
print(df.info())

# ---- STEP 3: CLEAN MISSING DATA ----
print("\nChecking for missing values in each column:")
print(df.isnull().sum())

df["math score"] = df["math score"].fillna(df["math score"].mean())
df["reading score"] = df["reading score"].fillna(df["reading score"].mean())
df["writing score"] = df["writing score"].fillna(df["writing score"].mean())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---- STEP 4: CALCULATE AVERAGE, MAX, MIN SCORES ----
print("\n--- MATH SCORE STATS ---")
print("Average:", df["math score"].mean())
print("Max:", df["math score"].max())
print("Min:", df["math score"].min())

print("\n--- READING SCORE STATS ---")
print("Average:", df["reading score"].mean())
print("Max:", df["reading score"].max())
print("Min:", df["reading score"].min())

print("\n--- WRITING SCORE STATS ---")
print("Average:", df["writing score"].mean())
print("Max:", df["writing score"].max())
print("Min:", df["writing score"].min())

# ---- STEP 5: DISPLAY RESULTS IN A CLEAN TABLE ----
summary = pd.DataFrame({
    "Subject": ["Math", "Reading", "Writing"],
    "Average": [df["math score"].mean(), df["reading score"].mean(), df["writing score"].mean()],
    "Max": [df["math score"].max(), df["reading score"].max(), df["writing score"].max()],
    "Min": [df["math score"].min(), df["reading score"].min(), df["writing score"].min()]
})

print("\n=== FINAL SUMMARY TABLE ===")
print(summary)

# ---- STEP 6: SAVE RESULTS TO A NEW CSV FILE ----
summary.to_csv("week1_summary_results.csv", index=False)
print("\nSummary saved to week1_summary_results.csv")