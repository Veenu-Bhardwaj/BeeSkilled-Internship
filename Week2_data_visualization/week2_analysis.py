import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- STEP 1: LOAD THE DATASET ----
df = pd.read_csv("country_wise_latest.csv")

print("Columns in dataset:")
print(df.columns)

# ---- STEP 2: TOP 5 COUNTRIES BY CONFIRMED CASES ----
top5 = df.sort_values("Confirmed", ascending=False).head(5)
print("\nTop 5 countries by Confirmed cases:")
print(top5[["Country/Region", "Confirmed", "Deaths", "Recovered"]])

# ---- STEP 3: BAR CHART - COMPARE TOP 5 COUNTRIES ----
plt.figure(figsize=(10, 6))
sns.barplot(data=top5, x="Country/Region", y="Confirmed")
plt.title("Top 5 Countries by Confirmed COVID-19 Cases")
plt.xlabel("Country")
plt.ylabel("Confirmed Cases")
plt.tight_layout()
plt.savefig("bar_chart_top5.png")
plt.show()

# ---- STEP 4: "TREND" LINE - CONFIRMED vs LAST WEEK (proxy since no date column) ----
plt.figure(figsize=(10, 6))
x = range(len(top5))
plt.plot(x, top5["Confirmed last week"], marker="o", label="Last Week")
plt.plot(x, top5["Confirmed"], marker="o", label="This Week")
plt.xticks(x, top5["Country/Region"])
plt.title("Confirmed Cases: Last Week vs This Week (Top 5 Countries)")
plt.xlabel("Country")
plt.ylabel("Confirmed Cases")
plt.legend()
plt.tight_layout()
plt.savefig("trend_lines_top5.png")
plt.show()

# ---- STEP 5: HEATMAP OF CORRELATION ----
numeric_cols = df[["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths"]]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - COVID-19 Data")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

# ---- STEP 6: SCATTER PLOT - CONFIRMED vs DEATHS ----
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Confirmed", y="Deaths")
plt.title("Confirmed Cases vs Deaths (All Countries)")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

print("\nAll charts saved as PNG files!")