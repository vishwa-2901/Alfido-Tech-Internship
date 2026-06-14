# ============================================================
# Task 3: Data Analysis with Pandas
# ============================================================

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD & INSPECT
# ─────────────────────────────────────────────

# Using the classic Titanic dataset (public, no download needed)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 2. CLEAN MISSING / INCORRECT DATA
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Fill missing Age with median age
median_age = df["Age"].median()
df["Age"].fillna(median_age, inplace=True)
print(f"Filled {df['Age'].isnull().sum()} missing Age values with median: {median_age}")

# Fill missing Embarked with mode
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"].fillna(mode_embarked, inplace=True)
print(f"Filled missing Embarked values with mode: '{mode_embarked}'")

# Drop Cabin column (too many missing values to be useful)
df.drop(columns=["Cabin"], inplace=True)
print("Dropped 'Cabin' column (>77% missing)")

# Remove duplicate rows if any
dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Removed {dupes} duplicate rows")

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 3. FILTERING, GROUPING & AGGREGATION
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("ANALYSIS")
print("=" * 50)

# --- Overall survival rate ---
survival_rate = df["Survived"].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.1f}%")

# --- Survival rate by passenger class ---
print("\nSurvival rate by Passenger Class:")
by_class = (
    df.groupby("Pclass")["Survived"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "Survival Rate", "count": "Passengers"})
)
by_class["Survival Rate"] = (by_class["Survival Rate"] * 100).round(1)
print(by_class)

# --- Survival rate by gender ---
print("\nSurvival rate by Gender:")
by_gender = (
    df.groupby("Sex")["Survived"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "Survival Rate", "count": "Passengers"})
)
by_gender["Survival Rate"] = (by_gender["Survival Rate"] * 100).round(1)
print(by_gender)

# --- Average fare paid per class ---
print("\nAverage Fare by Passenger Class:")
avg_fare = df.groupby("Pclass")["Fare"].mean().round(2)
print(avg_fare)

# --- Age distribution of survivors vs non-survivors ---
print("\nAverage Age (Survived vs Not):")
age_survival = df.groupby("Survived")["Age"].mean().round(1)
age_survival.index = age_survival.index.map({0: "Did not survive", 1: "Survived"})
print(age_survival)

# --- Filter: children (under 12) survival ---
children = df[df["Age"] < 12]
child_survival = children["Survived"].mean() * 100
print(f"\nChildren (<12 yrs) survival rate: {child_survival:.1f}%  ({len(children)} children total)")

# --- Filter: top 10% highest fares ---
fare_90th = df["Fare"].quantile(0.90)
top_payers = df[df["Fare"] >= fare_90th]
top_survival = top_payers["Survived"].mean() * 100
print(f"Top 10% fare payers (≥£{fare_90th:.1f}) survival rate: {top_survival:.1f}%")

# ─────────────────────────────────────────────
# 4. INSIGHT SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("INSIGHT SUMMARY")
print("=" * 50)
print(f"""
1. Overall, only {survival_rate:.1f}% of passengers survived the Titanic disaster.

2. Class mattered a lot:
   - 1st class: {by_class.loc[1, 'Survival Rate']}% survived
   - 2nd class: {by_class.loc[2, 'Survival Rate']}% survived
   - 3rd class: {by_class.loc[3, 'Survival Rate']}% survived
   Wealthier passengers had far better survival odds.

3. Gender was the strongest predictor:
   - Women: {by_gender.loc['female', 'Survival Rate']}% survived
   - Men:   {by_gender.loc['male', 'Survival Rate']}% survived
   The "women and children first" policy is clearly reflected.

4. Children under 12 had a {child_survival:.1f}% survival rate,
   higher than the overall average.

5. The top 10% of fare-payers survived at {top_survival:.1f}%, reinforcing
   that wealth and cabin location (closer to lifeboats) played a key role.
""")