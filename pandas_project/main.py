import pandas as pd  # type: ignore

df = pd.read_csv("test.csv")

print("🚨------🚨-------🚨-------🚨------🚨-----🚨-------🚨-----🚨")
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
print("🚨------🚨-------🚨-------🚨------🚨-----🚨-------🚨-----🚨")

# print(df[["Name", "Age", "Sex"]].head(10))
print("🫵------🫵-------🫵-------🫵------🫵-----🫵-------🫵-----🫵")

# sexFemale = df["Sex"] == "female"
# print(len(df[sexFemale & (df["Pclass"] == 1)]))
# print(df[(df["Age"] >= 20) & (df["Age"] <= 30)])
# print(df[(df["Pclass"] == 1) | (df["Pclass"] == 2)])
# print(df[(df["Age"] >= 18)][["Age","Name"]])
# print(df['Pclass'].value_counts().sort_values())

# print(df.groupby(["Sex","Pclass"]).size())
# print(type(df.groupby("Sex")))
# print(df.groupby("Sex")["Age"].mean())
# print(df.groupby("Pclass")["Age"].mean())
# print(df.groupby("Sex")["Age"].agg(["mean", "median", "std", "count"]))
# print(df.groupby("Pclass")["Fare"].mean());
# print(df.groupby("Embarked")["Fare"].agg(["mean", "median", "std", "count"]))

# print(df.groupby(['Pclass','Sex'])["Fare"].mean().unstack())
medians = df.groupby("Sex")["Age"].median()
df["Age"] = df.apply(
    lambda row: medians[row["Sex"]] if pd.isnull(row["Age"]) else row["Age"], axis=1
)
# print(df.isnull().sum())
df_cleaned = df.copy()
medians = df_cleaned.groupby("Sex")["Age"].median()
print(df_cleaned.row)
