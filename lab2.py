# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    LabelEncoder,
    OneHotEncoder
)

from sklearn.feature_selection import SelectKBest, f_classif

# Display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Create sample dataset
data = {
    "Name": ["Aman", "Riya", "Rahul", "Priya", "Karan",
             "Neha", "Vikas", "Anjali", "Rohit", "Sneha"],

    "Age": [20, 21, np.nan, 22, 25, 23, 21, np.nan, 24, 26],

    "Salary": [25000, 30000, 28000, np.nan, 35000,
               40000, 32000, 30000, 1000000, 38000],

    "Department": ["IT", "HR", "IT", "Sales", np.nan,
                   "HR", "IT", "Sales", "IT", "HR"],

    "Experience": [1, 2, 2, 3, 4, 5, 3, 2, 10, 6],

    "JoinDate": [
        "2022-01-15", "2021-03-20", "2022-06-10",
        "2020-08-25", "2019-04-12", "2021-11-05",
        "2022-02-18", "2020-09-30", "2018-07-22",
        "2019-12-10"
    ]
}

df = pd.DataFrame(data)

print(df)

#1
print("Missing values in each column:")
print(df.isnull().sum())

#2
# Calculate percentage of missing values

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("Percentage of missing values:")
print(missing_percentage)

#3
# Create a copy of original dataset
df_drop_rows = df.copy()

# Remove rows containing missing values
df_drop_rows = df_drop_rows.dropna()

print("Dataset after removing rows with missing values:")
print(df_drop_rows)

#4
df_drop_columns = df.copy()

# Calculate missing percentage
missing_percentage = df_drop_columns.isnull().mean() * 100

# Select columns having <= 50% missing values
df_drop_columns = df_drop_columns.loc[
    :, missing_percentage <= 50
]

print("Dataset after removing columns with >50% missing values:")
print(df_drop_columns)

#5
df_mean = df.copy()

# Replace missing Age with mean
df_mean["Age"] = df_mean["Age"].fillna(
    df_mean["Age"].mean()
)

# Replace missing Salary with mean
df_mean["Salary"] = df_mean["Salary"].fillna(
    df_mean["Salary"].mean()
)

print("After mean imputation:")
print(df_mean)

#6
df_median = df.copy()

# Replace missing numerical values with median
df_median["Age"] = df_median["Age"].fillna(
    df_median["Age"].median()
)

df_median["Salary"] = df_median["Salary"].fillna(
    df_median["Salary"].median()
)

print("After median imputation:")
print(df_median)

#7
df_mode = df.copy()

# Find mode of Department
department_mode = df_mode["Department"].mode()[0]

# Replace missing categorical value
df_mode["Department"] = df_mode["Department"].fillna(
    department_mode
)

print("After mode imputation:")
print(df_mode)

#8
df_forward = df.copy()

# Forward fill
df_forward = df_forward.ffill()

print("After Forward Fill:")
print(df_forward)

#9
df_backward = df.copy()

# Backward fill
df_backward = df_backward.bfill()

print("After Backward Fill:")
print(df_backward)

#10
print("Missing values BEFORE treatment:")
print(df.isnull().sum())

# Handle missing values
df_clean = df.copy()

df_clean["Age"] = df_clean["Age"].fillna(
    df_clean["Age"].median()
)

df_clean["Salary"] = df_clean["Salary"].fillna(
    df_clean["Salary"].median()
)

df_clean["Department"] = df_clean["Department"].fillna(
    df_clean["Department"].mode()[0]
)

print("\nMissing values AFTER treatment:")
print(df_clean.isnull().sum())

#11
df_iqr = df_clean.copy()

Q1 = df_iqr["Salary"].quantile(0.25)
Q3 = df_iqr["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df_iqr[
    (df_iqr["Salary"] < lower_limit) |
    (df_iqr["Salary"] > upper_limit)
]

print("Outliers using IQR:")
print(outliers)

#12
df_zscore = df_clean.copy()

mean_salary = df_zscore["Salary"].mean()
std_salary = df_zscore["Salary"].std()

df_zscore["Z_Score"] = (
    (df_zscore["Salary"] - mean_salary) / std_salary
)

outliers = df_zscore[
    abs(df_zscore["Z_Score"]) > 3
]

print("Outliers using Z-score:")
print(outliers)

#13
plt.figure(figsize=(8, 5))

sns.boxplot(y=df_clean["Salary"])

plt.title("Box Plot of Salary")
plt.ylabel("Salary")

plt.show()

#14
plt.figure(figsize=(8, 5))

plt.scatter(
    df_clean["Experience"],
    df_clean["Salary"]
)

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")

plt.show()

#15
df_remove_outliers = df_clean.copy()

Q1 = df_remove_outliers["Salary"].quantile(0.25)
Q3 = df_remove_outliers["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df_remove_outliers = df_remove_outliers[
    (df_remove_outliers["Salary"] >= lower_limit) &
    (df_remove_outliers["Salary"] <= upper_limit)
]

print("Dataset after removing outliers:")
print(df_remove_outliers)

#16
df_replace_outliers = df_clean.copy()

Q1 = df_replace_outliers["Salary"].quantile(0.25)
Q3 = df_replace_outliers["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

median_salary = df_replace_outliers["Salary"].median()

# Replace outliers with median
df_replace_outliers["Salary"] = np.where(
    (df_replace_outliers["Salary"] < lower_limit) |
    (df_replace_outliers["Salary"] > upper_limit),
    median_salary,
    df_replace_outliers["Salary"]
)

print(df_replace_outliers)

#17
df_capped = df_clean.copy()

lower_percentile = df_capped["Salary"].quantile(0.05)
upper_percentile = df_capped["Salary"].quantile(0.95)

df_capped["Salary"] = df_capped["Salary"].clip(
    lower=lower_percentile,
    upper=upper_percentile
)

print("Dataset after percentile capping:")
print(df_capped)

#18
print("Original Salary:")
print(df_clean["Salary"])

print("\nSalary after outlier treatment:")
print(df_capped["Salary"])

print("\nOriginal Statistics:")
print(df_clean["Salary"].describe())

print("\nAfter Treatment Statistics:")
print(df_capped["Salary"].describe())

#19
df_minmax = df_clean.copy()

scaler = MinMaxScaler()

df_minmax[["Age", "Salary", "Experience"]] = scaler.fit_transform(
    df_minmax[["Age", "Salary", "Experience"]]
)

print("After Min-Max Normalization:")
print(df_minmax)

#20
df_standard = df_clean.copy()

scaler = StandardScaler()

df_standard[["Age", "Salary", "Experience"]] = scaler.fit_transform(
    df_standard[["Age", "Salary", "Experience"]]
)

print("After Standardization:")
print(df_standard)

#21
df_robust = df_clean.copy()

scaler = RobustScaler()

df_robust[["Age", "Salary", "Experience"]] = scaler.fit_transform(
    df_robust[["Age", "Salary", "Experience"]]
)

print("After Robust Scaling:")
print(df_robust)

#22
df_maxabs = df_clean.copy()

scaler = MaxAbsScaler()

df_maxabs[["Age", "Salary", "Experience"]] = scaler.fit_transform(
    df_maxabs[["Age", "Salary", "Experience"]]
)

print("After Max Absolute Scaling:")
print(df_maxabs)

#23
comparison = pd.DataFrame({
    "Original Salary": df_clean["Salary"],
    "MinMax Salary": df_minmax["Salary"],
    "Standardized Salary": df_standard["Salary"],
    "Robust Salary": df_robust["Salary"],
    "MaxAbs Salary": df_maxabs["Salary"]
})

print(comparison)

#24
plt.figure(figsize=(8, 5))

plt.hist(
    df_clean["Salary"],
    bins=8,
    alpha=0.7
)

plt.title("Original Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(8, 5))

plt.hist(
    df_minmax["Salary"],
    bins=8,
    alpha=0.7
)

plt.title("Min-Max Normalized Salary")
plt.xlabel("Normalized Salary")
plt.ylabel("Frequency")

plt.show()

#25
scaled_data = pd.DataFrame({
    "MinMax": df_minmax["Salary"],
    "Standard": df_standard["Salary"],
    "Robust": df_robust["Salary"],
    "MaxAbs": df_maxabs["Salary"]
})

plt.figure(figsize=(8, 5))

sns.boxplot(data=scaled_data)

plt.title("Comparison of Scaling Techniques")
plt.ylabel("Scaled Values")

plt.show()

#26
print("Original Data:")
print(df_clean[["Age", "Salary", "Experience"]])

print("\nMin-Max Scaling:")
print(df_minmax[["Age", "Salary", "Experience"]])

print("\nStandard Scaling:")
print(df_standard[["Age", "Salary", "Experience"]])

print("\nRobust Scaling:")
print(df_robust[["Age", "Salary", "Experience"]])

print("\nMax Absolute Scaling:")
print(df_maxabs[["Age", "Salary", "Experience"]])

#27
df_label = df_clean.copy()

label_encoder = LabelEncoder()

df_label["Department_Encoded"] = label_encoder.fit_transform(
    df_label["Department"]
)

print("After Label Encoding:")
print(df_label[["Department", "Department_Encoded"]])

#28
df_onehot = df_clean.copy()

one_hot = pd.get_dummies(
    df_onehot,
    columns=["Department"],
    dtype=int
)

print("After One-Hot Encoding:")
print(one_hot)

#29
df_binary = df_clean.copy()

# Convert categories to numerical codes
category_codes = df_binary["Department"].astype("category").cat.codes

# Convert numerical codes into binary
binary_values = category_codes.apply(
    lambda x: format(x, "02b")
)

df_binary["Department_Binary"] = binary_values

print("After Binary Encoding:")
print(df_binary[["Department", "Department_Binary"]])

#30
df_feature = df_clean.copy()

df_feature["SalaryPerExperience"] = (
    df_feature["Salary"] /
    df_feature["Experience"]
)

print(df_feature)

#31
df_date = df_clean.copy()

# Convert JoinDate to datetime
df_date["JoinDate"] = pd.to_datetime(
    df_date["JoinDate"]
)

# Extract date features
df_date["JoinYear"] = df_date["JoinDate"].dt.year
df_date["JoinMonth"] = df_date["JoinDate"].dt.month
df_date["JoinDay"] = df_date["JoinDate"].dt.day

print(df_date)

#32
df_math = df_clean.copy()

# Square root transformation
df_math["Salary_Sqrt"] = np.sqrt(
    df_math["Salary"]
)

# Square of experience
df_math["Experience_Squared"] = (
    df_math["Experience"] ** 2
)

print(df_math)

#33
df_log = df_clean.copy()

# Log transformation
df_log["Salary_Log"] = np.log1p(
    df_log["Salary"]
)

print(df_log[["Salary", "Salary_Log"]])

# Visualize
plt.figure(figsize=(8, 5))

plt.hist(
    df_log["Salary_Log"],
    bins=8
)

plt.title("Log Transformed Salary")
plt.xlabel("Log Salary")
plt.ylabel("Frequency")

plt.show()

#34
df_corr = df_clean.copy()

# Select numerical columns
numeric_data = df_corr[
    ["Age", "Salary", "Experience"]
]

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlation
plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")

plt.show()

#35
# Start with cleaned dataset
final_df = df_clean.copy()

# Convert date
final_df["JoinDate"] = pd.to_datetime(
    final_df["JoinDate"]
)

# Extract date features
final_df["JoinYear"] = final_df["JoinDate"].dt.year
final_df["JoinMonth"] = final_df["JoinDate"].dt.month
final_df["JoinDay"] = final_df["JoinDate"].dt.day

# Create new feature
final_df["SalaryPerExperience"] = (
    final_df["Salary"] /
    final_df["Experience"]
)

# Log transformation
final_df["Salary_Log"] = np.log1p(
    final_df["Salary"]
)

# Remove original date column
final_df = final_df.drop(
    columns=["JoinDate"]
)

# One-hot encode categorical variable
final_df = pd.get_dummies(
    final_df,
    columns=["Department"],
    dtype=int
)

# Scale numerical columns
numerical_columns = [
    "Age",
    "Salary",
    "Experience",
    "SalaryPerExperience",
    "Salary_Log",
    "JoinYear",
    "JoinMonth",
    "JoinDay"
]

scaler = StandardScaler()

final_df[numerical_columns] = scaler.fit_transform(
    final_df[numerical_columns]
)

print("FINAL PREPROCESSED DATASET")
print(final_df)

# Save final dataset
final_df.to_csv(
    "final_preprocessed_dataset.csv",
    index=False
)

print("\nFinal dataset saved successfully.")



