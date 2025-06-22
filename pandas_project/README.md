# 🐼 Pandas Mastery: Beginner to Advanced

Welcome to the **Pandas Mastery** repository! This repo is your complete guide to learning Pandas — the most powerful Python library for data manipulation and analysis, with a special focus on **Machine Learning (ML)**, **Artificial Intelligence (AI)**, and **Data Science (DS)**.

---

## 📘 Table of Contents

- [🤔 What is Pandas?](#-what-is-pandas)
- [🚀 Why Use Pandas?](#-why-use-pandas)
- [⚙️ Installation](#️-installation)
- [🟢 Getting Started (Beginner)](#-getting-started-beginner)
- [🟡 Intermediate Concepts](#-intermediate-concepts)
- [🔴 Advanced Techniques](#-advanced-techniques)
- [🧑‍💻 Pandas for Machine Learning, AI, and Data Science](#-pandas-for-machine-learning-ai-and-data-science)
- [📚 Resources](#-resources)
- [📄 License](#-license)
- [🙌 Contributions](#-contributions)

---

## 🤔 What is Pandas?

**Pandas (Python Data Analysis Library)** is an open-source Python library that provides high-performance, easy-to-use data structures and data analysis tools.

Built on top of **NumPy**, Pandas is the go-to library for data manipulation, cleaning, and analysis in Python. It's essential for **Data Science**, **Machine Learning**, and **Business Intelligence** workflows.

---

## 🚀 Why Use Pandas?

- 📊 Powerful data structures: **DataFrame** and **Series**
- 🧹 Easy data cleaning and preprocessing capabilities
- 📈 Seamless integration with NumPy, Matplotlib, and ML libraries
- 📁 Read/write multiple file formats (CSV, Excel, JSON, SQL, etc.)
- 🔍 Advanced data filtering, grouping, and aggregation
- ⏰ Time series analysis and date/time handling
- 🔗 Data merging, joining, and reshaping operations
- 📊 Built-in visualization capabilities

---

## ⚙️ Installation

Install using pip:

```bash
pip install pandas
```

For additional functionality:

```bash
pip install pandas[all]  # Includes optional dependencies
```

Or in a Jupyter Notebook:

```python
!pip install pandas
```

---

## 🟢 Getting Started (Beginner)

**Importing Pandas**

```python
import pandas as pd
import numpy as np
```

**Creating DataFrames and Series**

```python
# Create a Series
s = pd.Series([1, 2, 3, 4, 5])
print(s)

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NY', 'LA', 'Chicago']
}
df = pd.DataFrame(data)
print(df)
```

**Basic DataFrame Operations**

```python
# View data
print(df.head())      # First 5 rows
print(df.tail())      # Last 5 rows
print(df.info())      # Data types and info
print(df.describe())  # Statistical summary
```

**Selecting Data**

```python
# Select columns
print(df['Name'])           # Single column
print(df[['Name', 'Age']])  # Multiple columns

# Select rows
print(df.iloc[0])    # By index
print(df.loc[0])     # By label
```

**Basic File Operations**

```python
# Read CSV
df = pd.read_csv('data.csv')

# Save to CSV
df.to_csv('output.csv', index=False)
```

---

## 🟡 Intermediate Concepts

**Data Filtering**

```python
# Boolean indexing
young_people = df[df['Age'] < 30]
ny_residents = df[df['City'] == 'NY']

# Multiple conditions
filtered = df[(df['Age'] > 25) & (df['City'] == 'LA')]
```

**Handling Missing Data**

```python
# Check for missing values
print(df.isnull().sum())

# Drop missing values
df_clean = df.dropna()

# Fill missing values
df_filled = df.fillna(0)  # Fill with 0
df_filled = df.fillna(df.mean())  # Fill with mean
```

**Data Transformation**

```python
# Add new columns
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Adult')

# String operations
df['Name_Upper'] = df['Name'].str.upper()
df['Name_Length'] = df['Name'].str.len()
```

**Grouping and Aggregation**

```python
# Group by column
grouped = df.groupby('City')
print(grouped['Age'].mean())

# Multiple aggregations
agg_data = df.groupby('City').agg({
    'Age': ['mean', 'max'],
    'Name': 'count'
})
```

**Sorting Data**

```python
# Sort by single column
df_sorted = df.sort_values('Age')

# Sort by multiple columns
df_sorted = df.sort_values(['City', 'Age'], ascending=[True, False])
```

---

## 🔴 Advanced Techniques

**Merging and Joining DataFrames**

```python
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Score': [90, 85, 95]})

# Inner join
merged = pd.merge(df1, df2, on='ID', how='inner')

# Left join
left_joined = pd.merge(df1, df2, on='ID', how='left')
```

**Pivot Tables and Reshaping**

```python
# Create pivot table
pivot = df.pivot_table(values='Age', index='City', columns='Name', aggfunc='mean')

# Melt (wide to long format)
melted = pd.melt(df, id_vars=['Name'], value_vars=['Age', 'City'])

# Stack and unstack
stacked = df.set_index(['Name', 'City']).stack()
unstacked = stacked.unstack()
```

**Time Series Analysis**

```python
# Create datetime index
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# Resample time series
monthly = ts.resample('M').mean()
weekly = ts.resample('W').sum()

# Date operations
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
```

**Advanced Data Cleaning**

```python
# Remove duplicates
df_unique = df.drop_duplicates()

# Replace values
df['City'] = df['City'].replace({'NY': 'New York', 'LA': 'Los Angeles'})

# Data type conversion
df['Age'] = df['Age'].astype(int)
df['Date'] = pd.to_datetime(df['Date'])
```

**Window Functions**

```python
# Rolling calculations
df['Rolling_Mean'] = df['Age'].rolling(window=3).mean()
df['Cumulative_Sum'] = df['Age'].cumsum()

# Rank data
df['Age_Rank'] = df['Age'].rank(ascending=False)
```

---

## 🧑‍💻 Pandas for Machine Learning, AI, and Data Science

Pandas is indispensable for ML, AI, and DS workflows. Here are key applications and examples:

### Data Exploration and Profiling

**Comprehensive Data Analysis**

```python
# Quick data overview
def data_profile(df):
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nUnique Values:")
    print(df.nunique())
    return df.describe(include='all')

profile = data_profile(df)
```

**Correlation Analysis**

```python
# Correlation matrix
corr_matrix = df.select_dtypes(include=[np.number]).corr()
print(corr_matrix)

# Find highly correlated features
high_corr = corr_matrix.abs() > 0.8
```

### Feature Engineering

**Creating New Features**

```python
# Binning continuous variables
df['Age_Bins'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior'])

# Interaction features
df['Age_Income_Ratio'] = df['Age'] / df['Income']

# Polynomial features
df['Age_Squared'] = df['Age'] ** 2
```

**Encoding Categorical Variables**

```python
# One-hot encoding
encoded = pd.get_dummies(df, columns=['City', 'Gender'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['City_Encoded'] = le.fit_transform(df['City'])
```

### Data Preprocessing Pipeline

**Complete Preprocessing Function**

```python
def preprocess_data(df):
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove outliers using IQR
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    
    return df

clean_df = preprocess_data(df.copy())
```

### Train-Test Split with Stratification

```python
from sklearn.model_selection import train_test_split

# Stratified split for classification
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Cross-Validation Data Preparation

```python
from sklearn.model_selection import KFold

# Prepare data for cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kfold.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    # Your model training code here
```

### Time Series for ML

**Feature Engineering for Time Series**

```python
# Create lag features
df['Sales_Lag1'] = df['Sales'].shift(1)
df['Sales_Lag7'] = df['Sales'].shift(7)

# Rolling statistics
df['Sales_MA7'] = df['Sales'].rolling(7).mean()
df['Sales_Std7'] = df['Sales'].rolling(7).std()

# Date features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
```

### Model Performance Analysis

**Results Analysis DataFrame**

```python
# Create results comparison
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'Train_Score': [0.85, 0.92, 0.94],
    'Test_Score': [0.82, 0.89, 0.91],
    'Training_Time': [0.1, 2.3, 5.7]
})

# Add performance metrics
results_df['Overfitting'] = results_df['Train_Score'] - results_df['Test_Score']
results_df['Efficiency'] = results_df['Test_Score'] / results_df['Training_Time']

print(results_df.sort_values('Test_Score', ascending=False))
```

### Working with Large Datasets

**Memory Optimization**

```python
# Optimize data types
def optimize_dtypes(df):
    # Optimize integers
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
    
    # Optimize floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    return df

optimized_df = optimize_dtypes(df.copy())
```

**Chunked Processing**

```python
# Process large files in chunks
chunk_list = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    # Process each chunk
    processed_chunk = chunk.groupby('category').sum()
    chunk_list.append(processed_chunk)

# Combine all chunks
final_result = pd.concat(chunk_list, ignore_index=True)
```

---

## 📚 Resources

* [Official Pandas Documentation](https://pandas.pydata.org/docs/)
* [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
* [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
* [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)
* [Real Python Pandas Tutorials](https://realpython.com/pandas-python-explore-dataset/)
* [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)
* [DataCamp Pandas Courses](https://www.datacamp.com/courses/data-manipulation-with-pandas)
* [Towards Data Science - Pandas Articles](https://towardsdatascience.com/tagged/pandas)

### YouTube Channels
* Corey Schafer - Pandas Tutorials
* Keith Galli - Complete Python Pandas Data Science Tutorial
* FreeCodeCamp - Data Analysis with Python

### Books
* "Python for Data Analysis" by Wes McKinney (Pandas creator)
* "Effective Pandas" by Matt Harrison
* "Pandas 1.x Cookbook" by Matt Harrison & Theodore Petrou

---

## 📄 License

This repository is licensed under the **MIT License**.

---

## 🙌 Contributions

Feel free to open **issues** or **pull requests** to improve this guide or add examples!

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution:
- More real-world examples
- Advanced visualization techniques
- Performance optimization tips
- Integration with other ML libraries
- Industry-specific use cases

⭐ If you find this helpful, **star the repo** and **share it** with others!

---

## 📊 Quick Reference Cheat Sheet

### Essential Operations
```python
# Data Loading
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_json('file.json')

# Data Inspection
df.head(), df.tail(), df.info(), df.describe()
df.shape, df.columns, df.dtypes

# Data Selection
df['column'], df[['col1', 'col2']]
df.iloc[0:5], df.loc[0:5]

# Data Filtering
df[df['column'] > value]
df.query('column > value')

# Data Cleaning
df.dropna(), df.fillna(value)
df.drop_duplicates()

# Data Transformation
df.groupby('column').agg()
df.pivot_table()
pd.merge(df1, df2)

# Data Export
df.to_csv('output.csv')
df.to_excel('output.xlsx')
```

Remember: Practice makes perfect! Start with simple operations and gradually work your way up to more complex data manipulations.