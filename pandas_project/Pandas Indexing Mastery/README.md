# 🎯 Pandas Indexing Mastery: Complete Guide

Welcome to the **Pandas Indexing Mastery** repository! This comprehensive guide covers **all methods of indexing and selecting data** in Pandas DataFrames and Series.

---

## 📘 Table of Contents

- [🤔 What is Pandas Indexing?](#-what-is-pandas-indexing)
- [🔢 Types of Indexing](#-types-of-indexing)
- [🏷️ Column Indexing](#️-column-indexing)
- [📍 Row Indexing](#-row-indexing)
- [🎯 Label-Based Indexing (.loc)](#-label-based-indexing-loc)
- [🔢 Position-Based Indexing (.iloc)](#-position-based-indexing-iloc)
- [🔍 Boolean Indexing](#-boolean-indexing)
- [🎪 Advanced Indexing Techniques](#-advanced-indexing-techniques)
- [⚡ Performance Tips](#-performance-tips)
- [🚀 Best Practices](#-best-practices)
- [📚 Quick Reference](#-quick-reference)

---

## 🤔 What is Pandas Indexing?

**Pandas Indexing** refers to the various methods used to select, filter, and access specific data from DataFrames and Series. It's the foundation of data manipulation in Pandas.

Understanding indexing is crucial for:
- Data selection and filtering
- Data cleaning and preprocessing
- Feature engineering for ML
- Exploratory data analysis

---

## 🔢 Types of Indexing

Pandas offers several indexing methods:

| Method | Type | Description | Use Case |
|--------|------|-------------|----------|
| `df['column']` | Column | Select single column | Quick column access |
| `df[['col1', 'col2']]` | Column | Select multiple columns | Multiple column selection |
| `df.loc[]` | Label-based | Select by labels/names | Flexible label-based selection |
| `df.iloc[]` | Position-based | Select by integer positions | Position-based selection |
| `df.at[]` | Label-based | Single value access | Fast single value access |
| `df.iat[]` | Position-based | Single value access | Fast single value by position |
| `df.query()` | Boolean | Query-style filtering | Complex conditional filtering |
| Boolean indexing | Conditional | Filter based on conditions | Data filtering |

---

## 🏷️ Column Indexing

### Single Column Selection

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['NY', 'LA', 'Chicago', 'Miami'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)

# Method 1: Bracket notation (most common)
name_series = df['Name']
print(type(name_series))  # pandas.Series

# Method 2: Dot notation (only for valid Python identifiers)
age_series = df.Age
print(type(age_series))  # pandas.Series
```

### Multiple Column Selection

```python
# Select multiple columns - returns DataFrame
subset = df[['Name', 'Age']]
print(type(subset))  # pandas.DataFrame

# Using a list variable
columns_to_select = ['Name', 'City', 'Salary']
subset2 = df[columns_to_select]

# Reorder columns
reordered = df[['Salary', 'Name', 'Age', 'City']]
```

### Column Selection with Conditions

```python
# Select columns based on data type
numeric_columns = df.select_dtypes(include=[np.number])
string_columns = df.select_dtypes(include=['object'])

# Select columns by name pattern
columns_with_a = df.filter(regex='^A')  # Columns starting with 'A'
columns_ending_e = df.filter(regex='e$')  # Columns ending with 'e'
```

---

## 📍 Row Indexing

### Basic Row Selection

```python
# First few rows
first_3_rows = df.head(3)

# Last few rows
last_2_rows = df.tail(2)

# Specific rows by slice
rows_1_to_3 = df[1:4]  # Note: excludes row 4

# Single row (returns Series)
first_row = df.iloc[0]
```

### Row Selection by Index Values

```python
# Set custom index
df_indexed = df.set_index('Name')
print(df_indexed.index)

# Select row by index label
alice_row = df_indexed.loc['Alice']

# Select multiple rows by index labels
selected_rows = df_indexed.loc[['Alice', 'Charlie']]
```

---

## 🎯 Label-Based Indexing (.loc)

The `.loc` indexer is used for **label-based** selection and is the most flexible indexing method.

### Syntax: `df.loc[row_indexer, column_indexer]`

### Single Value Selection

```python
# Select single value
value = df.loc[0, 'Name']  # First row, 'Name' column
print(value)  # 'Alice'

# Using at[] for faster single value access
value_fast = df.at[0, 'Name']
```

### Row Selection with .loc

```python
# Single row
first_row = df.loc[0]

# Multiple rows
first_three = df.loc[0:2]  # Note: includes endpoint!

# Multiple non-consecutive rows
selected_rows = df.loc[[0, 2]]
```

### Column Selection with .loc

```python
# All rows, specific columns
subset = df.loc[:, ['Name', 'Age']]

# All rows, column slice
subset2 = df.loc[:, 'Name':'City']  # From 'Name' to 'City'
```

### Combined Row and Column Selection

```python
# Specific rows and columns
subset = df.loc[0:2, ['Name', 'Salary']]

# Using slices
subset2 = df.loc[1:3, 'Age':'Salary']

# Mixed selection
subset3 = df.loc[[0, 2], 'Name':'City']
```

### Advanced .loc Usage

```python
# Boolean array for rows
high_salary = df['Salary'] > 55000
subset = df.loc[high_salary, ['Name', 'Salary']]

# Lambda functions
subset2 = df.loc[lambda x: x['Age'] > 30, :]

# Setting values with .loc
df.loc[df['Age'] > 30, 'Category'] = 'Senior'
df.loc[df['Age'] <= 30, 'Category'] = 'Junior'
```

---

## 🔢 Position-Based Indexing (.iloc)

The `.iloc` indexer is used for **integer position-based** selection.

### Syntax: `df.iloc[row_positions, column_positions]`

### Single Value Selection

```python
# Select single value by position
value = df.iloc[0, 1]  # First row, second column
print(value)  # 25 (Age of Alice)

# Using iat[] for faster single value access
value_fast = df.iat[0, 1]
```

### Row Selection with .iloc

```python
# Single row by position
first_row = df.iloc[0]

# Multiple rows
first_three = df.iloc[0:3]  # Excludes row 3

# Last row
last_row = df.iloc[-1]

# Every other row
every_other = df.iloc[::2]
```

### Column Selection with .iloc

```python
# All rows, specific columns by position
subset = df.iloc[:, [0, 2]]  # First and third columns

# Column slice
subset2 = df.iloc[:, 1:3]  # Second and third columns

# Last two columns
last_cols = df.iloc[:, -2:]
```

### Combined Selection with .iloc

```python
# Specific rows and columns
subset = df.iloc[0:2, [0, 3]]  # First 2 rows, 1st and 4th columns

# Using slices
subset2 = df.iloc[1:3, 1:4]

# Mixed selection
subset3 = df.iloc[[0, 2], -2:]  # 1st and 3rd rows, last 2 columns
```

### Advanced .iloc Patterns

```python
# Random sampling
random_rows = df.iloc[np.random.choice(len(df), 3, replace=False)]

# Top-left corner
top_left = df.iloc[:2, :2]

# Bottom-right corner
bottom_right = df.iloc[-2:, -2:]

# Diagonal selection (custom function needed)
def diagonal_select(df, n):
    indices = [(i, i) for i in range(min(n, len(df), len(df.columns)))]
    return [df.iat[i, j] for i, j in indices]

diagonal_values = diagonal_select(df, 3)
```

---

## 🔍 Boolean Indexing

Boolean indexing allows filtering data based on conditions.

### Basic Boolean Indexing

```python
# Single condition
high_salary = df[df['Salary'] > 55000]

# Multiple conditions with & (and)
young_high_salary = df[(df['Age'] < 30) & (df['Salary'] > 55000)]

# Multiple conditions with | (or)
extreme_ages = df[(df['Age'] < 26) | (df['Age'] > 34)]

# Negation with ~
not_from_ny = df[~(df['City'] == 'NY')]
```

### Advanced Boolean Conditions

```python
# String operations
name_contains_a = df[df['Name'].str.contains('a', case=False)]

# Multiple values with isin()
selected_cities = df[df['City'].isin(['NY', 'LA'])]

# Null value checks
has_null = df[df['Name'].isnull()]
no_null = df[df['Name'].notnull()]

# Between values
mid_age = df[df['Age'].between(25, 35)]
```

### Boolean Indexing with .loc

```python
# Combine boolean indexing with column selection
result = df.loc[df['Salary'] > 55000, ['Name', 'City']]

# Complex conditions
complex_filter = (df['Age'] > 25) & (df['City'] != 'Chicago')
result2 = df.loc[complex_filter, :]
```

### Query Method

```python
# Using query() for readable conditions
result1 = df.query('Age > 30')
result2 = df.query('Age > 30 and Salary < 65000')
result3 = df.query('City in ["NY", "LA"]')

# Using variables in query
min_age = 28
result4 = df.query('Age >= @min_age')

# Complex string conditions
result5 = df.query('Name.str.startswith("A")', engine='python')
```

---

## 🎪 Advanced Indexing Techniques

### MultiIndex (Hierarchical) Indexing

```python
# Create MultiIndex DataFrame
arrays = [
    ['A', 'A', 'B', 'B'],
    ['one', 'two', 'one', 'two']
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df_multi = pd.DataFrame(np.random.randn(4, 2), index=index, columns=['X', 'Y'])

# Select by level
level_a = df_multi.loc['A']

# Select specific index
specific = df_multi.loc[('A', 'one')]

# Cross-section
xs_result = df_multi.xs('one', level='second')

# Slice multiindex
slice_result = df_multi.loc[('A', 'one'):('B', 'one')]
```

### Index Alignment and Reindexing

```python
# Reindex with new index
new_index = ['Alice', 'Bob', 'Eve', 'Frank']
df_reindexed = df.set_index('Name').reindex(new_index, fill_value=0)

# Reset index
df_reset = df.set_index('Name').reset_index()

# Set multiple columns as index
df_multi_idx = df.set_index(['City', 'Name'])
```

### Conditional Selection with where()

```python
# Replace values that don't meet condition with NaN
result = df.where(df['Age'] > 30)

# Replace with custom value
result2 = df.where(df['Age'] > 30, 'Young')

# Apply different condition to different columns
conditions = {
    'Age': df['Age'] > 30,
    'Salary': df['Salary'] > 55000
}
```

### Fancy Indexing with NumPy Arrays

```python
# Using numpy arrays for indexing
idx = np.array([0, 2])
subset = df.iloc[idx]

# Boolean numpy array
bool_array = np.array([True, False, True, False])
subset2 = df.loc[bool_array]
```

---

## ⚡ Performance Tips

### Optimization Strategies

```python
# 1. Use .at[] and .iat[] for single value access
# Slow
value = df.loc[0, 'Name']

# Fast
value = df.at[0, 'Name']

# 2. Avoid chained indexing
# Slow and problematic
df[df['Age'] > 30]['Salary'] = 100000  # SettingWithCopyWarning

# Fast and correct
df.loc[df['Age'] > 30, 'Salary'] = 100000

# 3. Use query() for complex conditions
# Can be faster for large DataFrames
result = df.query('Age > 30 and City == "NY"')

# 4. Index optimization
df_indexed = df.set_index('Name')  # Faster repeated access by name
```

### Memory Efficient Indexing

```python
# Use categorical data for repeated strings
df['City'] = df['City'].astype('category')

# Use appropriate data types
df['Age'] = df['Age'].astype('int8')  # If age < 128

# Avoid unnecessary copies
view = df.loc[df['Age'] > 30]  # Creates a view when possible
```

---

## 🚀 Best Practices

### 1. Choose the Right Indexing Method

```python
# For column selection
single_col = df['column_name']        # Simple and clear
multi_cols = df[['col1', 'col2']]     # Multiple columns

# For row selection
by_label = df.loc[row_labels]         # When you know labels
by_position = df.iloc[positions]      # When you know positions

# For conditional selection
filtered = df[df['column'] > value]   # Simple conditions
complex = df.query('condition')       # Complex conditions
```

### 2. Avoid Common Pitfalls

```python
# ❌ Chained indexing (can cause SettingWithCopyWarning)
df[df['Age'] > 30]['New_Column'] = 'value'

# ✅ Proper indexing
df.loc[df['Age'] > 30, 'New_Column'] = 'value'

# ❌ Using deprecated ix indexer
# df.ix[0, 'Name']  # Don't use this

# ✅ Use loc or iloc explicitly
df.loc[0, 'Name']  # Label-based
df.iloc[0, 0]      # Position-based
```

### 3. Handle Missing Data

```python
# Check for missing values before indexing
if not df['column'].isnull().any():
    result = df[df['column'] > threshold]

# Use safe indexing methods
safe_result = df.query('column > @threshold').dropna()
```

### 4. Use Meaningful Index Names

```python
# Set descriptive index
df_with_index = df.set_index('Name')
df_with_index.index.name = 'Employee_Name'

# Use meaningful MultiIndex names
df_multi.index.names = ['Department', 'Employee_ID']
```

---

## 📚 Quick Reference

### Indexing Cheat Sheet

```python
# COLUMN SELECTION
df['col']                    # Single column (Series)
df[['col1', 'col2']]        # Multiple columns (DataFrame)
df.col                      # Dot notation (if valid identifier)

# ROW SELECTION
df[start:end]               # Slice rows
df.head(n)                  # First n rows
df.tail(n)                  # Last n rows

# LABEL-BASED (.loc)
df.loc[row, col]            # Single value
df.loc[rows, cols]          # Multiple rows/columns
df.loc[:, 'col1':'col3']    # Column slice
df.loc[condition, cols]     # Conditional with columns

# POSITION-BASED (.iloc)
df.iloc[row, col]           # Single value by position
df.iloc[rows, cols]         # Multiple by position
df.iloc[:, 1:4]            # Column slice by position
df.iloc[-1]                # Last row

# BOOLEAN INDEXING
df[df['col'] > value]       # Simple condition
df[(df['col1'] > val1) & (df['col2'] < val2)]  # Multiple conditions
df[df['col'].isin(values)]  # Check membership
df.query('col > value')     # Query string

# FAST ACCESS
df.at[row, col]            # Fast single value (label)
df.iat[row, col]           # Fast single value (position)

# SETTING VALUES
df.loc[condition, 'col'] = value    # Set with condition
df['new_col'] = values              # Add new column
```

### Common Patterns

```python
# Data exploration
df.head()                           # Quick peek
df.info()                          # Data types and memory
df.describe()                      # Statistics
df.isnull().sum()                  # Missing values count

# Filtering patterns
high_values = df[df['numeric_col'] > df['numeric_col'].mean()]
recent_data = df[df['date_col'] > '2023-01-01']
text_filter = df[df['text_col'].str.contains('pattern')]

# Selection patterns
features = df.drop('target', axis=1)  # All except target
numeric_only = df.select_dtypes(include=[np.number])
sample_data = df.sample(frac=0.1)     # Random 10% sample
```

### Performance Comparison

| Operation | Slow ❌ | Fast ✅ |
|-----------|---------|---------|
| Single value | `df.loc[0, 'col']` | `df.at[0, 'col']` |
| Single value by position | `df.iloc[0, 1]` | `df.iat[0, 1]` |
| Chained operations | `df[cond1][cond2]` | `df[cond1 & cond2]` |
| Multiple conditions | Multiple brackets | `df.query()` |
| String operations | Loop through values | `df['col'].str.method()` |

---

## 🎓 Practice Exercises

Try these exercises to master Pandas indexing:

1. **Basic Selection**: Select all rows where age > 30 and salary > 60000
2. **Advanced Filtering**: Find employees whose name starts with 'A' and work in either 'NY' or 'LA'
3. **Performance Challenge**: Compare the speed of `.loc`, `.iloc`, `.at`, and `.iat` for single value access
4. **MultiIndex Practice**: Create a hierarchical index and practice cross-section selection
5. **Data Cleaning**: Use boolean indexing to identify and handle outliers in numerical columns

Remember: The key to mastering Pandas indexing is practice and understanding when to use each method! 🚀