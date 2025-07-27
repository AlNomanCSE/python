# Pandas: Complete Guide for AI/DS/ML

## Table of Contents
1. [What is Pandas?](#what-is-pandas)
2. [Why Use Pandas?](#why-use-pandas)
3. [Core Data Structures](#core-data-structures)
4. [Essential Operations](#essential-operations)
5. [Data Manipulation](#data-manipulation)
6. [Data Analysis](#data-analysis)
7. [Advanced Techniques](#advanced-techniques)
8. [Common Patterns for AI/DS/ML](#common-patterns-for-aids-ml)

## What is Pandas?

**Pandas** is a powerful Python library for data manipulation and analysis. It provides:
- **DataFrames**: 2D labeled data structures (like Excel spreadsheets or SQL tables)
- **Series**: 1D labeled arrays (like a single column)
- **High-performance**: Built on NumPy for fast operations
- **Flexible**: Handle missing data, different data types, and complex indexing
- **Integrated**: Works seamlessly with other data science libraries

```python
import pandas as pd
import numpy as np

# Basic pandas structures
series = pd.Series([1, 2, 3, 4, 5])
dataframe = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85.5, 92.0, 78.5]
})

print("Series:")
print(series)
print("\nDataFrame:")
print(dataframe)
```

## Why Use Pandas?

Pandas is essential in AI/DS/ML because it:

1. **Data loading**: Read from CSV, Excel, JSON, SQL databases, APIs
2. **Data cleaning**: Handle missing values, duplicates, and data type issues
3. **Data transformation**: Reshape, pivot, merge, and aggregate data
4. **Exploratory analysis**: Quick statistics, grouping, and visualization
5. **Feature engineering**: Create new features from existing data
6. **Data preprocessing**: Prepare data for machine learning models
7. **Time series**: Advanced date/time handling and analysis

```python
# Common AI/DS/ML use cases
# Loading dataset
df = pd.read_csv('dataset.csv')  # Load data from file

# Quick exploration
print(df.info())          # Data types and missing values
print(df.describe())      # Statistical summary
print(df.head())          # First 5 rows

# Feature engineering
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 50, 100], labels=['Young', 'Middle', 'Senior'])
df['score_normalized'] = (df['score'] - df['score'].mean()) / df['score'].std()
```

## Core Data Structures

### Series - 1D Labeled Array

```python
# Creating Series
# Method 1: From list
ages = pd.Series([25, 30, 35, 28, 32])
print(f"Basic series:\n{ages}")

# Method 2: With custom index
named_ages = pd.Series([25, 30, 35, 28, 32], 
                      index=['Alice', 'Bob', 'Charlie', 'David', 'Eve'])
print(f"\nNamed series:\n{named_ages}")

# Method 3: From dictionary
scores_dict = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'David': 88}
scores = pd.Series(scores_dict)
print(f"\nFrom dictionary:\n{scores}")

# Method 4: With data type specification
temperatures = pd.Series([20.5, 22.3, 19.8, 21.1], dtype='float64')
print(f"\nWith dtype:\n{temperatures}")

# Series attributes
print(f"\nSeries properties:")
print(f"Values: {scores.values}")
print(f"Index: {scores.index}")
print(f"Name: {scores.name}")
print(f"Data type: {scores.dtype}")
print(f"Size: {scores.size}")
```

### DataFrame - 2D Labeled Data Structure

```python
# Creating DataFrames
# Method 1: From dictionary
data_dict = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing'],
    'salary': [75000, 65000, 85000, 60000, 70000],
    'performance_score': [4.2, 3.8, 4.5, 3.5, 4.0]
}
df = pd.DataFrame(data_dict)
print("DataFrame from dictionary:")
print(df)

# Method 2: From list of dictionaries
employees = [
    {'name': 'Alice', 'age': 25, 'dept': 'Eng', 'salary': 75000},
    {'name': 'Bob', 'age': 30, 'dept': 'Marketing', 'salary': 65000},
    {'name': 'Charlie', 'age': 35, 'dept': 'Eng', 'salary': 85000}
]
df_from_list = pd.DataFrame(employees)
print(f"\nFrom list of dicts:\n{df_from_list}")

# Method 3: From numpy array with column names
np_data = np.random.randn(4, 3)
df_numpy = pd.DataFrame(np_data, 
                       columns=['feature_1', 'feature_2', 'feature_3'],
                       index=['sample_1', 'sample_2', 'sample_3', 'sample_4'])
print(f"\nFrom numpy array:\n{df_numpy}")

# DataFrame attributes
print(f"\nDataFrame properties:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print(f"Data types:\n{df.dtypes}")
```

## Essential Operations

### Data Access and Selection

```python
# Create sample dataset
df = pd.DataFrame({
    'student_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'math_score': [95, 87, 92, 78, 89],
    'science_score': [88, 91, 85, 82, 94],
    'english_score': [92, 79, 88, 91, 87],
    'grade': ['A', 'B', 'A', 'C', 'B']
})

# Column selection
print("Single column (Series):")
print(df['name'])

print("\nMultiple columns (DataFrame):")
print(df[['name', 'math_score', 'science_score']])

# Row selection
print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))

print("\nSpecific rows by index:")
print(df.iloc[1:4])  # Index-based selection

# Conditional selection
print("\nStudents with math score > 90:")
high_math = df[df['math_score'] > 90]
print(high_math[['name', 'math_score']])

print("\nMultiple conditions:")
top_students = df[(df['math_score'] > 85) & (df['science_score'] > 85)]
print(top_students[['name', 'math_score', 'science_score']])

# Using query method
print("\nUsing query method:")
print(df.query('math_score > 90 and grade == "A"')[['name', 'math_score', 'grade']])
```

### Data Types and Conversion

```python
# Data type inspection
print("Data types:")
print(df.dtypes)

# Type conversion
df_copy = df.copy()
df_copy['student_id'] = df_copy['student_id'].astype('str')
df_copy['grade'] = df_copy['grade'].astype('category')

print("\nAfter type conversion:")
print(df_copy.dtypes)

# Memory usage
print(f"\nMemory usage comparison:")
print(f"Original: {df.memory_usage(deep=True).sum()} bytes")
print(f"Optimized: {df_copy.memory_usage(deep=True).sum()} bytes")

# Numeric conversion with error handling
mixed_data = pd.Series(['1', '2', '3.5', 'invalid', '5'])
print(f"\nOriginal mixed data: {mixed_data}")

# Convert to numeric, coerce errors to NaN
numeric_data = pd.to_numeric(mixed_data, errors='coerce')
print(f"Converted to numeric: {numeric_data}")

# Date conversion
date_strings = pd.Series(['2024-01-15', '2024-02-20', '2024-03-10'])
dates = pd.to_datetime(date_strings)
print(f"\nConverted to datetime: {dates}")
```

### Missing Data Handling

```python
# Create dataset with missing values
data_with_missing = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 7, 8, np.nan],
    'C': [9, 10, 11, np.nan, 13],
    'D': ['x', 'y', np.nan, 'z', 'w']
}
df_missing = pd.DataFrame(data_with_missing)
print("Dataset with missing values:")
print(df_missing)

# Detect missing values
print(f"\nMissing values per column:")
print(df_missing.isnull().sum())

print(f"\nPercentage missing:")
print((df_missing.isnull().sum() / len(df_missing)) * 100)

# Drop missing values
print(f"\nDrop rows with any missing values:")
print(df_missing.dropna())

print(f"\nDrop columns with any missing values:")
print(df_missing.dropna(axis=1))

print(f"\nDrop rows where all values are missing:")
print(df_missing.dropna(how='all'))

# Fill missing values
print(f"\nFill with constant value:")
print(df_missing.fillna(0))

print(f"\nFill with column mean (numeric only):")
df_filled = df_missing.copy()
for col in ['A', 'B', 'C']:
    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
print(df_filled)

print(f"\nForward fill:")
print(df_missing.fillna(method='ffill'))

print(f"\nBackward fill:")
print(df_missing.fillna(method='bfill'))
```

## Data Manipulation

### Adding and Modifying Data

```python
# Create base dataset
df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D', 'E'],
    'price': [10.5, 15.2, 8.7, 12.3, 9.8],
    'quantity': [100, 85, 120, 95, 110],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Clothing']
})

# Add new columns
df['revenue'] = df['price'] * df['quantity']
df['price_tier'] = pd.cut(df['price'], bins=[0, 10, 15, float('inf')], 
                         labels=['Low', 'Medium', 'High'])

print("With new columns:")
print(df)

# Conditional column creation
df['high_revenue'] = df['revenue'] > 1000
df['status'] = np.where(df['quantity'] > 100, 'High Stock', 'Low Stock')

print(f"\nWith conditional columns:")
print(df[['product', 'quantity', 'status', 'high_revenue']])

# Apply functions
df['price_rounded'] = df['price'].apply(lambda x: round(x, 1))
df['product_upper'] = df['product'].apply(str.upper)

# Complex transformations with apply
def categorize_product(row):
    if row['category'] == 'Electronics' and row['price'] > 10:
        return 'Premium Electronics'
    elif row['category'] == 'Electronics':
        return 'Budget Electronics'
    else:
        return row['category']

df['detailed_category'] = df.apply(categorize_product, axis=1)
print(f"\nWith detailed category:")
print(df[['product', 'category', 'detailed_category']])
```

### Sorting and Ranking

```python
# Sample dataset for sorting
students = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'grade': ['A', 'B', 'A', 'C', 'B', 'A'],
    'math_score': [95, 87, 92, 78, 89, 94],
    'total_score': [280, 250, 275, 230, 265, 285]
})

# Sort by single column
print("Sorted by math score (ascending):")
print(students.sort_values('math_score'))

print("\nSorted by math score (descending):")
print(students.sort_values('math_score', ascending=False))

# Sort by multiple columns
print("\nSorted by grade, then by math score:")
print(students.sort_values(['grade', 'math_score'], ascending=[True, False]))

# Ranking
students['math_rank'] = students['math_score'].rank(ascending=False)
students['total_rank'] = students['total_score'].rank(method='dense', ascending=False)

print(f"\nWith rankings:")
print(students[['name', 'math_score', 'math_rank', 'total_score', 'total_rank']])

# Top N selection
print(f"\nTop 3 students by total score:")
top_3 = students.nlargest(3, 'total_score')
print(top_3[['name', 'total_score']])
```

### Grouping and Aggregation

```python
# Sales dataset for grouping
sales_data = pd.DataFrame({
    'region': ['North', 'South', 'North', 'East', 'West', 'South', 'East', 'West'],
    'product': ['A', 'A', 'B', 'A', 'B', 'B', 'A', 'A'],
    'sales': [100, 150, 200, 120, 180, 160, 140, 110],
    'profit': [20, 30, 40, 24, 36, 32, 28, 22],
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2']
})

print("Sales dataset:")
print(sales_data)

# Basic grouping
print(f"\nSales by region:")
region_sales = sales_data.groupby('region')['sales'].sum()
print(region_sales)

# Multiple aggregations
print(f"\nMultiple aggregations by region:")
region_agg = sales_data.groupby('region').agg({
    'sales': ['sum', 'mean', 'count'],
    'profit': ['sum', 'mean']
})
print(region_agg)

# Multiple grouping columns
print(f"\nGrouped by region and product:")
region_product = sales_data.groupby(['region', 'product'])['sales'].sum()
print(region_product)

# Custom aggregation functions
def sales_range(series):
    return series.max() - series.min()

print(f"\nCustom aggregation (sales range by region):")
custom_agg = sales_data.groupby('region')['sales'].agg(['sum', 'mean', sales_range])
print(custom_agg)

# Transform and filter
print(f"\nSales with group statistics:")
sales_data['region_total'] = sales_data.groupby('region')['sales'].transform('sum')
sales_data['sales_pct_of_region'] = (sales_data['sales'] / sales_data['region_total']) * 100
print(sales_data[['region', 'sales', 'region_total', 'sales_pct_of_region']])
```

### Merging and Joining

```python
# Create sample datasets for merging
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 2, 3, 4, 6],  # Note: customer 6 doesn't exist in customers
    'product': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone', 'Tablet'],
    'amount': [1200, 800, 500, 1200, 800, 500]
})

print("Customers:")
print(customers)
print("\nOrders:")
print(orders)

# Inner join (default)
print(f"\nInner join:")
inner_merged = pd.merge(customers, orders, on='customer_id')
print(inner_merged)

# Left join
print(f"\nLeft join:")
left_merged = pd.merge(customers, orders, on='customer_id', how='left')
print(left_merged)

# Right join
print(f"\nRight join:")
right_merged = pd.merge(customers, orders, on='customer_id', how='right')
print(right_merged)

# Outer join
print(f"\nOuter join:")
outer_merged = pd.merge(customers, orders, on='customer_id', how='outer')
print(outer_merged)

# Merge with different column names
products = pd.DataFrame({
    'prod_name': ['Laptop', 'Phone', 'Tablet'],
    'category': ['Electronics', 'Electronics', 'Electronics'],
    'cost': [1000, 600, 400]
})

print(f"\nMerge with different column names:")
detailed_orders = pd.merge(orders, products, left_on='product', right_on='prod_name')
print(detailed_orders[['order_id', 'customer_id', 'product', 'amount', 'category', 'cost']])
```

## Data Analysis

### Statistical Analysis

```python
# Create sample dataset for analysis
np.random.seed(42)
n_samples = 1000

dataset = pd.DataFrame({
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10, 0.5, n_samples),
    'experience': np.random.gamma(2, 3, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], n_samples)
})

# Clean up data
dataset['age'] = dataset['age'].clip(22, 65)
dataset['experience'] = dataset['experience'].clip(0, 40)

print("Dataset info:")
print(dataset.info())

# Descriptive statistics
print(f"\nDescriptive statistics:")
print(dataset.describe())

print(f"\nStatistics for categorical variables:")
print(dataset.describe(include=['object']))

# Specific statistics
print(f"\nSpecific statistics:")
print(f"Age - Mean: {dataset['age'].mean():.2f}, Median: {dataset['age'].median():.2f}")
print(f"Income - Mean: {dataset['income'].mean():.2f}, Std: {dataset['income'].std():.2f}")
print(f"Experience - Min: {dataset['experience'].min():.2f}, Max: {dataset['experience'].max():.2f}")

# Correlation analysis
numeric_cols = ['age', 'income', 'experience']
correlation_matrix = dataset[numeric_cols].corr()
print(f"\nCorrelation matrix:")
print(correlation_matrix)

# Value counts for categorical variables
print(f"\nEducation distribution:")
print(dataset['education'].value_counts())

print(f"\nDepartment distribution:")
print(dataset['department'].value_counts(normalize=True))  # Proportions
```

### Pivot Tables and Cross-tabulation

```python
# Sample sales data for pivot analysis
sales_pivot_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=200, freq='D'),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
    'product': np.random.choice(['Product_A', 'Product_B', 'Product_C'], 200),
    'salesperson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David'], 200),
    'sales': np.random.uniform(100, 1000, 200),
    'units': np.random.randint(1, 20, 200)
})

sales_pivot_data['month'] = sales_pivot_data['date'].dt.month
sales_pivot_data['quarter'] = sales_pivot_data['date'].dt.quarter

print("Sample sales data:")
print(sales_pivot_data.head())

# Basic pivot table
print(f"\nSales by region and product:")
pivot_basic = sales_pivot_data.pivot_table(
    values='sales', 
    index='region', 
    columns='product', 
    aggfunc='sum'
)
print(pivot_basic)

# Multiple aggregations
print(f"\nMultiple aggregations:")
pivot_multi = sales_pivot_data.pivot_table(
    values=['sales', 'units'],
    index='region',
    columns='product',
    aggfunc={'sales': 'sum', 'units': 'mean'}
)
print(pivot_multi)

# Cross-tabulation
print(f"\nCross-tabulation of region and product:")
crosstab = pd.crosstab(sales_pivot_data['region'], sales_pivot_data['product'])
print(crosstab)

print(f"\nCross-tabulation with percentages:")
crosstab_pct = pd.crosstab(sales_pivot_data['region'], sales_pivot_data['product'], 
                          normalize='index') * 100
print(crosstab_pct.round(2))

# Time-based pivot
print(f"\nQuarterly sales by region:")
quarterly_pivot = sales_pivot_data.pivot_table(
    values='sales',
    index='quarter',
    columns='region',
    aggfunc='sum'
)
print(quarterly_pivot)
```

### Time Series Analysis

```python
# Create time series dataset
date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
time_series_data = pd.DataFrame({
    'date': date_range,
    'value': np.random.randn(len(date_range)).cumsum() + 100,
    'category': np.random.choice(['A', 'B'], len(date_range))
})

# Add noise and trend
time_series_data['value'] += np.sin(np.arange(len(date_range)) * 2 * np.pi / 365) * 10  # Seasonal
time_series_data['value'] += np.arange(len(date_range)) * 0.01  # Trend

# Set date as index
time_series_data.set_index('date', inplace=True)

print("Time series data:")
print(time_series_data.head())

# Resampling
print(f"\nMonthly average:")
monthly_avg = time_series_data['value'].resample('M').mean()
print(monthly_avg.head())

print(f"\nQuarterly statistics:")
quarterly_stats = time_series_data['value'].resample('Q').agg(['mean', 'std', 'min', 'max'])
print(quarterly_stats)

# Rolling calculations
time_series_data['rolling_mean_7d'] = time_series_data['value'].rolling(window=7).mean()
time_series_data['rolling_std_30d'] = time_series_data['value'].rolling(window=30).std()

print(f"\nWith rolling statistics:")
print(time_series_data[['value', 'rolling_mean_7d', 'rolling_std_30d']].head(10))

# Date/time features
time_series_data['year'] = time_series_data.index.year
time_series_data['month'] = time_series_data.index.month
time_series_data['day_of_week'] = time_series_data.index.dayofweek
time_series_data['is_weekend'] = time_series_data['day_of_week'].isin([5, 6])

print(f"\nWith date features:")
print(time_series_data.head())

# Seasonal analysis
print(f"\nAverage value by month:")
monthly_seasonal = time_series_data.groupby('month')['value'].mean()
print(monthly_seasonal)
```

## Advanced Techniques

### Multi-Index and Hierarchical Data

```python
# Create multi-index DataFrame
arrays = [
    ['A', 'A', 'B', 'B', 'C', 'C'],
    ['X', 'Y', 'X', 'Y', 'X', 'Y']
]
tuples = list(zip(*arrays))
multi_index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

multi_df = pd.DataFrame({
    'value1': [1, 2, 3, 4, 5, 6],
    'value2': [10, 20, 30, 40, 50, 60]
}, index=multi_index)

print("Multi-index DataFrame:")
print(multi_df)

# Accessing multi-index data
print(f"\nLevel 'A' data:")
print(multi_df.loc['A'])

print(f"\nSpecific combination:")
print(multi_df.loc[('B', 'Y')])

# Cross-section
print(f"\nCross-section for second level 'X':")
print(multi_df.xs('X', level='second'))

# Groupby with multi-index
print(f"\nGroup by first level:")
print(multi_df.groupby(level='first').sum())

# Stacking and unstacking
print(f"\nUnstack (pivot):")
unstacked = multi_df.unstack('second')
print(unstacked)

print(f"\nStack back:")
stacked = unstacked.stack('second')
print(stacked)
```

### Data Reshaping

```python
# Wide to long format
wide_data = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'math_q1': [85, 90, 78],
    'math_q2': [88, 92, 80],
    'science_q1': [90, 85, 82],
    'science_q2': [92, 87, 85]
})

print("Wide format data:")
print(wide_data)

# Melt to long format
long_data = pd.melt(wide_data, 
                   id_vars=['id', 'name'],
                   var_name='subject_quarter',
                   value_name='score')

# Parse subject and quarter
long_data[['subject', 'quarter']] = long_data['subject_quarter'].str.split('_', expand=True)
long_data = long_data.drop('subject_quarter', axis=1)

print(f"\nLong format data:")
print(long_data)

# Pivot back to wide
wide_back = long_data.pivot_table(
    index=['id', 'name'], 
    columns=['subject', 'quarter'], 
    values='score'
).reset_index()

# Flatten column names
wide_back.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                    for col in wide_back.columns.values]

print(f"\nPivoted back to wide:")
print(wide_back)
```

### Performance Optimization

```python
# Performance comparison examples
import time

# Create large dataset
n_rows = 100000
large_df = pd.DataFrame({
    'A': np.random.randn(n_rows),
    'B': np.random.randn(n_rows),
    'C': np.random.choice(['X', 'Y', 'Z'], n_rows),
    'D': np.random.randint(1, 100, n_rows)
})

# Vectorized operations vs loops
def time_operation(func, description):
    start = time.time()
    result = func()
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")
    return result

# Example 1: Vectorized vs loop
def vectorized_operation():
    return large_df['A'] * large_df['B'] + large_df['D']

def loop_operation():
    result = []
    for idx, row in large_df.iterrows():
        result.append(row['A'] * row['B'] + row['D'])
    return pd.Series(result)

print("Performance comparison:")
vec_result = time_operation(vectorized_operation, "Vectorized operation")
# loop_result = time_operation(loop_operation, "Loop operation")  # Too slow for demo

# Memory optimization
print(f"\nMemory usage optimization:")
print(f"Original memory usage: {large_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize data types
large_df_optimized = large_df.copy()
large_df_optimized['C'] = large_df_optimized['C'].astype('category')
large_df_optimized['D'] = large_df_optimized['D'].astype('int8')

print(f"Optimized memory usage: {large_df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Use query instead of boolean indexing for large datasets
def boolean_indexing():
    return large_df[(large_df['A'] > 0) & (large_df['D'] < 50)]

def query_method():
    return large_df.query('A > 0 and D < 50')

print(f"\nFiltering performance:")
time_operation(boolean_indexing, "Boolean indexing")
time_operation(query_method, "Query method")
```

## Common Patterns for AI/DS/ML

### Data Loading and Initial Exploration

```python
# Simulate loading different data sources
def create_sample_datasets():
    """Create sample datasets simulating real-world scenarios"""
    
    # Customer dataset
    customers = pd.DataFrame({
        'customer_id': range(1, 1001),
        'age': np.random.normal(40, 15, 1000).clip(18, 80),
        'income': np.random.lognormal(10, 0.5, 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000, 
                                    p=[0.3, 0.4, 0.2, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000]
    })
    
    # Transaction dataset
    transactions = pd.DataFrame({
        'transaction_id': range(1, 5001),
        'customer_id': np.random.choice(range(1, 1001), 5000),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 5000),
        'amount': np.random.gamma(2, 50, 5000),
        'transaction_date': pd.date_range('2023-01-01', periods=5000, freq='H')[:5000]
    })
    
    return customers, transactions

# Load and explore data
customers, transactions = create_sample_datasets()

def initial_data_exploration(df, name):
    """Comprehensive initial data exploration"""
    print(f"\n{'='*50}")
    print(f"DATASET: {name}")
    print(f"{'='*50}")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types and missing values
    print(f"\nData types and missing values:")
    info_df = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100
    })
    print(info_df)
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric columns statistics:")
        print(df[numeric_cols].describe())
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical columns unique values:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"    Values: {df[col].value_counts().to_dict()}")



### Feature Engineering Pipeline

```python
# Comprehensive feature engineering pipeline
class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
    
    def create_temporal_features(self, df, date_col):
        """Create temporal features from date column"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_day'] = df[date_col].dt.day
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter
        
        # Derived features
        df[f'{date_col}_is_weekend'] = df[f'{date_col}_dayofweek'].isin([5, 6])
        df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start
        df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end
        
        return df
    
    def create_aggregation_features(self, df, group_cols, agg_cols, agg_funcs):
        """Create aggregation features"""
        df = df.copy()
        
        for group_col in group_cols:
            for agg_col in agg_cols:
                for func in agg_funcs:
                    feature_name = f'{group_col}_{agg_col}_{func}'
                    agg_values = df.groupby(group_col)[agg_col].transform(func)
                    df[feature_name] = agg_values
        
        return df
    
    def create_categorical_features(self, df, cat_cols):
        """Create categorical encodings"""
        df = df.copy()
        
        for col in cat_cols:
            # Frequency encoding
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map)
            
            # Binary encoding for top categories
            top_categories = df[col].value_counts().head(3).index
            for cat in top_categories:
                df[f'{col}_is_{cat}'] = (df[col] == cat).astype(int)
        
        return df
    
    def create_interaction_features(self, df, numeric_cols):
        """Create interaction features between numeric columns"""
        df = df.copy()
        
        from itertools import combinations
        for col1, col2 in combinations(numeric_cols, 2):
            # Multiplicative interactions
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            # Ratio interactions (avoid division by zero)
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        return df

# Apply feature engineering
fe = FeatureEngineer()

# Engineer customer features
customers_engineered = customers.copy()
customers_engineered = fe.create_temporal_features(customers_engineered, 'signup_date')
customers_engineered = fe.create_categorical_features(customers_engineered, ['education', 'region'])

# Create derived features
customers_engineered['income_log'] = np.log1p(customers_engineered['income'])
customers_engineered['age_group'] = pd.cut(customers_engineered['age'], 
                                         bins=[0, 25, 35, 50, 100], 
                                         labels=['Young', 'Adult', 'Middle', 'Senior'])

print("Engineered customer features:")
print(customers_engineered.columns.tolist())
print(f"\nSample of engineered features:")
print(customers_engineered[['customer_id', 'age_group', 'income_log', 
                           'education_frequency', 'region_is_North']].head())

# Engineer transaction features
transactions_engineered = transactions.copy()
transactions_engineered = fe.create_temporal_features(transactions_engineered, 'transaction_date')

# Customer-level aggregations
customer_transaction_features = transactions.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'std', 'count'],
    'transaction_id': 'count'
}).round(2)

# Flatten column names
customer_transaction_features.columns = ['_'.join(col).strip() for col in customer_transaction_features.columns.values]
customer_transaction_features = customer_transaction_features.reset_index()

print(f"\nCustomer transaction aggregations:")
print(customer_transaction_features.head())
```

### Data Cleaning and Preprocessing

```python
# Comprehensive data cleaning pipeline
class DataCleaner:
    def __init__(self):
        self.cleaning_log = []
    
    def log_action(self, action, details):
        """Log cleaning actions"""
        self.cleaning_log.append({'action': action, 'details': details})
    
    def handle_missing_values(self, df, strategy='smart'):
        """Handle missing values with different strategies"""
        df = df.copy()
        initial_missing = df.isnull().sum().sum()
        
        if strategy == 'smart':
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                
                if missing_pct > 50:
                    # Drop columns with >50% missing
                    df = df.drop(columns=[col])
                    self.log_action('drop_column', f'{col}: {missing_pct:.1f}% missing')
                elif df[col].dtype in ['float64', 'int64']:
                    # Fill numeric with median
                    df[col] = df[col].fillna(df[col].median())
                    self.log_action('fill_numeric', f'{col}: filled with median')
                elif df[col].dtype == 'object':
                    # Fill categorical with mode
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
                    self.log_action('fill_categorical', f'{col}: filled with mode ({mode_value})')
        
        final_missing = df.isnull().sum().sum()
        self.log_action('missing_summary', f'Reduced from {initial_missing} to {final_missing} missing values')
        return df
    
    def handle_outliers(self, df, numeric_cols, method='iqr', factor=1.5):
        """Handle outliers using IQR or Z-score method"""
        df = df.copy()
        outliers_removed = 0
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower_bound, upper_bound)
                outliers_removed += outliers_count
                
                self.log_action('outlier_treatment', f'{col}: {outliers_count} outliers capped')
        
        return df
    
    def standardize_text(self, df, text_cols):
        """Standardize text columns"""
        df = df.copy()
        
        for col in text_cols:
            if col in df.columns:
                # Convert to lowercase, strip whitespace, handle common variations
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].replace({'n/a': None, 'na': None, '': None})
                self.log_action('text_standardization', f'{col}: standardized')
        
        return df
    
    def remove_duplicates(self, df, subset=None):
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset)
        final_count = len(df)
        duplicates_removed = initial_count - final_count
        
        self.log_action('duplicate_removal', f'Removed {duplicates_removed} duplicate rows')
        return df
    
    def get_cleaning_summary(self):
        """Get summary of cleaning actions"""
        return pd.DataFrame(self.cleaning_log)

# Apply data cleaning
cleaner = DataCleaner()

# Create some messy data for demonstration
messy_data = customers.copy()
# Add some missing values and outliers
messy_data.loc[0:50, 'income'] = np.nan
messy_data.loc[100:110, 'age'] = 150  # Outliers
messy_data.loc[200:210, 'education'] = 'N/A'

print("Before cleaning:")
print(f"Shape: {messy_data.shape}")
print(f"Missing values: {messy_data.isnull().sum().sum()}")
print(f"Age statistics: {messy_data['age'].describe()}")

# Clean the data
cleaned_data = cleaner.handle_missing_values(messy_data)
cleaned_data = cleaner.handle_outliers(cleaned_data, ['age', 'income'])
cleaned_data = cleaner.standardize_text(cleaned_data, ['education', 'region'])
cleaned_data = cleaner.remove_duplicates(cleaned_data)

print(f"\nAfter cleaning:")
print(f"Shape: {cleaned_data.shape}")
print(f"Missing values: {cleaned_data.isnull().sum().sum()}")
print(f"Age statistics: {cleaned_data['age'].describe()}")

print(f"\nCleaning summary:")
print(cleaner.get_cleaning_summary())
```

### Model Preparation and Data Splitting

```python
# Model preparation pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ModelPreparator:
    def __init__(self):
        self.feature_columns = None
        self.target_column = None
        self.encoders = {}
        self.scaler = None
    
    def prepare_features_target(self, df, target_col, exclude_cols=None):
        """Separate features and target, handle categorical encoding"""
        df = df.copy()
        
        if exclude_cols is None:
            exclude_cols = []
        
        # Separate target
        y = df[target_col].copy()
        
        # Separate features
        feature_cols = [col for col in df.columns 
                       if col != target_col and col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        self.feature_columns = X.columns.tolist()
        self.target_column = target_col
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Data split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        self.scaler = StandardScaler()
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_val_scaled, X_test_scaled

# Prepare data for modeling
# Create a target variable for demonstration
model_data = customers_engineered.copy()
model_data['high_income'] = (model_data['income'] > model_data['income'].median()).astype(int)

# Initialize preparator
preparator = ModelPreparator()

# Prepare features and target
exclude_cols = ['customer_id', 'signup_date', 'income']  # Exclude ID and original income
X, y = preparator.prepare_features_target(model_data, 'high_income', exclude_cols)

print(f"Features prepared:")
print(f"  Feature columns: {len(X.columns)}")
print(f"  Sample features: {X.columns[:5].tolist()}")
print(f"  Target distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_data(X, y)

# Scale features
X_train_scaled, X_val_scaled, X_test_scaled = preparator.scale_features(X_train, X_val, X_test)

print(f"\nScaled features sample:")
print(X_train_scaled.head())

# Feature importance analysis
feature_stats = pd.DataFrame({
    'feature': X.columns,
    'train_mean': X_train.mean(),
    'train_std': X_train.std(),
    'missing_pct': (X_train.isnull().sum() / len(X_train)) * 100
})

print(f"\nFeature statistics:")
print(feature_stats.sort_values('train_std', ascending=False).head(10))
```

### Performance Monitoring and Validation

```python
# Model performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = []
        self.feature_importance = None
    
    def calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive classification metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, confusion_matrix)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def track_experiment(self, experiment_name, model_params, train_metrics, val_metrics):
        """Track experiment results"""
        experiment_data = {
            'experiment': experiment_name,
            'timestamp': pd.Timestamp.now(),
            'model_params': model_params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'overfitting_score': train_metrics['accuracy'] - val_metrics['accuracy']
        }
        
        self.metrics_history.append(experiment_data)
    
    def get_experiment_summary(self):
        """Get summary of all experiments"""
        if not self.metrics_history:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.metrics_history:
            row = {
                'experiment': exp['experiment'],
                'timestamp': exp['timestamp'],
                'train_accuracy': exp['train_metrics']['accuracy'],
                'val_accuracy': exp['val_metrics']['accuracy'],
                'train_f1': exp['train_metrics']['f1_score'],
                'val_f1': exp['val_metrics']['f1_score'],
                'overfitting_score': exp['overfitting_score']
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def analyze_feature_importance(self, model, feature_names):
        """Analyze feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            return importance_df
        else:
            print("Model doesn't support feature importance")
            return None

# Example usage with dummy model results
monitor = PerformanceMonitor()

# Simulate model experiments
experiments = [
    {
        'name': 'baseline_logistic',
        'params': {'C': 1.0, 'solver': 'lbfgs'},
        'train_acc': 0.85, 'val_acc': 0.82, 'train_f1': 0.84, 'val_f1': 0.81
    },
    {
        'name': 'tuned_logistic',
        'params': {'C': 0.1, 'solver': 'lbfgs'},
        'train_acc': 0.83, 'val_acc': 0.84, 'train_f1': 0.82, 'val_f1': 0.83
    },
    {
        'name': 'random_forest',
        'params': {'n_estimators': 100, 'max_depth': 10},
        'train_acc': 0.92, 'val_acc': 0.86, 'train_f1': 0.91, 'val_f1': 0.85
    }
]

# Track experiments
for exp in experiments:
    train_metrics = {'accuracy': exp['train_acc'], 'f1_score': exp['train_f1']}
    val_metrics = {'accuracy': exp['val_acc'], 'f1_score': exp['val_f1']}
    monitor.track_experiment(exp['name'], exp['params'], train_metrics, val_metrics)

# Get experiment summary
experiment_summary = monitor.get_experiment_summary()
print("Experiment Summary:")
print(experiment_summary)

# Find best model
best_model = experiment_summary.loc[experiment_summary['val_f1'].idxmax()]
print(f"\nBest model: {best_model['experiment']}")
print(f"Validation F1: {best_model['val_f1']:.3f}")
print(f"Overfitting score: {best_model['overfitting_score']:.3f}")
```

## Key Takeaways for AI/DS/ML

1. **Data exploration**: Use `info()`, `describe()`, and `value_counts()` for initial analysis
2. **Missing data**: Choose appropriate strategies based on data type and missingness pattern
3. **Feature engineering**: Create temporal, categorical, and interaction features systematically
4. **Data cleaning**: Handle outliers, duplicates, and text standardization consistently
5. **Performance**: Use vectorized operations, optimize data types, and avoid loops
6. **Memory management**: Monitor memory usage and optimize data types for large datasets
7. **Reproducibility**: Set random seeds and track all preprocessing steps

## Practice Exercises

Try these exercises to reinforce your understanding:

```python
# Exercise 1: Customer Analytics
customer_data = pd.DataFrame({
    'customer_id': range(1, 1001),
    'age': np.random.normal(35, 12, 1000),
    'income': np.random.lognormal(10, 0.5, 1000),
    'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000],
    'last_purchase': pd.date_range('2023-01-01', periods=1000, freq='3D')[:1000]
})
# Task: Calculate customer lifetime value, segment customers, identify churn risk

# Exercise 2: Sales Forecasting Data Prep
sales_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', '2023-12-31', freq='D'),
    'sales': np.random.gamma(2, 100, 1461) + np.sin(np.arange(1461) * 2 * np.pi / 365) * 50
})
# Task: Create seasonal features, calculate moving averages, detect anomalies

# Exercise 3: A/B Test Analysis
ab_test_data = pd.DataFrame({
    'user_id': range(1, 10001),
    'variant': np.random.choice(['A', 'B'], 10000),
    'conversion': np.random.binomial(1, 0.1, 10000),
    'revenue': np.random.gamma(2, 25, 10000),
    'signup_date': pd.date_range('2023-01-01', periods=10000, freq='H')[:10000]
})
# Task: Calculate conversion rates, statistical significance, segment analysis

# Exercise 4: Text Data Processing
text_data = pd.DataFrame({
    'review_id': range(1, 1001),
    'product': np.random.choice(['Product_A', 'Product_B', 'Product_C'], 1000),
    'rating': np.random.choice([1, 2, 3, 4, 5], 1000),
    'review_text': ['Sample review text ' + str(i) for i in range(1000)]
})
# Task: Text length features, sentiment analysis prep, rating prediction features

print("Practice these exercises to master pandas for AI/DS/ML!")
```

---

**Remember**: Pandas is the foundation of data manipulation in Python for AI/DS/ML. Master these concepts and patterns to efficiently handle real-world datasets, from initial exploration to model-ready preprocessing!