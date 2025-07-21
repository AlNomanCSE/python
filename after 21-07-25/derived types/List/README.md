# Python Lists: Complete Guide for AI/DS/ML

## Table of Contents
1. [What is a List?](#what-is-a-list)
2. [Why Use Lists?](#why-use-lists)
3. [How to Use Lists](#how-to-use-lists)
4. [Essential List Methods](#essential-list-methods)
5. [Advanced Operations](#advanced-operations)
6. [List Comprehensions](#list-comprehensions)
7. [Performance Considerations](#performance-considerations)
8. [Common Patterns for AI/DS/ML](#common-patterns-for-aids-ml)

## What is a List?

A **list** in Python is an ordered collection of items that can store multiple values in a single variable. Lists are:
- **Mutable**: You can change, add, or remove items after creation
- **Ordered**: Items have a defined order and maintain that order
- **Allow duplicates**: The same value can appear multiple times
- **Heterogeneous**: Can store different data types in the same list

```python
# Examples of lists
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "hello", 3.14, True, [1, 2, 3]]
empty_list = []
```

## Why Use Lists?

Lists are fundamental in AI/DS/ML because they:

1. **Store datasets**: Hold collections of data points, features, or samples
2. **Represent sequences**: Time series data, text sequences, image pixel arrays
3. **Handle batches**: Group multiple samples for batch processing
4. **Store results**: Collect predictions, scores, or intermediate results
5. **Flexible data manipulation**: Easy to filter, transform, and analyze data

```python
# Common use cases in AI/DS/ML
dataset = [
    [1.2, 0.8, 0.3],  # Feature vectors
    [2.1, 1.5, 0.7],
    [0.9, 0.4, 0.2]
]

predictions = [0.85, 0.92, 0.78, 0.95]  # Model predictions
labels = ['cat', 'dog', 'cat', 'dog']   # Classification labels
```

## How to Use Lists

### Creating Lists

```python
# Method 1: Square brackets
fruits = ["apple", "banana", "orange"]

# Method 2: list() constructor
numbers = list(range(1, 6))  # [1, 2, 3, 4, 5]

# Method 3: List comprehension
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# Method 4: Creating from other iterables
text_list = list("hello")  # ['h', 'e', 'l', 'l', 'o']
```

### Accessing Elements

```python
data = [10, 20, 30, 40, 50]

# Indexing (0-based)
first_element = data[0]    # 10
last_element = data[-1]    # 50

# Slicing
subset = data[1:4]         # [20, 30, 40]
every_second = data[::2]   # [10, 30, 50]
reversed_list = data[::-1] # [50, 40, 30, 20, 10]

print(f"First: {first_element}")
print(f"Subset: {subset}")
print(f"Reversed: {reversed_list}")
```

## Essential List Methods

### 1. Adding Elements

```python
# append() - Add single element to end
data = [1, 2, 3]
data.append(4)
print(data)  # [1, 2, 3, 4]

# insert() - Add element at specific position
data.insert(1, 'new')
print(data)  # [1, 'new', 2, 3, 4]

# extend() - Add multiple elements
data.extend([5, 6, 7])
print(data)  # [1, 'new', 2, 3, 4, 5, 6, 7]

# + operator - Concatenate lists
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2
print(combined)  # [1, 2, 3, 4, 5, 6]
```

### 2. Removing Elements

```python
data = [1, 2, 3, 2, 4, 5]

# remove() - Remove first occurrence of value
data.remove(2)
print(data)  # [1, 3, 2, 4, 5]

# pop() - Remove and return element at index
removed = data.pop(1)  # Removes element at index 1
print(f"Removed: {removed}, List: {data}")  # Removed: 3, List: [1, 2, 4, 5]

# pop() without index removes last element
last = data.pop()
print(f"Last: {last}, List: {data}")  # Last: 5, List: [1, 2, 4]

# del - Delete element or slice
del data[0]
print(data)  # [2, 4]

# clear() - Remove all elements
data.clear()
print(data)  # []
```

### 3. Finding and Counting

```python
data = [1, 2, 3, 2, 4, 2, 5]

# index() - Find first occurrence index
first_two_index = data.index(2)
print(f"First 2 at index: {first_two_index}")  # 1

# count() - Count occurrences
count_of_twos = data.count(2)
print(f"Number of 2s: {count_of_twos}")  # 3

# in operator - Check if element exists
exists = 3 in data
print(f"3 exists: {exists}")  # True

# len() - Get list length
length = len(data)
print(f"Length: {length}")  # 7
```

### 4. Sorting and Reversing

```python
data = [3, 1, 4, 1, 5, 9, 2, 6]

# sort() - Sort in place
data.sort()
print(f"Sorted: {data}")  # [1, 1, 2, 3, 4, 5, 6, 9]

# Sort in descending order
data.sort(reverse=True)
print(f"Descending: {data}")  # [9, 6, 5, 4, 3, 2, 1, 1]

# sorted() - Return new sorted list
original = [3, 1, 4, 1, 5]
sorted_copy = sorted(original)
print(f"Original: {original}")      # [3, 1, 4, 1, 5]
print(f"Sorted copy: {sorted_copy}") # [1, 1, 3, 4, 5]

# reverse() - Reverse in place
data.reverse()
print(f"Reversed: {data}")  # [1, 1, 2, 3, 4, 5, 6, 9]
```

### 5. Copying Lists

```python
original = [1, 2, [3, 4], 5]

# Shallow copy
shallow_copy1 = original.copy()
shallow_copy2 = original[:]
shallow_copy3 = list(original)

# Deep copy (for nested lists)
import copy
deep_copy = copy.deepcopy(original)

# Demonstrate difference
original[2][0] = 999
print(f"Original: {original}")      # [1, 2, [999, 4], 5]
print(f"Shallow copy: {shallow_copy1}")  # [1, 2, [999, 4], 5] - nested list affected
print(f"Deep copy: {deep_copy}")    # [1, 2, [3, 4], 5] - unaffected
```

## Advanced Operations

### Statistical Operations

```python
# Useful for data analysis
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Basic statistics
total = sum(data)
average = sum(data) / len(data)
minimum = min(data)
maximum = max(data)

print(f"Sum: {total}, Average: {average}")
print(f"Min: {minimum}, Max: {maximum}")

# Finding min/max with custom key
names = ["Alice", "Bob", "Charlie", "David"]
shortest = min(names, key=len)
longest = max(names, key=len)
print(f"Shortest name: {shortest}, Longest: {longest}")
```

### Filtering and Mapping

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = [x for x in numbers if x % 2 == 0]
print(f"Even numbers: {evens}")

# Map: square all numbers
squares = [x**2 for x in numbers]
print(f"Squares: {squares}")

# Using filter() and map() functions
evens_func = list(filter(lambda x: x % 2 == 0, numbers))
squares_func = list(map(lambda x: x**2, numbers))
print(f"Evens (filter): {evens_func}")
print(f"Squares (map): {squares_func}")
```

## List Comprehensions

List comprehensions are crucial for data manipulation in AI/DS/ML:

```python
# Basic syntax: [expression for item in iterable if condition]

# Example 1: Transform data
temperatures_f = [32, 68, 86, 104, 122]
temperatures_c = [(f - 32) * 5/9 for f in temperatures_f]
print(f"Celsius: {[round(temp, 1) for temp in temperatures_c]}")

# Example 2: Filter and transform
numbers = range(-5, 6)
positive_squares = [x**2 for x in numbers if x > 0]
print(f"Positive squares: {positive_squares}")

# Example 3: Nested comprehensions (matrices)
matrix = [[i*j for j in range(3)] for i in range(3)]
print("3x3 multiplication table:")
for row in matrix:
    print(row)

# Example 4: Flatten nested lists
nested = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in nested for item in sublist]
print(f"Flattened: {flattened}")
```

## Performance Considerations

```python
import time

# Timing list operations
def time_operation(operation, description):
    start = time.time()
    operation()
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")

# Comparing append vs extend
data1 = []
data2 = []

def append_method():
    for i in range(10000):
        data1.append(i)

def extend_method():
    data2.extend(range(10000))

time_operation(append_method, "Append in loop")
time_operation(extend_method, "Extend with range")

# Memory efficiency tip: Use generators for large datasets
def large_numbers():
    return [x**2 for x in range(1000000)]  # Memory intensive

def large_numbers_generator():
    return (x**2 for x in range(1000000))   # Memory efficient

print("Use generators for large datasets to save memory!")
```

## Common Patterns for AI/DS/ML

### 1. Data Preprocessing

```python
# Sample dataset
raw_data = [
    {'name': 'Alice', 'age': 25, 'score': 85.5},
    {'name': 'Bob', 'age': 30, 'score': 92.0},
    {'name': 'Charlie', 'age': 35, 'score': 78.5},
    {'name': None, 'age': 28, 'score': 88.0}  # Missing data
]

# Extract features
ages = [person['age'] for person in raw_data if person['age'] is not None]
scores = [person['score'] for person in raw_data]
names = [person['name'] for person in raw_data if person['name'] is not None]

print(f"Ages: {ages}")
print(f"Average age: {sum(ages)/len(ages):.1f}")

# Normalize scores to 0-1 range
max_score = max(scores)
min_score = min(scores)
normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
print(f"Normalized scores: {[round(s, 3) for s in normalized_scores]}")
```

### 2. Batch Processing

```python
# Simulate processing data in batches
data = list(range(1, 101))  # 100 data points
batch_size = 10

# Create batches
batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

# Process each batch
results = []
for i, batch in enumerate(batches):
    batch_result = sum(batch) / len(batch)  # Average of batch
    results.append(batch_result)
    print(f"Batch {i+1}: Average = {batch_result}")

print(f"Overall results: {results}")
```

### 3. Train/Validation/Test Split

```python
import random

# Sample dataset
dataset = [(i, f"label_{i%3}") for i in range(100)]  # 100 samples
random.shuffle(dataset)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate split indices
total_samples = len(dataset)
train_end = int(total_samples * train_ratio)
val_end = train_end + int(total_samples * val_ratio)

# Split the data
train_data = dataset[:train_end]
val_data = dataset[train_end:val_end]
test_data = dataset[val_end:]

print(f"Train: {len(train_data)} samples")
print(f"Validation: {len(val_data)} samples")
print(f"Test: {len(test_data)} samples")
```

### 4. Feature Engineering

```python
# Text processing example
texts = [
    "machine learning is awesome",
    "python programming rocks",
    "data science is fascinating",
    "artificial intelligence rocks"
]

# Tokenization
tokenized = [text.split() for text in texts]
print(f"Tokenized: {tokenized}")

# Vocabulary creation
vocabulary = list(set(word for text in tokenized for word in text))
vocabulary.sort()
print(f"Vocabulary: {vocabulary}")

# Word to index mapping
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
print(f"Word to index: {word_to_idx}")

# Convert texts to numerical features (bag of words)
def text_to_vector(text, vocab_dict):
    vector = [0] * len(vocab_dict)
    for word in text.split():
        if word in vocab_dict:
            vector[vocab_dict[word]] += 1
    return vector

vectors = [text_to_vector(text, word_to_idx) for text in texts]
print(f"Text vectors: {vectors}")
```

## Key Takeaways for AI/DS/ML

1. **Lists are fundamental**: Master them before moving to NumPy arrays
2. **List comprehensions**: Essential for data transformation and filtering
3. **Slicing**: Critical for data splitting and batch creation
4. **Built-in functions**: `sum()`, `len()`, `min()`, `max()` are frequently used
5. **Memory considerations**: Use generators for large datasets
6. **Combining operations**: Chain methods for complex data processing

## Practice Exercises

Try these exercises to reinforce your understanding:

```python
# Exercise 1: Clean and analyze survey data
survey_responses = [85, None, 92, 78, None, 88, 95, 82, None, 90]
# Task: Remove None values, calculate average, find outliers

# Exercise 2: Process time series data
daily_temperatures = [22, 25, 28, 26, 24, 27, 29, 31, 28, 25]
# Task: Calculate 3-day moving average

# Exercise 3: Prepare data for machine learning
features = [[1, 2], [3, 4], [5, 6], [7, 8]]
labels = [0, 1, 0, 1]
# Task: Combine features and labels, shuffle, split into train/test

print("Practice these exercises to master lists!")
```

---

**Remember**: Lists are the foundation of data manipulation in Python. Master these concepts, and you'll be well-prepared for AI, Data Science, and Machine Learning work!