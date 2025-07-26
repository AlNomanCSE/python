Python Dictionaries: Complete Guide for AI/DS/ML
Table of Contents

What is a Dictionary?
Why Use Dictionaries?
How to Use Dictionaries
Essential Dictionary Methods
Advanced Operations
Dictionary Comprehensions
Performance Considerations
Common Patterns for AI/DS/ML

What is a Dictionary?
A dictionary in Python is an unordered collection of key-value pairs that provides a fast way to store and retrieve data. Dictionaries are:

Mutable: You can modify, add, or remove key-value pairs
Unordered: No specific order of items (prior to Python 3.7; insertion order preserved in 3.7+)
Unique keys: Each key can appear only once
Heterogeneous: Can store different data types as values

# Examples of dictionaries
student_scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
features = {"x1": 1.2, "x2": 0.8, "label": "positive"}
nested_dict = {"person": {"name": "Alice", "age": 25}, "scores": [85, 90, 88]}
empty_dict = {}

Why Use Dictionaries?
Dictionaries are essential in AI/DS/ML because they:

Store structured data: Map features to values or samples to attributes
Fast lookups: O(1) average time complexity for accessing values
Handle metadata: Store configurations, hyperparameters, or model info
Represent sparse data: Efficient for datasets with many missing values
Flexible data manipulation: Easy to update, merge, or transform data

# Common use cases in AI/DS/ML
model_config = {
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 100
}

feature_vector = {
    "age": 30,
    "income": 50000,
    "purchases": 12
}

label_mapping = {
    0: "negative",
    1: "positive"
}

How to Use Dictionaries
Creating Dictionaries
# Method 1: Curly braces
student = {"name": "Alice", "age": 25, "grade": "A"}

# Method 2: dict() constructor
scores = dict(Alice=85, Bob=92, Charlie=78)

# Method 3: Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Method 4: From list of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
letters = dict(pairs)  # {'a': 1, 'b': 2, 'c': 3}

Accessing Elements
data = {"name": "Alice", "age": 25, "scores": [85, 90, 88]}

# Accessing values
name = data["name"]          # "Alice"
age = data.get("age")        # 25
missing = data.get("grade", "N/A")  # "N/A" (default if key doesn't exist)

# Checking if key exists
has_name = "name" in data    # True
has_grade = "grade" in data  # False

print(f"Name: {name}, Age: {age}, Missing: {missing}")

Essential Dictionary Methods
1. Adding and Updating Elements
data = {"a": 1, "b": 2}

# Add new key-value pair
data["c"] = 3
print(data)  # {'a': 1, 'b': 2, 'c': 3}

# Update existing value
data["a"] = 10
print(data)  # {'a': 10, 'b': 2, 'c': 3}

# update() - Merge another dictionary or key-value pairs
data.update({"b": 20, "d": 4})
print(data)  # {'a': 10, 'b': 20, 'c': 3, 'd': 4}

# setdefault() - Add key with default if not exists
data.setdefault("e", 5)
print(data)  # {'a': 10, 'b': 20, 'c': 3, 'd': 4, 'e': 5}

2. Removing Elements
data = {"a": 1, "b": 2, "c": 3, "d": 4}

# pop() - Remove and return value for key
value = data.pop("b")
print(f"Removed value: {value}, Dict: {data}")  # Removed value: 2, Dict: {'a': 1, 'c': 3, 'd': 4}

# popitem() - Remove and return last key-value pair
key, value = data.popitem()
print(f"Removed pair: ({key}, {value}), Dict: {data}")  # Removed pair: ('d', 4), Dict: {'a': 1, 'c': 3}

# del - Delete specific key
del data["a"]
print(data)  # {'c': 3}

# clear() - Remove all items
data.clear()
print(data)  # {}

3. Accessing Keys and Values
data = {"a": 1, "b": 2, "c": 3}

# Get all keys
keys = data.keys()      # dict_keys(['a', 'b', 'c'])
print(f"Keys: {list(keys)}")

# Get all values
values = data.values()  # dict_values([1, 2, 3])
print(f"Values: {list(values)}")

# Get all key-value pairs
items = data.items()    # dict_items([('a', 1), ('b', 2), ('c', 3)])
print(f"Items: {list(items)}")

# Length of dictionary
length = len(data)
print(f"Length: {length}")  # 3

4. Copying Dictionaries
original = {"a": 1, "b": [1, 2], "c": {"x": 10}}

# Shallow copy
shallow_copy1 = original.copy()
shallow_copy2 = dict(original)

# Deep copy (for nested structures)
import copy
deep_copy = copy.deepcopy(original)

# Demonstrate difference
original["b"][0] = 999
original["c"]["x"] = 999
print(f"Original: {original}")          # {'a': 1, 'b': [999, 2], 'c': {'x': 999}}
print(f"Shallow copy: {shallow_copy1}") # {'a': 1, 'b': [999, 2], 'c': {'x': 999}}
print(f"Deep copy: {deep_copy}")        # {'a': 1, 'b': [1, 2], 'c': {'x': 10}}

Advanced Operations
Merging Dictionaries
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}

# Method 1: update()
merged1 = dict1.copy()
merged1.update(dict2)
print(f"Merged (update): {merged1}")

# Method 2: | operator (Python 3.9+)
merged2 = dict1 | dict2
print(f"Merged (|): {merged2}")

# Method 3: Dictionary unpacking
merged3 = {**dict1, **dict2}
print(f"Merged (unpacking): {merged3}")

Statistical Operations
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}

# Basic statistics
total = sum(scores.values())
average = total / len(scores)
max_score = max(scores.values())
top_scorer = max(scores, key=scores.get)

print(f"Total: {total}, Average: {average:.1f}")
print(f"Highest score: {max_score} by {top_scorer}")

Dictionary Comprehensions
Dictionary comprehensions are powerful for data transformation:
# Basic syntax: {key_expr: value_expr for item in iterable if condition}

# Example 1: Create key-value pairs
squares = {x: x**2 for x in range(5)}
print(f"Squares: {squares}")  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Example 2: Filter and transform
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
high_scores = {name: score for name, score in scores.items() if score > 80}
print(f"High scores: {high_scores}")  # {'Alice': 85, 'Bob': 92}

# Example 3: Swap keys and values
swapped = {value: key for key, value in scores.items()}
print(f"Swapped: {swapped}")  # {85: 'Alice', 92: 'Bob', 78: 'Charlie'}

# Example 4: Nested dictionary comprehension
matrix = {f"row{i}": {f"col{j}": i*j for j in range(3)} for i in range(3)}
print("3x3 multiplication table:")
for row, values in matrix.items():
    print(f"{row}: {values}")

Performance Considerations
import time

# Timing dictionary operations
def time_operation(operation, description):
    start = time.time()
    operation()
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")

# Comparing dictionary creation methods
def create_dict_loop():
    d = {}
    for i in range(10000):
        d[i] = i**2

def create_dict_comprehension():
    d = {i: i**2 for i in range(10000)}

time_operation(create_dict_loop, "Dictionary creation with loop")
time_operation(create_dict_comprehension, "Dictionary creation with comprehension")

# Memory efficiency tip: Use defaultdict for missing keys
from collections import defaultdict
def count_words(texts):
    word_count = defaultdict(int)
    for text in texts:
        for word in text.split():
            word_count[word] += 1
    return dict(word_count)

texts = ["hello world", "hello python", "world of python"]
print(f"Word counts: {count_words(texts)}")

Common Patterns for AI/DS/ML
1. Data Preprocessing
# Sample dataset
raw_data = [
    {"id": 1, "name": "Alice", "features": [1.2, 0.8], "label": "positive"},
    {"id": 2, "name": "Bob", "features": [1.5, 0.9], "label": "negative"},
    {"id": 3, "name": None, "features": [0.7, 1.1], "label": "positive"}
]

# Create feature dictionary
feature_dict = {item["id"]: item["features"] for item in raw_data}
print(f"Feature dictionary: {feature_dict}")

# Normalize features
max_values = [max(f[i] for f in feature_dict.values()) for i in range(len(next(iter(feature_dict.values()))))]
normalized = {k: [v[i]/max_values[i] for i in range(len(v))] for k, v in feature_dict.items()}
print(f"Normalized features: {normalized}")

2. Model Configuration
# Model hyperparameters
default_config = {
    "learning_rate": 0.01,
    "batch_size": 32,
    "hidden_layers": [64, 32],
    "activation": "relu"
}

# Update for specific experiment
experiment_config = default_config.copy()
experiment_config.update({
    "learning_rate": 0.001,
    "dropout": 0.2
})

print(f"Experiment config: {experiment_config}")

# Save multiple experiment configurations
experiments = {
    f"exp_{i}": {**default_config, "learning_rate": lr}
    for i, lr in enumerate([0.01, 0.005, 0.001])
}
print(f"Experiment variations: {experiments}")

3. Feature Encoding
# One-hot encoding example
categories = ["cat", "dog", "bird", "cat", "dog"]
unique_categories = sorted(set(categories))
category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}

# Convert to one-hot vectors
one_hot = {
    cat: [1 if i == category_to_idx[cat] else 0 for i in range(len(unique_categories))]
    for cat in categories
}
print(f"One-hot encoding: {one_hot}")

# Count encoding
count_encoding = {cat: categories.count(cat) for cat in unique_categories}
print(f"Count encoding: {count_encoding}")

4. Caching Results
# Cache expensive computations
cache = {}

def expensive_computation(x):
    if x not in cache:
        cache[x] = x**2 + x**3  # Example computation
    return cache[x]

# Example usage
data = [1, 2, 1, 3, 2, 1]
results = [expensive_computation(x) for x in data]
print(f"Results: {results}")
print(f"Cache: {cache}")

Key Takeaways for AI/DS/ML

Dictionaries are key for mapping: Perfect for feature-to-value or ID-to-data mappings
Fast lookups: Use for quick data retrieval in preprocessing
Dictionary comprehensions: Efficient for data transformation
Flexible structure: Ideal for storing model configurations and metadata
Memory efficiency: Use defaultdict for automatic key handling
Merging and updating: Essential for combining datasets or configurations

Practice Exercises
Try these exercises to reinforce your understanding:
# Exercise 1: Clean and analyze student data
student_data = [
    {"id": 1, "score": 85, "grade": None},
    {"id": 2, "score": 92, "grade": "A"},
    {"id": 3, "score": None, "grade": "B"}
]
# Task: Remove None values, calculate average score, create id-to-grade mapping

# Exercise 2: Create feature dictionary
samples = [[1.2, 0.8, 0.3], [2.1, 1.5, 0.7], [0.9, 0.4, 0.2]]
# Task: Create dictionary mapping sample index to features and normalize values

# Exercise 3: Word frequency counter
texts = ["hello world", "hello python", "world of python"]
# Task: Create dictionary of word frequencies using defaultdict

# Exercise 4: Model configuration manager
base_config = {"lr": 0.01, "batch_size": 32}
# Task: Create variations of configurations with different learning rates

print("Practice these exercises to master dictionaries!")


Remember: Dictionaries are crucial for efficient data organization and retrieval in Python. Master these concepts to excel in AI, Data Science, and Machine Learning tasks!