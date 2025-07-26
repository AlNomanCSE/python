# Python Dictionaries: Complete Guide for AI/DS/ML

## Table of Contents
1. [What is a Dictionary?](#what-is-a-dictionary)
2. [Why Use Dictionaries?](#why-use-dictionaries)
3. [How to Use Dictionaries](#how-to-use-dictionaries)
4. [Essential Dictionary Methods](#essential-dictionary-methods)
5. [Advanced Operations](#advanced-operations)
6. [Dictionary Comprehensions](#dictionary-comprehensions)
7. [Performance Considerations](#performance-considerations)
8. [Common Patterns for AI/DS/ML](#common-patterns-for-aids-ml)

## What is a Dictionary?

A **dictionary** in Python is an unordered collection of key-value pairs that stores data in a mapping format. Dictionaries are:
- **Mutable**: You can change, add, or remove items after creation
- **Unordered**: Items don't have a defined order (though insertion order is preserved in Python 3.7+)
- **Key-unique**: Each key can appear only once, but values can be duplicated
- **Heterogeneous**: Keys and values can be of different data types

```python
# Examples of dictionaries
student = {"name": "Alice", "age": 22, "grade": "A"}
coordinates = {"x": 10.5, "y": 20.3, "z": 15.8}
model_config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
}
empty_dict = {}
```

## Why Use Dictionaries?

Dictionaries are essential in AI/DS/ML because they:

1. **Store structured data**: Represent records, configurations, and metadata
2. **Fast lookups**: O(1) average time complexity for key-based access
3. **Model parameters**: Store hyperparameters, weights, and model configurations
4. **Data mapping**: Create lookup tables, encodings, and transformations
5. **Feature engineering**: Store feature names, statistics, and transformations
6. **JSON compatibility**: Seamlessly work with APIs and data interchange formats

```python
# Common use cases in AI/DS/ML
# Dataset record
sample = {
    "features": [1.2, 0.8, 0.3, 2.1],
    "label": "positive",
    "metadata": {"source": "sensor_A", "timestamp": "2024-01-15"}
}

# Model evaluation metrics
metrics = {
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.96,
    "f1_score": 0.94
}

# Feature statistics
feature_stats = {
    "mean": 125.6,
    "std": 23.4,
    "min": 45.2,
    "max": 234.7
}
```

## How to Use Dictionaries

### Creating Dictionaries

```python
# Method 1: Curly braces
person = {"name": "Bob", "age": 30, "city": "New York"}

# Method 2: dict() constructor
coordinates = dict(x=10, y=20, z=30)

# Method 3: From key-value pairs
pairs = [("a", 1), ("b", 2), ("c", 3)]
letter_numbers = dict(pairs)

# Method 4: Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Method 5: From lists (zip)
keys = ["name", "age", "score"]
values = ["Charlie", 25, 88.5]
student_dict = dict(zip(keys, values))

print(f"Person: {person}")
print(f"Coordinates: {coordinates}")
print(f"Squares: {squares}")
print(f"Student: {student_dict}")
```

### Accessing Elements

```python
data = {
    "model": "neural_network",
    "accuracy": 0.95,
    "layers": [64, 32, 16],
    "config": {"learning_rate": 0.001}
}

# Direct key access
model_type = data["model"]  # "neural_network"
accuracy = data["accuracy"]  # 0.95

# Using get() method (safer - returns None if key doesn't exist)
precision = data.get("precision")  # None
precision_default = data.get("precision", 0.0)  # 0.0

# Nested access
learning_rate = data["config"]["learning_rate"]  # 0.001

# Check if key exists
has_precision = "precision" in data  # False
has_accuracy = "accuracy" in data    # True

print(f"Model: {model_type}")
print(f"Accuracy: {accuracy}")
print(f"Precision (default): {precision_default}")
print(f"Has precision: {has_precision}")
```

## Essential Dictionary Methods

### 1. Adding and Updating Elements

```python
# Starting dictionary
config = {"batch_size": 32, "learning_rate": 0.001}

# Add new key-value pair
config["epochs"] = 100
print(f"After adding epochs: {config}")

# Update existing value
config["learning_rate"] = 0.01
print(f"After updating learning_rate: {config}")

# update() - Add multiple key-value pairs
config.update({"optimizer": "adam", "dropout": 0.2})
print(f"After update(): {config}")

# update() with keyword arguments
config.update(momentum=0.9, weight_decay=1e-4)
print(f"After update with kwargs: {config}")

# setdefault() - Add key only if it doesn't exist
config.setdefault("patience", 10)  # Adds patience: 10
config.setdefault("epochs", 200)   # Doesn't change epochs (already exists)
print(f"After setdefault(): {config}")
```

### 2. Removing Elements

```python
data = {
    "name": "experiment_1",
    "model": "cnn",
    "accuracy": 0.92,
    "loss": 0.08,
    "optimizer": "sgd"
}

# pop() - Remove and return value
accuracy = data.pop("accuracy")
print(f"Removed accuracy: {accuracy}, Remaining: {data}")

# pop() with default value
precision = data.pop("precision", "Not found")
print(f"Precision (with default): {precision}")

# popitem() - Remove and return last inserted key-value pair
last_item = data.popitem()
print(f"Last item removed: {last_item}, Remaining: {data}")

# del - Delete specific key
del data["loss"]
print(f"After del loss: {data}")

# clear() - Remove all elements
data.clear()
print(f"After clear(): {data}")
```

### 3. Dictionary Views and Iteration

```python
model_metrics = {
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.96,
    "f1_score": 0.94
}

# Get keys, values, and items
keys = model_metrics.keys()
values = model_metrics.values()
items = model_metrics.items()

print(f"Keys: {list(keys)}")
print(f"Values: {list(values)}")
print(f"Items: {list(items)}")

# Iteration patterns
print("\n--- Iteration Examples ---")

# Iterate over keys
for metric in model_metrics:
    print(f"Metric: {metric}")

# Iterate over values
for value in model_metrics.values():
    print(f"Value: {value}")

# Iterate over key-value pairs
for metric, value in model_metrics.items():
    print(f"{metric}: {value}")

# Iterate with enumerate for indexing
for i, (metric, value) in enumerate(model_metrics.items()):
    print(f"{i}: {metric} = {value}")
```

### 4. Copying Dictionaries

```python
original = {
    "model_name": "resnet50",
    "params": {"layers": 50, "weights": "imagenet"},
    "metrics": [0.92, 0.88, 0.95]
}

# Shallow copy
shallow_copy1 = original.copy()
shallow_copy2 = dict(original)

# Deep copy (for nested structures)
import copy
deep_copy = copy.deepcopy(original)

# Demonstrate difference
original["params"]["layers"] = 101
original["metrics"].append(0.97)

print(f"Original: {original}")
print(f"Shallow copy: {shallow_copy1}")  # Nested objects affected
print(f"Deep copy: {deep_copy}")         # Completely independent
```

## Advanced Operations

### Dictionary Merging (Python 3.9+)

```python
# Model configurations
base_config = {"batch_size": 32, "learning_rate": 0.001}
experiment_config = {"epochs": 100, "dropout": 0.2}
override_config = {"learning_rate": 0.01, "batch_size": 64}

# Method 1: Union operator (Python 3.9+)
# final_config = base_config | experiment_config | override_config

# Method 2: Unpacking (works in older versions)
final_config = {**base_config, **experiment_config, **override_config}

print(f"Final config: {final_config}")

# Method 3: Using update() for in-place merging
config = base_config.copy()
config.update(experiment_config)
config.update(override_config)
print(f"Updated config: {config}")
```

### Nested Dictionary Operations

```python
# Complex model configuration
model_structure = {
    "input_layer": {"type": "dense", "units": 128, "activation": "relu"},
    "hidden_layers": [
        {"type": "dense", "units": 64, "activation": "relu", "dropout": 0.2},
        {"type": "dense", "units": 32, "activation": "relu", "dropout": 0.3}
    ],
    "output_layer": {"type": "dense", "units": 1, "activation": "sigmoid"}
}

# Access nested values
input_units = model_structure["input_layer"]["units"]
first_hidden_dropout = model_structure["hidden_layers"][0]["dropout"]

print(f"Input units: {input_units}")
print(f"First hidden dropout: {first_hidden_dropout}")

# Safe nested access function
def safe_get(dictionary, keys, default=None):
    """Safely get nested dictionary values"""
    for key in keys:
        if isinstance(dictionary, dict) and key in dictionary:
            dictionary = dictionary[key]
        else:
            return default
    return dictionary

# Example usage
units = safe_get(model_structure, ["input_layer", "units"], 0)
missing = safe_get(model_structure, ["input_layer", "bias"], "not_found")
print(f"Safe get units: {units}")
print(f"Safe get missing: {missing}")
```

## Dictionary Comprehensions

Dictionary comprehensions are powerful for data transformation in AI/DS/ML:

```python
# Basic syntax: {key_expr: value_expr for item in iterable if condition}

# Example 1: Square numbers
numbers = [1, 2, 3, 4, 5]
squares_dict = {num: num**2 for num in numbers}
print(f"Squares: {squares_dict}")

# Example 2: Filter and transform
words = ["apple", "banana", "cherry", "date"]
word_lengths = {word: len(word) for word in words if len(word) > 4}
print(f"Long words: {word_lengths}")

# Example 3: Transform existing dictionary
celsius_temps = {"morning": 20, "noon": 30, "evening": 25}
fahrenheit_temps = {time: (temp * 9/5) + 32 for time, temp in celsius_temps.items()}
print(f"Fahrenheit: {fahrenheit_temps}")

# Example 4: Create lookup tables
categories = ["cat", "dog", "bird", "fish"]
category_to_id = {category: idx for idx, category in enumerate(categories)}
id_to_category = {idx: category for category, idx in category_to_id.items()}
print(f"Category to ID: {category_to_id}")
print(f"ID to Category: {id_to_category}")

# Example 5: Group data
data_points = [
    ("A", 10), ("B", 20), ("A", 15), ("C", 30), ("B", 25), ("A", 5)
]
grouped = {}
for category, value in data_points:
    if category not in grouped:
        grouped[category] = []
    grouped[category].append(value)

# Or using dictionary comprehension with grouping logic
from collections import defaultdict
grouped_dict = defaultdict(list)
for category, value in data_points:
    grouped_dict[category].append(value)

print(f"Grouped data: {dict(grouped_dict)}")
```

## Performance Considerations

```python
import time
from collections import defaultdict, Counter

# Timing dictionary operations
def time_operation(operation, description):
    start = time.time()
    result = operation()
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")
    return result

# Dictionary vs List lookup performance
large_dict = {i: f"value_{i}" for i in range(100000)}
large_list = [(i, f"value_{i}") for i in range(100000)]

def dict_lookup():
    return large_dict.get(50000)

def list_lookup():
    for key, value in large_list:
        if key == 50000:
            return value

time_operation(dict_lookup, "Dictionary lookup")
time_operation(list_lookup, "List lookup")

# Memory considerations
print("\n--- Memory Tips ---")
print("1. Use defaultdict for automatic key initialization")
print("2. Use Counter for counting operations")
print("3. Consider dict vs list based on access patterns")

# defaultdict example
dd = defaultdict(list)
for i in range(5):
    dd[i % 3].append(i)
print(f"defaultdict result: {dict(dd)}")

# Counter example
text = "hello world hello python world"
word_counts = Counter(text.split())
print(f"Word counts: {word_counts}")
```

## Common Patterns for AI/DS/ML

### 1. Configuration Management

```python
# Model configuration with validation
class ModelConfig:
    def __init__(self, **kwargs):
        self.defaults = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "loss_function": "categorical_crossentropy"
        }
        
        # Merge with provided config
        self.config = {**self.defaults, **kwargs}
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config["learning_rate"] <= 0:
            raise ValueError("Learning rate must be positive")
        if self.config["batch_size"] <= 0:
            raise ValueError("Batch size must be positive")
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        self.config.update(kwargs)
        self._validate_config()

# Usage
config = ModelConfig(learning_rate=0.01, epochs=50)
print(f"Config: {config.config}")

# Experiment tracking
experiments = {}
experiments["exp_001"] = {
    "config": {"lr": 0.001, "batch_size": 32},
    "results": {"accuracy": 0.92, "loss": 0.08},
    "timestamp": "2024-01-15"
}
experiments["exp_002"] = {
    "config": {"lr": 0.01, "batch_size": 64},
    "results": {"accuracy": 0.94, "loss": 0.06},
    "timestamp": "2024-01-16"
}

print(f"Experiments: {len(experiments)}")
```

### 2. Feature Engineering and Encoding

```python
# Categorical encoding
categories = ["cat", "dog", "bird", "cat", "dog", "fish", "bird", "cat"]

# One-hot encoding preparation
unique_categories = list(set(categories))
category_to_index = {cat: idx for idx, cat in enumerate(unique_categories)}
index_to_category = {idx: cat for cat, idx in category_to_index.items()}

print(f"Category to index: {category_to_index}")
print(f"Index to category: {index_to_category}")

# Label encoding
def create_label_encoder(labels):
    unique_labels = list(set(labels))
    encoder = {label: idx for idx, label in enumerate(unique_labels)}
    decoder = {idx: label for label, idx in encoder.items()}
    return encoder, decoder

encoder, decoder = create_label_encoder(categories)
encoded_labels = [encoder[cat] for cat in categories]
print(f"Encoded: {encoded_labels}")
print(f"Decoded: {[decoder[idx] for idx in encoded_labels]}")

# Vocabulary building for text processing
documents = [
    "machine learning is great",
    "deep learning rocks",
    "python programming is awesome",
    "data science with python"
]

# Build vocabulary
vocab = {}
word_id = 0
for doc in documents:
    for word in doc.split():
        if word not in vocab:
            vocab[word] = word_id
            word_id += 1

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocab: {dict(list(vocab.items())[:5])}")

# Reverse vocabulary
reverse_vocab = {idx: word for word, idx in vocab.items()}
```

### 3. Data Statistics and Analysis

```python
# Dataset statistics
dataset = [
    {"age": 25, "income": 50000, "category": "A"},
    {"age": 30, "income": 60000, "category": "B"},
    {"age": 35, "income": 70000, "category": "A"},
    {"age": 28, "income": 55000, "category": "C"},
    {"age": 32, "income": 65000, "category": "B"}
]

# Calculate statistics by feature
def calculate_stats(data, numeric_features):
    stats = {}
    for feature in numeric_features:
        values = [record[feature] for record in data]
        stats[feature] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }
    return stats

numeric_features = ["age", "income"]
feature_stats = calculate_stats(dataset, numeric_features)
print("Feature Statistics:")
for feature, stats in feature_stats.items():
    print(f"  {feature}: {stats}")

# Categorical analysis
def analyze_categorical(data, categorical_features):
    analysis = {}
    for feature in categorical_features:
        categories = [record[feature] for record in data]
        analysis[feature] = {
            "unique_values": list(set(categories)),
            "counts": {cat: categories.count(cat) for cat in set(categories)},
            "most_frequent": max(set(categories), key=categories.count)
        }
    return analysis

categorical_features = ["category"]
categorical_analysis = analyze_categorical(dataset, categorical_features)
print("\nCategorical Analysis:")
for feature, analysis in categorical_analysis.items():
    print(f"  {feature}: {analysis}")
```

### 4. Model Performance Tracking

```python
# Model performance metrics tracking
class PerformanceTracker:
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def add_epoch_metrics(self, epoch, **metrics):
        """Add metrics for a specific epoch"""
        epoch_data = {"epoch": epoch, **metrics}
        self.history.append(epoch_data)
        
        # Update best metrics
        for metric, value in metrics.items():
            if metric not in self.metrics:
                self.metrics[metric] = {"best": value, "worst": value, "current": value}
            else:
                current_best = self.metrics[metric]["best"]
                current_worst = self.metrics[metric]["worst"]
                
                # Assume higher is better (modify logic as needed)
                if value > current_best:
                    self.metrics[metric]["best"] = value
                if value < current_worst:
                    self.metrics[metric]["worst"] = value
                self.metrics[metric]["current"] = value
    
    def get_best_epoch(self, metric="accuracy"):
        """Get epoch with best performance for given metric"""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x.get(metric, 0))
    
    def get_summary(self):
        """Get performance summary"""
        return {
            "total_epochs": len(self.history),
            "metrics_summary": self.metrics,
            "best_epoch": self.get_best_epoch()
        }

# Usage example
tracker = PerformanceTracker()

# Simulate training epochs
training_data = [
    (1, {"accuracy": 0.85, "loss": 0.45, "val_accuracy": 0.82}),
    (2, {"accuracy": 0.88, "loss": 0.38, "val_accuracy": 0.86}),
    (3, {"accuracy": 0.91, "loss": 0.32, "val_accuracy": 0.89}),
    (4, {"accuracy": 0.93, "loss": 0.28, "val_accuracy": 0.91}),
    (5, {"accuracy": 0.94, "loss": 0.25, "val_accuracy": 0.92})
]

for epoch, metrics in training_data:
    tracker.add_epoch_metrics(epoch, **metrics)

summary = tracker.get_summary()
print("Training Summary:")
print(f"Total epochs: {summary['total_epochs']}")
print(f"Best epoch: {summary['best_epoch']}")
print("Metrics summary:")
for metric, stats in summary['metrics_summary'].items():
    print(f"  {metric}: Best={stats['best']:.3f}, Current={stats['current']:.3f}")
```

### 5. Hyperparameter Grid Search

```python
# Hyperparameter combinations
def generate_param_grid(param_dict):
    """Generate all combinations of hyperparameters"""
    from itertools import product
    
    keys = param_dict.keys()
    values = param_dict.values()
    
    combinations = []
    for combination in product(*values):
        combinations.append(dict(zip(keys, combination)))
    
    return combinations

# Define parameter grid
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64],
    "dropout": [0.2, 0.3, 0.5]
}

# Generate all combinations
param_combinations = generate_param_grid(param_grid)
print(f"Total combinations: {len(param_combinations)}")
print("First 3 combinations:")
for i, params in enumerate(param_combinations[:3]):
    print(f"  {i+1}: {params}")

# Results tracking
results = {}
for i, params in enumerate(param_combinations[:5]):  # Simulate first 5
    # Simulate model training and evaluation
    # In practice, you would train your model here
    simulated_accuracy = 0.8 + (i * 0.02) + (params["learning_rate"] * 0.1)
    
    results[i] = {
        "params": params,
        "accuracy": simulated_accuracy,
        "loss": 1 - simulated_accuracy
    }

# Find best parameters
best_result = max(results.values(), key=lambda x: x["accuracy"])
print(f"\nBest parameters: {best_result['params']}")
print(f"Best accuracy: {best_result['accuracy']:.3f}")
```

## Key Takeaways for AI/DS/ML

1. **Fast lookups**: Use dictionaries for O(1) key-based access
2. **Configuration management**: Store and validate model parameters
3. **Feature engineering**: Create encodings, mappings, and transformations
4. **Data organization**: Structure datasets and metadata efficiently
5. **Performance tracking**: Monitor metrics and experiment results
6. **JSON compatibility**: Seamlessly work with APIs and data formats

## Practice Exercises

Try these exercises to reinforce your understanding:

```python
# Exercise 1: Feature statistics calculator
dataset = [
    {"feature1": 1.2, "feature2": 0.8, "label": "A"},
    {"feature1": 2.1, "feature2": 1.5, "label": "B"},
    {"feature1": 0.9, "feature2": 0.4, "label": "A"}
]
# Task: Calculate mean, std, min, max for each numeric feature

# Exercise 2: Model comparison tracker
models = ["model_A", "model_B", "model_C"]
metrics = ["accuracy", "precision", "recall", "f1_score"]
# Task: Create a structure to track multiple metrics for multiple models

# Exercise 3: Text preprocessing pipeline
documents = ["hello world", "python programming", "machine learning"]
# Task: Build vocabulary, create word-to-id mapping, implement TF-IDF weights

# Exercise 4: Experiment configuration validator
config_template = {"learning_rate": float, "epochs": int, "batch_size": int}
user_config = {"learning_rate": "0.01", "epochs": 100, "batch_size": "32"}
# Task: Validate and convert types according to template

print("Practice these exercises to master dictionaries!")
```

---

**Remember**: Dictionaries are the backbone of data organization in Python. Master these concepts, and you'll efficiently handle configurations, features, and structured data in AI, Data Science, and Machine Learning projects!