# Python Tuples: Complete Guide for AI/DS/ML

## Table of Contents
1. [What is a Tuple?](#what-is-a-tuple)
2. [Why Use Tuples?](#why-use-tuples)
3. [How to Use Tuples](#how-to-use-tuples)
4. [Essential Tuple Operations](#essential-tuple-operations)
5. [Advanced Operations](#advanced-operations)
6. [Named Tuples](#named-tuples)
7. [Performance Considerations](#performance-considerations)
8. [Common Patterns for AI/DS/ML](#common-patterns-for-aids-ml)

## What is a Tuple?

A **tuple** in Python is an ordered collection of items that stores multiple values in a single variable. Tuples are:
- **Immutable**: Cannot be changed after creation (elements cannot be modified, added, or removed)
- **Ordered**: Items have a defined order and maintain that order
- **Allow duplicates**: The same value can appear multiple times
- **Heterogeneous**: Can store different data types in the same tuple
- **Hashable**: Can be used as dictionary keys (unlike lists)

```python
# Examples of tuples
coordinates = (10.5, 20.3, 15.8)
rgb_color = (255, 128, 0)
model_info = ("ResNet50", 25.6, "ImageNet", True)
empty_tuple = ()
single_item = (42,)  # Note the comma for single-item tuples

# Tuples without parentheses (tuple packing)
point = 1, 2, 3
name_age = "Alice", 25
```

## Why Use Tuples?

Tuples are essential in AI/DS/ML because they:

1. **Immutable data structures**: Ensure data integrity and prevent accidental modification
2. **Memory efficient**: More memory-efficient than lists for fixed data
3. **Fast access**: Faster than lists for accessing elements
4. **Dictionary keys**: Can be used as keys in dictionaries (hashable)
5. **Function returns**: Return multiple values from functions elegantly
6. **Data integrity**: Represent fixed collections like coordinates, RGB values, model parameters
7. **Unpacking**: Enable clean variable assignments and function arguments

```python
# Common use cases in AI/DS/ML
# Image dimensions (height, width, channels)
image_shape = (224, 224, 3)

# Model architecture layers
layer_config = (128, 64, 32, 1)

# Training/validation split
data_split = (0.7, 0.15, 0.15)  # train, val, test

# Coordinate points
data_points = [(1.2, 2.3), (4.5, 6.7), (8.9, 1.0)]

# Model metadata
model_metadata = ("CNN", "2024-01-15", 0.94, "classification")
```

## How to Use Tuples

### Creating Tuples

```python
# Method 1: Parentheses (most common)
dimensions = (1920, 1080)
rgb = (255, 0, 128)

# Method 2: tuple() constructor
numbers = tuple([1, 2, 3, 4, 5])
letters = tuple("hello")  # ('h', 'e', 'l', 'l', 'o')

# Method 3: Tuple packing (without parentheses)
point = 10, 20, 30
info = "model", 0.95, True

# Method 4: From other iterables
range_tuple = tuple(range(5))  # (0, 1, 2, 3, 4)
list_to_tuple = tuple([1, 2, 3])

# Special cases
empty = ()           # Empty tuple
single = (42,)       # Single element (comma required!)
single_alt = 42,     # Alternative syntax

print(f"Dimensions: {dimensions}")
print(f"Numbers: {numbers}")
print(f"Point: {point}")
print(f"Single: {single}")
```

### Accessing Elements

```python
model_info = ("ResNet50", 25.6, "ImageNet", True, (224, 224, 3))

# Indexing (0-based)
model_name = model_info[0]      # "ResNet50"
model_size = model_info[1]      # 25.6
last_element = model_info[-1]   # (224, 224, 3)

# Slicing
subset = model_info[1:4]        # (25.6, "ImageNet", True)
first_three = model_info[:3]    # ("ResNet50", 25.6, "ImageNet")
every_second = model_info[::2]  # ("ResNet50", "ImageNet", (224, 224, 3))

# Nested tuple access
input_shape = model_info[-1]    # (224, 224, 3)
height = input_shape[0]         # 224
width = input_shape[1]          # 224
channels = input_shape[2]       # 3

print(f"Model: {model_name}")
print(f"Size: {model_size} MB")
print(f"Input shape: {input_shape}")
print(f"Dimensions: {height}x{width}x{channels}")
```

### Tuple Unpacking

```python
# Basic unpacking
point = (10, 20, 30)
x, y, z = point
print(f"Coordinates: x={x}, y={y}, z={z}")

# Unpacking with different number of variables
model_data = ("CNN", 0.94, "classification")
name, accuracy, task = model_data
print(f"Model {name}: {accuracy:.2%} accuracy for {task}")

# Using underscore for unwanted values
results = ("experiment_1", 0.92, 0.08, "completed", "2024-01-15")
exp_name, accuracy, _, status, date = results
print(f"{exp_name}: {accuracy:.2%} ({status} on {date})")

# Extended unpacking with * (Python 3+)
scores = (0.85, 0.88, 0.92, 0.87, 0.90, 0.94, 0.89)
first, second, *middle, second_last, last = scores
print(f"First: {first}, Second: {second}")
print(f"Middle scores: {middle}")
print(f"Last two: {second_last}, {last}")

# Swapping variables using tuples
a, b = 10, 20
print(f"Before swap: a={a}, b={b}")
a, b = b, a  # Elegant swapping
print(f"After swap: a={a}, b={b}")
```

## Essential Tuple Operations

### 1. Basic Operations

```python
# Length
coordinates = (1.5, 2.7, 3.2)
length = len(coordinates)
print(f"Length: {length}")

# Membership testing
rgb = (255, 128, 0)
has_255 = 255 in rgb      # True
has_100 = 100 in rgb      # False
print(f"Has 255: {has_255}, Has 100: {has_100}")

# Concatenation
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
combined = tuple1 + tuple2
print(f"Combined: {combined}")

# Repetition
pattern = (0, 1)
repeated = pattern * 3
print(f"Repeated pattern: {repeated}")

# Comparison
point1 = (1, 2, 3)
point2 = (1, 2, 4)
point3 = (1, 2, 3)
print(f"point1 == point3: {point1 == point3}")
print(f"point1 < point2: {point1 < point2}")  # Lexicographic comparison
```

### 2. Finding Elements

```python
data = (1, 3, 5, 3, 7, 3, 9)

# index() - Find first occurrence
first_three = data.index(3)
print(f"First occurrence of 3 at index: {first_three}")

# index() with start and end parameters
second_three = data.index(3, first_three + 1)
print(f"Second occurrence of 3 at index: {second_three}")

# count() - Count occurrences
count_threes = data.count(3)
print(f"Number of 3s: {count_threes}")

# Finding all occurrences
def find_all_indices(tuple_data, value):
    """Find all indices of a value in tuple"""
    return [i for i, x in enumerate(tuple_data) if x == value]

all_threes = find_all_indices(data, 3)
print(f"All indices of 3: {all_threes}")
```

### 3. Converting to Other Types

```python
# Tuple to list (for modification)
original_tuple = (1, 2, 3, 4, 5)
temp_list = list(original_tuple)
temp_list.append(6)
modified_tuple = tuple(temp_list)
print(f"Original: {original_tuple}")
print(f"Modified: {modified_tuple}")

# Tuple to set (remove duplicates)
tuple_with_duplicates = (1, 2, 2, 3, 3, 3, 4)
unique_values = tuple(set(tuple_with_duplicates))
print(f"With duplicates: {tuple_with_duplicates}")
print(f"Unique values: {unique_values}")

# Tuple to dictionary (with keys)
values = (10, 20, 30)
keys = ("a", "b", "c")
dictionary = dict(zip(keys, values))
print(f"Dictionary: {dictionary}")
```

### 4. Iteration Patterns

```python
data_points = ((1, 2), (3, 4), (5, 6), (7, 8))

# Simple iteration
print("Data points:")
for point in data_points:
    print(f"Point: {point}")

# Iteration with unpacking
print("\nCoordinates:")
for x, y in data_points:
    print(f"x={x}, y={y}")

# Iteration with enumerate
print("\nIndexed points:")
for i, (x, y) in enumerate(data_points):
    print(f"Point {i}: ({x}, {y})")

# Iteration over multiple tuples simultaneously
x_coords = (1, 3, 5, 7)
y_coords = (2, 4, 6, 8)
labels = ("A", "B", "C", "D")

print("\nCombined data:")
for x, y, label in zip(x_coords, y_coords, labels):
    print(f"Point {label}: ({x}, {y})")
```

## Advanced Operations

### Working with Nested Tuples

```python
# Matrix representation using nested tuples
matrix = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9)
)

# Access matrix elements
element = matrix[1][2]  # Row 1, Column 2 = 6
print(f"Element at [1][2]: {element}")

# Transpose matrix using nested comprehension
transposed = tuple(zip(*matrix))
print(f"Original matrix: {matrix}")
print(f"Transposed: {transposed}")

# Flatten nested tuples
def flatten_tuple(nested_tuple):
    """Flatten a nested tuple structure"""
    result = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            result.extend(flatten_tuple(item))
        else:
            result.append(item)
    return tuple(result)

nested = ((1, 2), (3, (4, 5)), 6)
flattened = flatten_tuple(nested)
print(f"Nested: {nested}")
print(f"Flattened: {flattened}")
```

### Tuple Comprehensions (Generator Expressions)

```python
# Note: There's no tuple comprehension, but you can use generator expressions
numbers = (1, 2, 3, 4, 5)

# Convert generator expression to tuple
squares = tuple(x**2 for x in numbers)
print(f"Squares: {squares}")

# Filter and transform
even_squares = tuple(x**2 for x in numbers if x % 2 == 0)
print(f"Even squares: {even_squares}")

# Working with coordinates
points = ((1, 2), (3, 4), (5, 6))
distances_from_origin = tuple(
    (x**2 + y**2)**0.5 for x, y in points
)
print(f"Distances from origin: {distances_from_origin}")

# Complex transformations
data = ((1, 'a'), (2, 'b'), (3, 'c'))
formatted = tuple(f"{num}:{letter}" for num, letter in data)
print(f"Formatted: {formatted}")
```

## Named Tuples

Named tuples provide a way to create tuple subclasses with named fields:

```python
from collections import namedtuple

# Define named tuple types
Point = namedtuple('Point', ['x', 'y', 'z'])
ModelResult = namedtuple('ModelResult', ['accuracy', 'precision', 'recall', 'f1_score'])
DataSample = namedtuple('DataSample', ['features', 'label', 'metadata'])

# Create named tuple instances
point1 = Point(1.5, 2.7, 3.2)
point2 = Point(x=4.1, y=5.3, z=6.8)

result = ModelResult(0.94, 0.92, 0.96, 0.94)

# Access by name or index
print(f"Point coordinates: x={point1.x}, y={point1.y}, z={point1.z}")
print(f"Point by index: {point1[0]}, {point1[1]}, {point1[2]}")
print(f"Model accuracy: {result.accuracy}")

# Named tuples are still tuples
print(f"Is Point a tuple? {isinstance(point1, tuple)}")
print(f"Point length: {len(point1)}")

# Useful methods
print(f"Point fields: {point1._fields}")
print(f"Point as dict: {point1._asdict()}")

# Create from iterable
coords = [7.1, 8.2, 9.3]
point3 = Point._make(coords)
print(f"Point from list: {point3}")

# Replace values (returns new instance)
point4 = point1._replace(z=10.0)
print(f"Original: {point1}")
print(f"Modified: {point4}")
```

### Custom Named Tuples for ML

```python
# Define ML-specific named tuples
Experiment = namedtuple('Experiment', [
    'name', 'model_type', 'dataset', 'hyperparams', 'results', 'timestamp'
])

Hyperparams = namedtuple('Hyperparams', [
    'learning_rate', 'batch_size', 'epochs', 'optimizer'
])

TrainingResult = namedtuple('TrainingResult', [
    'train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'duration'
])

# Create experiment record
hyperparams = Hyperparams(0.001, 32, 100, 'adam')
results = TrainingResult(0.95, 0.92, 0.05, 0.08, 3600)

experiment = Experiment(
    name="exp_001",
    model_type="CNN",
    dataset="CIFAR-10",
    hyperparams=hyperparams,
    results=results,
    timestamp="2024-01-15T10:30:00"
)

print(f"Experiment: {experiment.name}")
print(f"Learning rate: {experiment.hyperparams.learning_rate}")
print(f"Validation accuracy: {experiment.results.val_accuracy}")

# Easy serialization
experiment_dict = experiment._asdict()
experiment_dict['hyperparams'] = experiment.hyperparams._asdict()
experiment_dict['results'] = experiment.results._asdict()
print(f"Serializable dict: {experiment_dict}")
```

## Performance Considerations

```python
import time
import sys

# Memory comparison
list_data = [i for i in range(1000)]
tuple_data = tuple(i for i in range(1000))

print(f"List memory usage: {sys.getsizeof(list_data)} bytes")
print(f"Tuple memory usage: {sys.getsizeof(tuple_data)} bytes")
print(f"Memory savings: {sys.getsizeof(list_data) - sys.getsizeof(tuple_data)} bytes")

# Access time comparison
def time_operation(operation, description, iterations=100000):
    start = time.time()
    for _ in range(iterations):
        operation()
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")

# Access performance
def list_access():
    return list_data[500]

def tuple_access():
    return tuple_data[500]

print("\nAccess Performance:")
time_operation(list_access, "List access")
time_operation(tuple_access, "Tuple access")

# Creation performance
def create_list():
    return [1, 2, 3, 4, 5]

def create_tuple():
    return (1, 2, 3, 4, 5)

print("\nCreation Performance:")
time_operation(create_list, "List creation")
time_operation(create_tuple, "Tuple creation")

print("\nKey Performance Points:")
print("1. Tuples are faster for access and iteration")
print("2. Tuples use less memory than lists")
print("3. Tuples are faster to create for small collections")
print("4. Tuples can be used as dictionary keys")
```

## Common Patterns for AI/DS/ML

### 1. Configuration Management

```python
# Immutable configuration using named tuples
ModelConfig = namedtuple('ModelConfig', [
    'architecture', 'input_shape', 'num_classes', 'learning_rate',
    'batch_size', 'epochs', 'optimizer', 'loss_function'
])

# Create different configurations
config_v1 = ModelConfig(
    architecture='resnet50',
    input_shape=(224, 224, 3),
    num_classes=10,
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    optimizer='adam',
    loss_function='categorical_crossentropy'
)

config_v2 = config_v1._replace(learning_rate=0.01, batch_size=64)

print(f"Config v1 learning rate: {config_v1.learning_rate}")
print(f"Config v2 learning rate: {config_v2.learning_rate}")

# Configuration validation
def validate_config(config):
    """Validate model configuration"""
    errors = []
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    if config.epochs <= 0:
        errors.append("Epochs must be positive")
    return errors

errors = validate_config(config_v1)
if errors:
    print(f"Configuration errors: {errors}")
else:
    print("Configuration is valid")
```

### 2. Data Point Representation

```python
# Represent data samples as tuples
DataPoint = namedtuple('DataPoint', ['features', 'label', 'sample_id'])

# Create dataset
dataset = [
    DataPoint(features=(1.2, 0.8, 0.3), label='positive', sample_id='sample_001'),
    DataPoint(features=(2.1, 1.5, 0.7), label='negative', sample_id='sample_002'),
    DataPoint(features=(0.9, 0.4, 0.2), label='positive', sample_id='sample_003'),
    DataPoint(features=(1.8, 1.2, 0.5), label='negative', sample_id='sample_004')
]

# Extract features and labels
features = [point.features for point in dataset]
labels = [point.label for point in dataset]

print(f"Features: {features}")
print(f"Labels: {labels}")

# Filter by label
positive_samples = [point for point in dataset if point.label == 'positive']
print(f"Positive samples: {len(positive_samples)}")

# Create train/test split preserving immutability
def split_dataset(data, train_ratio=0.8):
    """Split dataset into train and test sets"""
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

train_data, test_data = split_dataset(dataset)
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
```

### 3. Model Architecture Definition

```python
# Define neural network layers as tuples
Layer = namedtuple('Layer', ['type', 'units', 'activation', 'dropout'])

# Define model architecture
model_architecture = (
    Layer('dense', 128, 'relu', 0.2),
    Layer('dense', 64, 'relu', 0.3),
    Layer('dense', 32, 'relu', 0.2),
    Layer('dense', 1, 'sigmoid', 0.0)
)

print("Model Architecture:")
for i, layer in enumerate(model_architecture):
    print(f"Layer {i+1}: {layer.type}({layer.units}) - {layer.activation}")
    if layer.dropout > 0:
        print(f"  Dropout: {layer.dropout}")

# Calculate total parameters (simplified)
def calculate_params(architecture, input_size):
    """Calculate approximate number of parameters"""
    total_params = 0
    prev_units = input_size
    
    for layer in architecture:
        if layer.type == 'dense':
            # weights + biases
            layer_params = (prev_units * layer.units) + layer.units
            total_params += layer_params
            prev_units = layer.units
            print(f"Layer {layer.type}({layer.units}): {layer_params:,} parameters")
    
    return total_params

input_features = 784  # Example: 28x28 flattened image
total_parameters = calculate_params(model_architecture, input_features)
print(f"\nTotal parameters: {total_parameters:,}")
```

### 4. Coordinate and Geometric Operations

```python
# 2D and 3D point operations
Point2D = namedtuple('Point2D', ['x', 'y'])
Point3D = namedtuple('Point3D', ['x', 'y', 'z'])

# Create points
points_2d = [
    Point2D(1, 2), Point2D(3, 4), Point2D(5, 6), Point2D(7, 8)
]

points_3d = [
    Point3D(1, 2, 3), Point3D(4, 5, 6), Point3D(7, 8, 9)
]

# Distance calculations
def euclidean_distance_2d(p1, p2):
    """Calculate Euclidean distance between two 2D points"""
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def euclidean_distance_3d(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5

# Calculate distances
origin_2d = Point2D(0, 0)
distances_2d = [euclidean_distance_2d(origin_2d, point) for point in points_2d]
print(f"2D distances from origin: {[round(d, 2) for d in distances_2d]}")

origin_3d = Point3D(0, 0, 0)
distances_3d = [euclidean_distance_3d(origin_3d, point) for point in points_3d]
print(f"3D distances from origin: {[round(d, 2) for d in distances_3d]}")

# Bounding box calculation
def calculate_bounding_box(points):
    """Calculate bounding box for 2D points"""
    if not points:
        return None
    
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    
    BoundingBox = namedtuple('BoundingBox', ['min_x', 'min_y', 'max_x', 'max_y'])
    return BoundingBox(min_x, min_y, max_x, max_y)

bbox = calculate_bounding_box(points_2d)
print(f"Bounding box: {bbox}")
```

### 5. Time Series and Sequence Data

```python
# Time series data points
TimeSeriesPoint = namedtuple('TimeSeriesPoint', ['timestamp', 'value', 'metadata'])

# Create time series data
time_series = [
    TimeSeriesPoint('2024-01-01', 100.5, {'sensor': 'A'}),
    TimeSeriesPoint('2024-01-02', 102.3, {'sensor': 'A'}),
    TimeSeriesPoint('2024-01-03', 98.7, {'sensor': 'A'}),
    TimeSeriesPoint('2024-01-04', 105.2, {'sensor': 'A'}),
    TimeSeriesPoint('2024-01-05', 99.8, {'sensor': 'A'})
]

# Extract values for analysis
values = [point.value for point in time_series]
timestamps = [point.timestamp for point in time_series]

print(f"Time series values: {values}")
print(f"Average value: {sum(values) / len(values):.2f}")

# Window-based operations
def sliding_window(data, window_size):
    """Create sliding windows of data"""
    windows = []
    for i in range(len(data) - window_size + 1):
        window = tuple(data[i:i + window_size])
        windows.append(window)
    return windows

# Create 3-point sliding windows
windows = sliding_window(values, 3)
print(f"Sliding windows (size 3): {windows}")

# Moving average calculation
moving_averages = [sum(window) / len(window) for window in windows]
print(f"Moving averages: {[round(avg, 2) for avg in moving_averages]}")
```

### 6. Model Evaluation and Metrics

```python
# Evaluation results using named tuples
ClassificationResult = namedtuple('ClassificationResult', [
    'true_positive', 'false_positive', 'true_negative', 'false_negative'
])

ModelMetrics = namedtuple('ModelMetrics', [
    'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'
])

# Create confusion matrix result
confusion_result = ClassificationResult(
    true_positive=85,
    false_positive=12,
    true_negative=78,
    false_negative=15
)

# Calculate metrics
def calculate_metrics(result):
    """Calculate classification metrics from confusion matrix"""
    tp, fp, tn, fn = result
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc_roc=0.92  # Example AUC-ROC value
    )

metrics = calculate_metrics(confusion_result)
print(f"Model Performance:")
print(f"  Accuracy: {metrics.accuracy:.3f}")
print(f"  Precision: {metrics.precision:.3f}")
print(f"  Recall: {metrics.recall:.3f}")
print(f"  F1-Score: {metrics.f1_score:.3f}")
print(f"  AUC-ROC: {metrics.auc_roc:.3f}")

# Compare multiple models
models_results = [
    ("Model_A", ModelMetrics(0.92, 0.90, 0.94, 0.92, 0.88)),
    ("Model_B", ModelMetrics(0.94, 0.92, 0.96, 0.94, 0.91)),
    ("Model_C", ModelMetrics(0.91, 0.89, 0.93, 0.91, 0.87))
]

# Find best model by F1-score
best_model = max(models_results, key=lambda x: x[1].f1_score)
print(f"\nBest model by F1-score: {best_model[0]} (F1: {best_model[1].f1_score:.3f})")
```

## Key Takeaways for AI/DS/ML

1. **Immutability**: Use tuples for data that shouldn't change (coordinates, configurations)
2. **Memory efficiency**: Tuples use less memory than lists for fixed-size data
3. **Hashable**: Tuples can be dictionary keys and set elements
4. **Named tuples**: Provide structure and readability for complex data
5. **Function returns**: Return multiple values elegantly
6. **Data integrity**: Prevent accidental data modification in critical applications
7. **Performance**: Faster access and iteration compared to lists

## Practice Exercises

Try these exercises to reinforce your understanding:

```python
# Exercise 1: Image processing coordinates
image_coordinates = [(10, 20), (30, 40), (50, 60), (70, 80)]
# Task: Calculate center point, bounding box, and distances between points

# Exercise 2: Model comparison
model_results = [
    ("CNN", 0.92, 0.89, 0.95),      # name, accuracy, precision, recall
    ("RNN", 0.88, 0.86, 0.91),
    ("Transformer", 0.95, 0.93, 0.97)
]
# Task: Create named tuples, calculate F1-scores, find best model

# Exercise 3: Time series analysis
sensor_data = [
    ("2024-01-01", 25.5), ("2024-01-02", 26.3), ("2024-01-03", 24.8),
    ("2024-01-04", 27.1), ("2024-01-05", 25.9), ("2024-01-06", 26.7)
]
# Task: Calculate moving averages, detect anomalies, create time windows

# Exercise 4: Neural network layer definition
layers_config = [
    ("conv2d", 32, "relu"), ("maxpool", 2, None), 
    ("conv2d", 64, "relu"), ("flatten", None, None), 
    ("dense", 10, "softmax")
]
# Task: Create named tuples for layers, calculate output shapes

print("Practice these exercises to master tuples!")
```

---

**Remember**: Tuples