# Python Lists Guide 📝

A complete beginner-friendly guide to understanding and using lists in Python. Learn what they are, how to use them, and master all their powerful features!

## Table of Contents
- [What is a List?](#what-is-a-list)
- [Creating Lists](#creating-lists)
- [Accessing List Elements](#accessing-list-elements)
- [Modifying Lists](#modifying-lists)
- [List Methods](#list-methods)
- [List Operations](#list-operations)
- [List Comprehensions](#list-comprehensions)
- [Advanced List Features](#advanced-list-features)
- [List vs Other Data Types](#list-vs-other-data-types)
- [Common Use Cases](#common-use-cases)
- [Practice Examples](#practice-examples)

---

## What is a List?

A **list** is one of the most versatile and commonly used data types in Python. It's a collection that can store multiple items in a single variable.

Think of lists as:
- 📋 A shopping list - you can add, remove, and change items
- 🎒 A backpack - you can put different things in different pockets
- 📚 A bookshelf - ordered collection where you can rearrange books

### Key Characteristics:
- ✅ **Ordered** - items have a defined order and maintain that order
- ✅ **Mutable** - can be changed after creation (add, remove, modify)
- ✅ **Allow duplicates** - can have the same value multiple times
- ✅ **Indexed** - can access items by their position (0, 1, 2, ...)
- ✅ **Dynamic size** - can grow and shrink as needed

---

## Creating Lists

### Method 1: Using Square Brackets []
```python
# Empty list
empty_list = []
print(empty_list)  # []

# List with numbers
numbers = [1, 2, 3, 4, 5]
print(numbers)  # [1, 2, 3, 4, 5]

# List with strings
fruits = ['apple', 'banana', 'orange']
print(fruits)  # ['apple', 'banana', 'orange']

# Mixed data types (allowed but not always recommended)
mixed = ['Alice', 25, True, 3.14]
print(mixed)  # ['Alice', 25, True, 3.14]
```

### Method 2: Using list() Constructor
```python
# From a string
letters = list('hello')
print(letters)  # ['h', 'e', 'l', 'l', 'o']

# From a tuple
numbers_tuple = (1, 2, 3, 4, 5)
numbers_list = list(numbers_tuple)
print(numbers_list)  # [1, 2, 3, 4, 5]

# From a range
range_list = list(range(1, 6))
print(range_list)  # [1, 2, 3, 4, 5]
```

### Method 3: Using List Multiplication
```python
# Create list with repeated values
zeros = [0] * 5
print(zeros)  # [0, 0, 0, 0, 0]

# Be careful with mutable objects!
# Wrong way (all sublists are the same object)
wrong = [[]] * 3
wrong[0].append(1)
print(wrong)  # [[1], [1], [1]] - all changed!

# Correct way
correct = [[] for _ in range(3)]
correct[0].append(1)
print(correct)  # [[1], [], []] - only first changed
```

---

## Accessing List Elements

### Indexing
```python
fruits = ['apple', 'banana', 'orange', 'grape', 'kiwi']

# Positive indexing (starts from 0)
print(fruits[0])   # apple (first item)
print(fruits[1])   # banana
print(fruits[4])   # kiwi (last item)

# Negative indexing (starts from -1)
print(fruits[-1])  # kiwi (last item)
print(fruits[-2])  # grape (second last)
print(fruits[-5])  # apple (first item)
```

### Slicing
```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic slicing [start:stop]
print(numbers[2:5])    # [2, 3, 4] (stop is exclusive)
print(numbers[:3])     # [0, 1, 2] (from beginning)
print(numbers[7:])     # [7, 8, 9] (to end)

# Step slicing [start:stop:step]
print(numbers[::2])    # [0, 2, 4, 6, 8] (every 2nd item)
print(numbers[1::2])   # [1, 3, 5, 7, 9] (every 2nd starting from 1)
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (reversed)

# Advanced slicing
print(numbers[2:8:2])  # [2, 4, 6] (from 2 to 8, every 2nd)
```

### Checking if Item Exists
```python
fruits = ['apple', 'banana', 'orange']

print('apple' in fruits)     # True
print('grape' in fruits)     # False
print('apple' not in fruits) # False
```

---

## Modifying Lists

### Changing Individual Items
```python
fruits = ['apple', 'banana', 'orange']
print("Original:", fruits)

# Change single item
fruits[1] = 'mango'
print("After change:", fruits)  # ['apple', 'mango', 'orange']

# Change multiple items using slicing
fruits[0:2] = ['grape', 'kiwi']
print("After slice change:", fruits)  # ['grape', 'kiwi', 'orange']
```

### Adding Items

#### append() - Add single item to end
```python
numbers = [1, 2, 3]
numbers.append(4)
print(numbers)  # [1, 2, 3, 4]

# Append can add any type
numbers.append('five')
print(numbers)  # [1, 2, 3, 4, 'five']
```

#### insert() - Add item at specific position
```python
fruits = ['apple', 'orange']
fruits.insert(1, 'banana')  # Insert at index 1
print(fruits)  # ['apple', 'banana', 'orange']

fruits.insert(0, 'grape')   # Insert at beginning
print(fruits)  # ['grape', 'apple', 'banana', 'orange']
```

#### extend() - Add multiple items
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

list1.extend(list2)
print(list1)  # [1, 2, 3, 4, 5, 6]

# Or extend with any iterable
list1.extend('abc')
print(list1)  # [1, 2, 3, 4, 5, 6, 'a', 'b', 'c']
```

### Removing Items

#### remove() - Remove first occurrence of value
```python
fruits = ['apple', 'banana', 'orange', 'banana']
fruits.remove('banana')  # Removes first 'banana'
print(fruits)  # ['apple', 'orange', 'banana']
```

#### pop() - Remove and return item by index
```python
numbers = [1, 2, 3, 4, 5]

# Remove last item (default)
last = numbers.pop()
print(f"Removed: {last}")  # Removed: 5
print(numbers)  # [1, 2, 3, 4]

# Remove item at specific index
second = numbers.pop(1)
print(f"Removed: {second}")  # Removed: 2
print(numbers)  # [1, 3, 4]
```

#### del - Delete by index or slice
```python
numbers = [1, 2, 3, 4, 5]

del numbers[0]     # Remove first item
print(numbers)     # [2, 3, 4, 5]

del numbers[1:3]   # Remove slice
print(numbers)     # [2, 5]

# del numbers      # This would delete entire list
```

#### clear() - Remove all items
```python
numbers = [1, 2, 3, 4, 5]
numbers.clear()
print(numbers)  # []
```

---

## List Methods

### Essential Methods

```python
fruits = ['apple', 'banana', 'orange', 'banana', 'kiwi']

# count() - Count occurrences
print(fruits.count('banana'))  # 2

# index() - Find first index of value
print(fruits.index('orange'))  # 2

# index() with start and stop
print(fruits.index('banana', 2))  # 3 (find 'banana' starting from index 2)
```

### Sorting and Reversing

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
fruits = ['banana', 'apple', 'orange', 'kiwi']

# sort() - Sort in place (modifies original list)
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]

fruits.sort()
print(fruits)   # ['apple', 'banana', 'kiwi', 'orange']

# sort() with reverse
numbers.sort(reverse=True)
print(numbers)  # [9, 6, 5, 4, 3, 2, 1, 1]

# reverse() - Reverse the list in place
fruits.reverse()
print(fruits)   # ['orange', 'kiwi', 'banana', 'apple']
```

### Copying Lists

```python
original = [1, 2, 3, 4, 5]

# Method 1: copy() method
copy1 = original.copy()

# Method 2: slice notation
copy2 = original[:]

# Method 3: list() constructor
copy3 = list(original)

# Verify they're independent
copy1.append(6)
print("Original:", original)  # [1, 2, 3, 4, 5]
print("Copy:", copy1)         # [1, 2, 3, 4, 5, 6]
```

---

## List Operations

### Concatenation and Repetition
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# Concatenation with +
combined = list1 + list2
print(combined)  # [1, 2, 3, 4, 5, 6]

# Repetition with *
repeated = list1 * 3
print(repeated)  # [1, 2, 3, 1, 2, 3, 1, 2, 3]

# += for extending
list1 += list2
print(list1)     # [1, 2, 3, 4, 5, 6]
```

### Comparison
```python
# Lists can be compared element by element
print([1, 2, 3] == [1, 2, 3])    # True
print([1, 2, 3] == [3, 2, 1])    # False
print([1, 2, 3] < [1, 2, 4])     # True (lexicographic comparison)
print([1, 2] < [1, 2, 3])        # True (shorter list is smaller)
```

### Length and Other Built-ins
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

print(len(numbers))    # 8 (length)
print(min(numbers))    # 1 (minimum value)
print(max(numbers))    # 9 (maximum value)
print(sum(numbers))    # 31 (sum of all numbers)

# sorted() - Returns new sorted list (doesn't modify original)
sorted_nums = sorted(numbers)
print("Original:", numbers)      # [3, 1, 4, 1, 5, 9, 2, 6]
print("Sorted:", sorted_nums)    # [1, 1, 2, 3, 4, 5, 6, 9]
```

---

## List Comprehensions

List comprehensions provide a concise way to create lists.

### Basic Syntax
```python
# [expression for item in iterable]

# Traditional way
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension way
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### With Conditions
```python
# [expression for item in iterable if condition]

# Even numbers only
evens = [x for x in range(20) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Square of odd numbers
odd_squares = [x**2 for x in range(10) if x % 2 == 1]
print(odd_squares)  # [1, 9, 25, 49, 81]
```

### String Operations
```python
words = ['hello', 'world', 'python', 'programming']

# Convert to uppercase
upper_words = [word.upper() for word in words]
print(upper_words)  # ['HELLO', 'WORLD', 'PYTHON', 'PROGRAMMING']

# Get word lengths
lengths = [len(word) for word in words]
print(lengths)  # [5, 5, 6, 11]

# Filter by length
long_words = [word for word in words if len(word) > 5]
print(long_words)  # ['python', 'programming']
```

### Nested List Comprehensions
```python
# Create a 3x3 matrix
matrix = [[i*j for i in range(3)] for j in range(3)]
print(matrix)  # [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# Flatten a 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## Advanced List Features

### Nested Lists
```python
# 2D list (list of lists)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Access elements
print(matrix[0])     # [1, 2, 3] (first row)
print(matrix[1][2])  # 6 (row 1, column 2)

# Modify elements
matrix[0][0] = 10
print(matrix)  # [[10, 2, 3], [4, 5, 6], [7, 8, 9]]

# Iterate through 2D list
for row in matrix:
    for item in row:
        print(item, end=' ')
    print()  # New line after each row
```

### List as Stack (LIFO - Last In, First Out)
```python
stack = []

# Push items
stack.append('first')
stack.append('second')
stack.append('third')
print(stack)  # ['first', 'second', 'third']

# Pop items
item = stack.pop()
print(f"Popped: {item}")  # Popped: third
print(stack)  # ['first', 'second']
```

### List as Queue (FIFO - First In, First Out)
```python
from collections import deque

# For better performance, use deque for queues
queue = deque(['first', 'second', 'third'])

# Add to right end
queue.append('fourth')

# Remove from left end
item = queue.popleft()
print(f"Removed: {item}")  # Removed: first
print(list(queue))  # ['second', 'third', 'fourth']
```

---

## List vs Other Data Types

| Feature | List | Tuple | Set | Dictionary |
|---------|------|-------|-----|------------|
| **Mutable** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Ordered** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes (3.7+) |
| **Duplicates** | ✅ Yes | ✅ Yes | ❌ No | ❌ No (keys) |
| **Indexed** | ✅ Yes | ✅ Yes | ❌ No | 🔑 By key |
| **Syntax** | `[1, 2, 3]` | `(1, 2, 3)` | `{1, 2, 3}` | `{'a': 1}` |

### When to Use Lists:
- ✅ Need to store ordered data that may change
- ✅ Need to access items by index
- ✅ Want to allow duplicate values
- ✅ Need to frequently add/remove items
- ✅ Building shopping lists, to-do lists, user inputs

### When NOT to Use Lists:
- ❌ Data should never change (use tuple)
- ❌ Need to eliminate duplicates (use set)
- ❌ Need key-value relationships (use dictionary)
- ❌ Need very fast membership testing (use set)

---

## Common Use Cases

### 1. Data Collection and Processing
```python
# Collect user inputs
scores = []
while True:
    score = input("Enter a test score (or 'done'): ")
    if score.lower() == 'done':
        break
    scores.append(float(score))

# Process the data
if scores:
    average = sum(scores) / len(scores)
    print(f"Average score: {average:.2f}")
    print(f"Highest score: {max(scores)}")
    print(f"Lowest score: {min(scores)}")
```

### 2. Shopping List Manager
```python
shopping_list = []

def add_item(item):
    shopping_list.append(item)
    print(f"Added '{item}' to shopping list")

def remove_item(item):
    if item in shopping_list:
        shopping_list.remove(item)
        print(f"Removed '{item}' from shopping list")
    else:
        print(f"'{item}' not found in shopping list")

def show_list():
    if shopping_list:
        print("Shopping List:")
        for i, item in enumerate(shopping_list, 1):
            print(f"{i}. {item}")
    else:
        print("Shopping list is empty")

# Usage
add_item("milk")
add_item("bread")
add_item("eggs")
show_list()
remove_item("bread")
show_list()
```

### 3. Grade Book System
```python
class GradeBook:
    def __init__(self):
        self.students = []
    
    def add_student(self, name, grades):
        student = {
            'name': name,
            'grades': grades,
            'average': sum(grades) / len(grades) if grades else 0
        }
        self.students.append(student)
    
    def get_class_average(self):
        if not self.students:
            return 0
        total = sum(student['average'] for student in self.students)
        return total / len(self.students)
    
    def get_top_students(self, n=3):
        sorted_students = sorted(self.students, 
                               key=lambda x: x['average'], 
                               reverse=True)
        return sorted_students[:n]

# Usage
gradebook = GradeBook()
gradebook.add_student("Alice", [85, 92, 78, 96])
gradebook.add_student("Bob", [76, 88, 82, 79])
gradebook.add_student("Charlie", [95, 87, 91, 98])

print(f"Class average: {gradebook.get_class_average():.2f}")
top_students = gradebook.get_top_students(2)
for student in top_students:
    print(f"{student['name']}: {student['average']:.2f}")
```

---

## Practice Examples

### Example 1: Number Analysis
```python
# Analyze a list of numbers
numbers = [12, 45, 7, 23, 56, 89, 34, 67, 91, 8, 43, 76]

# Find even and odd numbers
evens = [n for n in numbers if n % 2 == 0]
odds = [n for n in numbers if n % 2 == 1]

# Find numbers in different ranges
small = [n for n in numbers if n < 20]
medium = [n for n in numbers if 20 <= n < 60]
large = [n for n in numbers if n >= 60]

print(f"Original numbers: {numbers}")
print(f"Even numbers: {evens}")
print(f"Odd numbers: {odds}")
print(f"Small (< 20): {small}")
print(f"Medium (20-59): {medium}")
print(f"Large (>= 60): {large}")
print(f"Sum of all: {sum(numbers)}")
print(f"Average: {sum(numbers) / len(numbers):.2f}")
```

### Example 2: Word Processor
```python
# Process a list of words
words = ["python", "programming", "is", "fun", "and", "powerful"]

# Various operations
print("Original words:", words)
print("Sorted alphabetically:", sorted(words))
print("Sorted by length:", sorted(words, key=len))
print("Capitalized:", [word.capitalize() for word in words])
print("Uppercase:", [word.upper() for word in words])
print("Words longer than 3 chars:", [word for word in words if len(word) > 3])
print("Total characters:", sum(len(word) for word in words))

# Create sentences
sentence = ' '.join(words)
print("As sentence:", sentence)

# Reverse word order
reversed_words = words[::-1]
print("Reversed order:", reversed_words)
```

### Example 3: Simple To-Do List
```python
todo_list = []

def show_menu():
    print("\n=== TO-DO LIST ===")
    print("1. Add task")
    print("2. View tasks")
    print("3. Mark task as done")
    print("4. Remove task")
    print("5. Exit")

def add_task():
    task = input("Enter a new task: ")
    todo_list.append({"task": task, "done": False})
    print(f"Added: {task}")

def view_tasks():
    if not todo_list:
        print("No tasks yet!")
        return
    
    print("\nYour tasks:")
    for i, item in enumerate(todo_list, 1):
        status = "✓" if item["done"] else "○"
        print(f"{i}. {status} {item['task']}")

def mark_done():
    view_tasks()
    if not todo_list:
        return
    
    try:
        index = int(input("Enter task number to mark as done: ")) - 1
        if 0 <= index < len(todo_list):
            todo_list[index]["done"] = True
            print(f"Marked as done: {todo_list[index]['task']}")
        else:
            print("Invalid task number!")
    except ValueError:
        print("Please enter a valid number!")

def remove_task():
    view_tasks()
    if not todo_list:
        return
    
    try:
        index = int(input("Enter task number to remove: ")) - 1
        if 0 <= index < len(todo_list):
            removed = todo_list.pop(index)
            print(f"Removed: {removed['task']}")
        else:
            print("Invalid task number!")
    except ValueError:
        print("Please enter a valid number!")

# Main program loop
while True:
    show_menu()
    choice = input("Choose an option (1-5): ")
    
    if choice == '1':
        add_task()
    elif choice == '2':
        view_tasks()
    elif choice == '3':
        mark_done()
    elif choice == '4':
        remove_task()
    elif choice == '5':
        print("Goodbye!")
        break
    else:
        print("Invalid choice! Please try again.")
```

---

## Quick Reference Cheat Sheet

### Creating Lists
```python
# Empty list
empty = []

# With values
numbers = [1, 2, 3, 4, 5]
fruits = ['apple', 'banana', 'orange']

# From other iterables
from_string = list('hello')     # ['h', 'e', 'l', 'l', 'o']
from_range = list(range(5))     # [0, 1, 2, 3, 4]
```

### Accessing Elements
```python
my_list = [10, 20, 30, 40, 50]

# By index
print(my_list[0])    # 10 (first)
print(my_list[-1])   # 50 (last)

# Slicing
print(my_list[1:4])  # [20, 30, 40]
print(my_list[:3])   # [10, 20, 30]
print(my_list[::2])  # [10, 30, 50]
```

### Modifying Lists
```python
my_list = [1, 2, 3]

# Add items
my_list.append(4)           # [1, 2, 3, 4]
my_list.insert(0, 0)        # [0, 1, 2, 3, 4]
my_list.extend([5, 6])      # [0, 1, 2, 3, 4, 5, 6]

# Remove items
my_list.remove(0)           # Remove first occurrence of 0
last = my_list.pop()        # Remove and return last item
del my_list[0]              # Remove by index
my_list.clear()             # Remove all items
```

### Common Operations
```python
numbers = [3, 1, 4, 1, 5, 9]

# Information
print(len(numbers))         # 6
print(min(numbers))         # 1
print(max(numbers))         # 9
print(sum(numbers))         # 23

# Sorting
numbers.sort()              # Sort in place
sorted_nums = sorted(numbers)  # Return new sorted list

# Other methods
print(numbers.count(1))     # Count occurrences
print(numbers.index(4))     # Find index of value
numbers.reverse()           # Reverse in place
```

### List Comprehensions
```python
# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(10)]

# With condition: [expression for item in iterable if condition]
evens = [x for x in range(20) if x % 2 == 0]

# Transform strings
words = ['hello', 'world']
upper = [word.upper() for word in words]
```

---

## Performance Tips

### ✅ Do This (Efficient)
```python
# Use list comprehensions instead of loops
squares = [x**2 for x in range(1000)]

# Use extend() instead of multiple append() calls
my_list.extend([1, 2, 3, 4, 5])

# Use enumerate() when you need both index and value
for i, value in enumerate(my_list):
    print(f"{i}: {value}")

# Pre-allocate list size if known
zeros = [0] * 1000  # Faster than 1000 append() calls
```

### ❌ Avoid This (Inefficient)
```python
# Don't use loops when list comprehension works
squares = []
for x in range(1000):
    squares.append(x**2)  # Slower

# Don't use multiple append() calls
my_list.append(1)
my_list.append(2)
my_list.append(3)  # Use extend() instead

# Don't use range(len()) when enumerate() works
for i in range(len(my_list)):
    print(f"{i}: {my_list[i]}")  # Less readable
```

---

## Common Mistakes and Solutions

### Mistake 1: Modifying List While Iterating
```python
# ❌ Wrong - can skip elements or cause errors
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # Don't do this!

# ✅ Correct - iterate over a copy
numbers = [1, 2, 3, 4, 5]
for num in numbers[:]:  # numbers[:] creates a copy
    if num % 2 == 0:
        numbers.remove(num)

# ✅ Better - use list comprehension
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
```

### Mistake 2: Shallow vs Deep Copy
```python
# ❌ Shallow copy issue with nested lists
original = [[1, 2], [3, 4]]
copy = original.copy()
copy[0].append(3)
print(original)  # [[1, 2, 3], [3, 4]] - original changed!

# ✅ Deep copy for nested structures
import copy
original = [[1, 2], [3, 4]]
deep_copy = copy.deepcopy(original)
deep_copy[0].append(3)
print(original)  # [[1, 2], [3, 4]] - original unchanged
```

---

## Key Takeaways

1. **Lists are mutable** - you can change them after creation
2. **Use square brackets** `[]` to create lists
3. **Indexing starts at 0** - first item is `list[0]`
4. **Negative indexing** works backwards - `list[-1]` is the last item
5. **List comprehensions** are often faster and more readable than loops
6. **append()** adds one item, **extend()** adds multiple items
7. **pop()** removes and returns an item, **remove()** just removes
8. **Lists are ordered** - they maintain the sequence you put items in
9. **Use lists when** you need to store ordered, changeable data
10. **Be careful** when modifying lists while iterating over them