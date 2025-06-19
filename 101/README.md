# Python Loops and Iterators Guide 🐍

A beginner-friendly guide to understanding loops and iterators in Python. Perfect for new programmers!

## Table of Contents
- [What are Loops?](#what-are-loops)
- [For Loops](#for-loops)
- [While Loops](#while-loops)
- [Iterator Functions](#iterator-functions)
- [Dictionary Loops](#dictionary-loops)
- [List Comprehensions](#list-comprehensions)
- [Loop Control](#loop-control)
- [Practice Examples](#practice-examples)

---

## What are Loops?

Loops let you repeat code multiple times without writing it over and over again. Think of it like telling someone "do this 10 times" instead of saying "do this" 10 separate times!

---

## For Loops

### Basic For Loop
The most common way to repeat actions in Python.

```python
# Print each fruit
fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(fruit)

# Output:
# apple
# banana
# orange
```

### Loop with Numbers (range)
```python
# Print numbers 0 to 4
for i in range(5):
    print(i)

# Print numbers 1 to 5
for i in range(1, 6):
    print(i)

# Print even numbers 0 to 8
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8
```

### Loop with Index Numbers (enumerate)
When you need both the position and the item:

```python
colors = ['red', 'green', 'blue']
for index, color in enumerate(colors):
    print(f"{index + 1}. {color}")

# Output:
# 1. red
# 2. green
# 3. blue
```

---

## While Loops

Repeats as long as a condition is true. Be careful not to create infinite loops!

```python
# Count from 1 to 5
count = 1
while count <= 5:
    print(f"Count: {count}")
    count += 1

# Simple game loop
playing = True
while playing:
    user_input = input("Type 'quit' to stop: ")
    if user_input == 'quit':
        playing = False
    else:
        print(f"You typed: {user_input}")
```

---

## Iterator Functions

### zip() - Combine Lists
Perfect when you have related data in separate lists:

```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
cities = ['NYC', 'LA', 'Chicago']

for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} years old and lives in {city}")
```

### map() - Transform Each Item
Apply the same operation to every item:

```python
numbers = [1, 2, 3, 4, 5]

# Square each number
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Convert to strings
strings = list(map(str, numbers))
print(strings)  # ['1', '2', '3', '4', '5']
```

### filter() - Keep Only Some Items
Keep items that match a condition:

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Keep only even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# Keep only numbers greater than 5
big_numbers = list(filter(lambda x: x > 5, numbers))
print(big_numbers)  # [6, 7, 8, 9, 10]
```

---

## Dictionary Loops

Dictionaries store key-value pairs. Here's how to loop through them:

```python
student = {
    'name': 'John',
    'age': 20,
    'grade': 'A',
    'city': 'Boston'
}

# Loop through keys only
print("Keys:")
for key in student:
    print(key)

# Loop through values only
print("\nValues:")
for value in student.values():
    print(value)

# Loop through both keys and values
print("\nKey-Value pairs:")
for key, value in student.items():
    print(f"{key}: {value}")
```

---

## List Comprehensions

A short way to create new lists. Think of it as a "one-line for loop":

```python
# Regular way
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension way (shorter!)
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With conditions
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Transform strings
words = ['hello', 'world', 'python']
upper_words = [word.upper() for word in words]
print(upper_words)  # ['HELLO', 'WORLD', 'PYTHON']
```

---

## Loop Control

### break - Stop the Loop Early
```python
# Find the first even number
numbers = [1, 3, 5, 8, 9, 10]
for num in numbers:
    if num % 2 == 0:
        print(f"Found first even number: {num}")
        break
```

### continue - Skip Current Round
```python
# Print only odd numbers
for i in range(10):
    if i % 2 == 0:  # If even, skip it
        continue
    print(i)  # Only odd numbers: 1, 3, 5, 7, 9
```

### else with Loops
The `else` runs only if the loop completes normally (no `break`):

```python
# Search for a number
numbers = [1, 2, 3, 4, 5]
search_for = 6

for num in numbers:
    if num == search_for:
        print(f"Found {search_for}!")
        break
else:
    print(f"{search_for} not found in the list")
```

---

## Practice Examples

### Example 1: Multiplication Table
```python
number = 5
print(f"Multiplication table for {number}:")
for i in range(1, 11):
    result = number * i
    print(f"{number} × {i} = {result}")
```

### Example 2: Count Vowels
```python
text = "Hello World"
vowels = 'aeiouAEIOU'
count = 0

for letter in text:
    if letter in vowels:
        count += 1

print(f"Number of vowels: {count}")
```

### Example 3: Shopping List
```python
shopping_list = ['bread', 'milk', 'eggs', 'cheese']
prices = [2.50, 3.00, 4.00, 5.50]

total = 0
print("Shopping List:")
for item, price in zip(shopping_list, prices):
    print(f"- {item}: ${price}")
    total += price

print(f"\nTotal: ${total}")
```

---

## Quick Reference

| Loop Type | When to Use | Example |
|-----------|-------------|---------|
| `for` | When you know how many times to repeat | `for i in range(5)` |
| `while` | When you repeat until a condition changes | `while user_input != 'quit'` |
| `enumerate` | When you need item + position | `for i, item in enumerate(list)` |
| `zip` | When combining multiple lists | `for a, b in zip(list1, list2)` |
| List comprehension | Creating new lists quickly | `[x*2 for x in numbers]` |

---

## Tips for Beginners

1. **Start simple**: Begin with basic `for` loops before moving to advanced features
2. **Avoid infinite loops**: Always make sure `while` loops have a way to stop
3. **Use meaningful names**: `for student in students` is better than `for s in students`
4. **Test with small data**: Use small lists when learning
5. **Practice daily**: The more you use loops, the more natural they become!

---

## Need Help?

- **Python Documentation**: [docs.python.org](https://docs.python.org)
- **Practice Online**: [repl.it](https://replit.com), [Python.org](https://python.org)
- **Community**: [r/learnpython](https://reddit.com/r/learnpython)

Happy coding! 🚀