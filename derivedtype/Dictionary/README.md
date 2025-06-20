# Python Dictionary Guide 📚

A complete beginner-friendly guide to understanding and using dictionaries in Python. Learn what they are, how to use them, and master all their powerful features!

## Table of Contents

- [What is a Dictionary?](#what-is-a-dictionary)
- [Creating Dictionaries](#creating-dictionaries)
- [Accessing Dictionary Elements](#accessing-dictionary-elements)
- [Modifying Dictionaries](#modifying-dictionaries)
- [Dictionary Methods](#dictionary-methods)
- [Dictionary Operations](#dictionary-operations)
- [Dictionary Comprehensions](#dictionary-comprehensions)
- [Advanced Dictionary Features](#advanced-dictionary-features)
- [Dictionary vs Other Data Types](#dictionary-vs-other-data-types)
- [Common Use Cases](#common-use-cases)
- [Practice Examples](#practice-examples)

---

## What is a Dictionary?

A **dictionary** is a collection data type that stores data in key-value pairs. It's one of the most powerful and flexible data structures in Python.

Think of dictionaries as:

- 📖 A real dictionary - you look up words (keys) to find their meanings (values)
- 📞 A phone book - names (keys) mapped to phone numbers (values)
- 🏠 Address book - people's names (keys) mapped to their addresses (values)
- 🎮 Game inventory - item names (keys) mapped to quantities (values)

### Key Characteristics:

- 🔑 **Key-Value pairs** - each item consists of a key and its associated value
- 📝 **Mutable** - can be changed after creation (add, remove, modify)
- 🚫 **No duplicates** - keys must be unique (values can be duplicated)
- 🎯 **Fast lookups** - O(1) average time complexity for access
- 📋 **Ordered** - maintains insertion order (Python 3.7+)
- 🔐 **Keys must be immutable** - strings, numbers, tuples (but not lists)

---

## Creating Dictionaries

### Method 1: Using Curly Braces {}

```python
# Empty dictionary
empty_dict = {}
print(empty_dict)  # {}

# Dictionary with initial values
person = {
    'name': 'Alice',
    'age': 25,
    'city': 'New York'
}
print(person)  # {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Mixed data types
mixed = {
    'string_key': 'Hello',
    42: 'number key',
    'list_value': [1, 2, 3],
    'nested_dict': {'inner': 'value'}
}
print(mixed)
```

### Method 2: Using dict() Constructor

```python
# From keyword arguments
person = dict(name='Bob', age=30, city='London')
print(person)  # {'name': 'Bob', 'age': 30, 'city': 'London'}

# From list of tuples
pairs = [('apple', 5), ('banana', 3), ('orange', 7)]
fruits = dict(pairs)
print(fruits)  # {'apple': 5, 'banana': 3, 'orange': 7}

# From zip of two lists
keys = ['name', 'age', 'city']
values = ['Charlie', 35, 'Paris']
person = dict(zip(keys, values))
print(person)  # {'name': 'Charlie', 'age': 35, 'city': 'Paris'}
```

### Method 3: Using Dictionary Comprehension

```python
# Create dictionary from range
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# From existing data
words = ['hello', 'world', 'python']
word_lengths = {word: len(word) for word in words}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}
```

---

## Accessing Dictionary Elements

### Using Keys

```python
person = {
    'name': 'Alice',
    'age': 25,
    'city': 'New York',
    'hobbies': ['reading', 'swimming']
}

# Direct key access
print(person['name'])     # Alice
print(person['age'])      # 25
print(person['hobbies'])  # ['reading', 'swimming']

# Accessing nested values
print(person['hobbies'][0])  # reading
```

### Using get() Method (Safer)

```python
person = {'name': 'Alice', 'age': 25}

# get() returns None if key doesn't exist
print(person.get('name'))     # Alice
print(person.get('salary'))   # None

# get() with default value
print(person.get('salary', 0))        # 0
print(person.get('city', 'Unknown'))  # Unknown

# Direct access raises KeyError if key doesn't exist
# print(person['salary'])  # KeyError!
```

### Checking if Key Exists

```python
person = {'name': 'Alice', 'age': 25}

# Using 'in' operator
print('name' in person)     # True
print('salary' in person)   # False
print('age' not in person)  # False

# Using get() with sentinel value
if person.get('salary') is not None:
    print("Salary exists")
else:
    print("No salary information")
```

---

## Modifying Dictionaries

### Adding and Updating Items

```python
person = {'name': 'Alice', 'age': 25}
print("Original:", person)

# Add new key-value pair
person['city'] = 'New York'
print("After adding city:", person)

# Update existing value
person['age'] = 26
print("After updating age:", person)

# Add multiple items
person['occupation'] = 'Engineer'
person['salary'] = 75000
print("After adding multiple:", person)
```

### Using update() Method

```python
person = {'name': 'Alice', 'age': 25}

# Update with another dictionary
new_info = {'city': 'Boston', 'occupation': 'Doctor'}
person.update(new_info)
print(person)  # {'name': 'Alice', 'age': 25, 'city': 'Boston', 'occupation': 'Doctor'}

# Update with keyword arguments
person.update(salary=80000, experience=5)
print(person)

# Update with list of tuples
person.update([('department', 'IT'), ('manager', 'John')])
print(person)
```

### Removing Items

#### del - Remove by key

```python
person = {'name': 'Alice', 'age': 25, 'city': 'New York'}

del person['city']
print(person)  # {'name': 'Alice', 'age': 25}

# Be careful - KeyError if key doesn't exist
# del person['salary']  # KeyError!
```

#### pop() - Remove and return value

```python
person = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Remove and get value
age = person.pop('age')
print(f"Removed age: {age}")  # Removed age: 25
print(person)  # {'name': 'Alice', 'city': 'New York'}

# pop() with default value (no KeyError)
salary = person.pop('salary', 'Not found')
print(f"Salary: {salary}")  # Salary: Not found
```

#### popitem() - Remove and return last inserted item

```python
person = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Remove last item (Python 3.7+)
last_item = person.popitem()
print(f"Removed: {last_item}")  # Removed: ('city', 'New York')
print(person)  # {'name': 'Alice', 'age': 25}
```

#### clear() - Remove all items

```python
person = {'name': 'Alice', 'age': 25, 'city': 'New York'}
person.clear()
print(person)  # {}
```

---

## Dictionary Methods

### Essential Methods

```python
person = {
    'name': 'Alice',
    'age': 25,
    'city': 'New York',
    'occupation': 'Engineer'
}

# keys() - Get all keys
print(person.keys())    # dict_keys(['name', 'age', 'city', 'occupation'])
print(list(person.keys()))  # ['name', 'age', 'city', 'occupation']

# values() - Get all values
print(person.values())  # dict_values(['Alice', 25, 'New York', 'Engineer'])
print(list(person.values()))  # ['Alice', 25, 'New York', 'Engineer']

# items() - Get all key-value pairs
print(person.items())   # dict_items([('name', 'Alice'), ('age', 25), ...])
print(list(person.items()))  # [('name', 'Alice'), ('age', 25), ...]
```

### Copying Dictionaries

```python
original = {'a': 1, 'b': 2, 'c': 3}

# Method 1: copy() method (shallow copy)
copy1 = original.copy()

# Method 2: dict() constructor
copy2 = dict(original)

# Method 3: dictionary comprehension
copy3 = {k: v for k, v in original.items()}

# Verify independence
copy1['d'] = 4
print("Original:", original)  # {'a': 1, 'b': 2, 'c': 3}
print("Copy:", copy1)         # {'a': 1, 'b': 2, 'c': 3, 'd': 4}
```

### setdefault() - Get or Set Default Value

```python
person = {'name': 'Alice', 'age': 25}

# Get existing key
name = person.setdefault('name', 'Unknown')
print(name)  # Alice (existing value)

# Set default for missing key
city = person.setdefault('city', 'Unknown')
print(city)    # Unknown (new default value)
print(person)  # {'name': 'Alice', 'age': 25, 'city': 'Unknown'}

# Useful for counters
text = "hello world"
char_count = {}
for char in text:
    char_count.setdefault(char, 0)
    char_count[char] += 1
print(char_count)  # {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
```

---

## Dictionary Operations

### Merging Dictionaries

```python
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
dict3 = {'e': 5, 'f': 6}

# Method 1: Using update()
merged = dict1.copy()
merged.update(dict2)
merged.update(dict3)
print(merged)  # {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}

# Method 2: Using ** operator (Python 3.5+)
merged = {**dict1, **dict2, **dict3}
print(merged)  # {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}

# Method 3: Using | operator (Python 3.9+)
merged = dict1 | dict2 | dict3
print(merged)  # {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
```

### Dictionary Comparison

```python
dict1 = {'a': 1, 'b': 2, 'c': 3}
dict2 = {'a': 1, 'b': 2, 'c': 3}
dict3 = {'a': 1, 'b': 2, 'c': 4}

print(dict1 == dict2)  # True (same key-value pairs)
print(dict1 == dict3)  # False (different values)

# Order doesn't matter for equality
dict4 = {'c': 3, 'a': 1, 'b': 2}
print(dict1 == dict4)  # True
```

### Length and Membership

```python
person = {'name': 'Alice', 'age': 25, 'city': 'New York'}

print(len(person))        # 3 (number of key-value pairs)
print('name' in person)   # True (key exists)
print('Alice' in person)  # False (checking keys, not values)

# To check if value exists
print('Alice' in person.values())  # True
```

---

## Dictionary Comprehensions

Dictionary comprehensions provide a concise way to create dictionaries.

### Basic Syntax

```python
# {key_expression: value_expression for item in iterable}

# Traditional way
squares = {}
for x in range(5):
    squares[x] = x**2

# Dictionary comprehension way
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### With Conditions

```python
# {key_expr: value_expr for item in iterable if condition}

# Even numbers and their squares
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
print(even_squares)  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Filter words by length
words = ['hello', 'world', 'python', 'programming', 'is', 'fun']
long_words = {word: len(word) for word in words if len(word) > 4}
print(long_words)  # {'hello': 5, 'world': 5, 'python': 6, 'programming': 11}
```

### From Existing Data

```python
# Convert list to dictionary with indices
fruits = ['apple', 'banana', 'orange']
fruit_dict = {i: fruit for i, fruit in enumerate(fruits)}
print(fruit_dict)  # {0: 'apple', 1: 'banana', 2: 'orange'}

# Reverse a dictionary
original = {'a': 1, 'b': 2, 'c': 3}
reversed_dict = {v: k for k, v in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# Transform values
prices = {'apple': 1.20, 'banana': 0.80, 'orange': 1.50}
rounded_prices = {fruit: round(price) for fruit, price in prices.items()}
print(rounded_prices)  # {'apple': 1, 'banana': 1, 'orange': 2}
```

### Nested Dictionary Comprehensions

```python
# Create multiplication table
multiplication_table = {
    i: {j: i*j for j in range(1, 4)}
    for i in range(1, 4)
}
print(multiplication_table)
# {1: {1: 1, 2: 2, 3: 3}, 2: {1: 2, 2: 4, 3: 6}, 3: {1: 3, 2: 6, 3: 9}}

# Group words by first letter
words = ['apple', 'banana', 'apricot', 'blueberry', 'cherry', 'avocado']
grouped = {}
for word in words:
    first_letter = word[0]
    if first_letter not in grouped:
        grouped[first_letter] = []
    grouped[first_letter].append(word)

# Using comprehension with setdefault
grouped = {}
for word in words:
    grouped.setdefault(word[0], []).append(word)
print(grouped)  # {'a': ['apple', 'apricot', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}
```

---

## Advanced Dictionary Features

### Nested Dictionaries

```python
# Company structure
company = {
    'departments': {
        'engineering': {
            'employees': ['Alice', 'Bob', 'Charlie'],
            'budget': 500000,
            'manager': 'Dave'
        },
        'sales': {
            'employees': ['Eve', 'Frank'],
            'budget': 300000,
            'manager': 'Grace'
        }
    },
    'company_info': {
        'name': 'TechCorp',
        'founded': 2010,
        'employees_total': 5
    }
}

# Access nested values
print(company['departments']['engineering']['manager'])  # Dave
print(company['company_info']['name'])  # TechCorp

# Modify nested values
company['departments']['engineering']['employees'].append('Helen')
company['departments']['marketing'] = {
    'employees': ['Ivan'],
    'budget': 200000,
    'manager': 'Jack'
}

print(company['departments']['engineering']['employees'])
# ['Alice', 'Bob', 'Charlie', 'Helen']
```

### Using defaultdict for Automatic Default Values

```python
from collections import defaultdict

# Regular dictionary - KeyError if key doesn't exist
regular_dict = {}
# regular_dict['missing_key'].append('value')  # KeyError!

# defaultdict - automatically creates default value
dd_list = defaultdict(list)  # Default value is empty list
dd_list['fruits'].append('apple')
dd_list['fruits'].append('banana')
dd_list['vegetables'].append('carrot')

print(dict(dd_list))  # {'fruits': ['apple', 'banana'], 'vegetables': ['carrot']}

# defaultdict with int (default value is 0)
dd_int = defaultdict(int)
text = "hello world"
for char in text:
    dd_int[char] += 1

print(dict(dd_int))  # {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
```

### Using Counter for Counting

```python
from collections import Counter

# Count elements in a list
fruits = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
fruit_count = Counter(fruits)
print(fruit_count)  # Counter({'apple': 3, 'banana': 2, 'orange': 1})

# Count characters in a string
text = "hello world"
char_count = Counter(text)
print(char_count)  # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

# Most common elements
print(char_count.most_common(3))  # [('l', 3), ('o', 2), ('h', 1)]

# Counter arithmetic
counter1 = Counter(['a', 'b', 'c', 'a'])
counter2 = Counter(['a', 'b', 'b', 'd'])
print(counter1 + counter2)  # Counter({'a': 3, 'b': 3, 'c': 1, 'd': 1})
print(counter1 - counter2)  # Counter({'c': 1, 'a': 1})
```

---

## Dictionary vs Other Data Types

| Feature           | Dictionary                    | List               | Tuple              | Set             |
| ----------------- | ----------------------------- | ------------------ | ------------------ | --------------- |
| **Mutable**       | ✅ Yes                        | ✅ Yes             | ❌ No              | ✅ Yes          |
| **Ordered**       | ✅ Yes (3.7+)                 | ✅ Yes             | ✅ Yes             | ❌ No           |
| **Duplicates**    | 🔑 Keys: No<br>📝 Values: Yes | ✅ Yes             | ✅ Yes             | ❌ No           |
| **Access Method** | 🔑 By key                     | 📍 By index        | 📍 By index        | 🚫 No access    |
| **Use Case**      | Key-value mapping             | Ordered collection | Immutable sequence | Unique elements |
| **Syntax**        | `{'key': 'value'}`            | `[1, 2, 3]`        | `(1, 2, 3)`        | `{1, 2, 3}`     |

### When to Use Dictionaries:

- ✅ Need fast lookups by key
- ✅ Want to associate values with meaningful names
- ✅ Building mappings or relationships
- ✅ Caching/memoization
- ✅ Configuration settings
- ✅ Counting occurrences
- ✅ Grouping data by categories

### When NOT to Use Dictionaries:

- ❌ Need ordered sequence (use list)
- ❌ Need to store only unique values without keys (use set)
- ❌ Data should be immutable (use tuple or frozenset)
- ❌ Need mathematical operations on data (use numpy arrays)

---

## Common Use Cases

### 1. Contact Book

```python
contacts = {}

def add_contact(name, phone, email):
    contacts[name] = {
        'phone': phone,
        'email': email,
        'added_date': '2024-01-15'  # In real app, use datetime
    }
    print(f"Added contact: {name}")

def find_contact(name):
    if name in contacts:
        contact = contacts[name]
        print(f"Name: {name}")
        print(f"Phone: {contact['phone']}")
        print(f"Email: {contact['email']}")
    else:
        print(f"Contact '{name}' not found")

def update_contact(name, **kwargs):
    if name in contacts:
        for key, value in kwargs.items():
            contacts[name][key] = value
        print(f"Updated {name}")
    else:
        print(f"Contact '{name}' not found")

# Usage
add_contact("Alice", "123-456-7890", "alice@email.com")
add_contact("Bob", "987-654-3210", "bob@email.com")
find_contact("Alice")
update_contact("Alice", phone="111-222-3333")
```

### 2. Student Grade Manager

```python
class GradeManager:
    def __init__(self):
        self.students = {}

    def add_student(self, name):
        if name not in self.students:
            self.students[name] = {
                'grades': [],
                'assignments': {},
                'attendance': 0
            }
            print(f"Added student: {name}")
        else:
            print(f"Student {name} already exists")

    def add_grade(self, name, subject, grade):
        if name in self.students:
            self.students[name]['grades'].append(grade)
            self.students[name]['assignments'][subject] = grade
            print(f"Added grade {grade} for {name} in {subject}")
        else:
            print(f"Student {name} not found")

    def get_average(self, name):
        if name in self.students and self.students[name]['grades']:
            grades = self.students[name]['grades']
            return sum(grades) / len(grades)
        return 0

    def get_class_average(self):
        if not self.students:
            return 0

        total_avg = 0
        count = 0
        for name in self.students:
            avg = self.get_average(name)
            if avg > 0:
                total_avg += avg
                count += 1

        return total_avg / count if count > 0 else 0

    def print_report(self):
        print("\n=== GRADE REPORT ===")
        for name, data in self.students.items():
            avg = self.get_average(name)
            print(f"{name}: Average = {avg:.2f}")
            for subject, grade in data['assignments'].items():
                print(f"  {subject}: {grade}")

# Usage
gm = GradeManager()
gm.add_student("Alice")
gm.add_student("Bob")
gm.add_grade("Alice", "Math", 85)
gm.add_grade("Alice", "Science", 92)
gm.add_grade("Bob", "Math", 78)
gm.add_grade("Bob", "Science", 88)
gm.print_report()
```

### 3. Word Frequency Counter

```python
def analyze_text(text):
    # Clean and split text
    words = text.lower().replace(',', '').replace('.', '').split()

    # Count word frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Count character frequencies
    char_freq = {}
    for char in text.lower():
        if char.isalpha():
            char_freq[char] = char_freq.get(char, 0) + 1

    # Analysis results
    results = {
        'total_words': len(words),
        'unique_words': len(word_freq),
        'word_frequencies': word_freq,
        'char_frequencies': char_freq,
        'most_common_words': [],
        'longest_words': []
    }

    # Find most common words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    results['most_common_words'] = sorted_words[:5]

    # Find longest words
    sorted_by_length = sorted(words, key=len, reverse=True)
    results['longest_words'] = list(set(sorted_by_length[:5]))

    return results

def print_analysis(results):
    print(f"Total words: {results['total_words']}")
    print(f"Unique words: {results['unique_words']}")
    print("\nMost common words:")
    for word, count in results['most_common_words']:
        print(f"  {word}: {count}")

    print(f"\nLongest words: {', '.join(results['longest_words'])}")

# Usage
text = """
Python is a powerful programming language. Python is easy to learn
and Python is widely used in data science, web development, and automation.
Python's simplicity makes it perfect for beginners and experts alike.
"""

analysis = analyze_text(text)
print_analysis(analysis)
```

---

## Practice Examples

### Example 1: Inventory Management System

```python
inventory = {
    'electronics': {
        'laptop': {'price': 999.99, 'quantity': 5, 'supplier': 'TechCorp'},
        'mouse': {'price': 25.50, 'quantity': 50, 'supplier': 'AccessoriesInc'},
        'keyboard': {'price': 75.00, 'quantity': 30, 'supplier': 'AccessoriesInc'}
    },
    'books': {
        'python_guide': {'price': 39.99, 'quantity': 20, 'supplier': 'BookPublisher'},
        'data_science': {'price': 49.99, 'quantity': 15, 'supplier': 'BookPublisher'}
    }
}

def add_item(category, item_name, price, quantity, supplier):
    if category not in inventory:
        inventory[category] = {}

    inventory[category][item_name] = {
        'price': price,
        'quantity': quantity,
        'supplier': supplier
    }
    print(f"Added {item_name} to {category}")

def update_quantity(category, item_name, new_quantity):
    if category in inventory and item_name in inventory[category]:
        inventory[category][item_name]['quantity'] = new_quantity
        print(f"Updated {item_name} quantity to {new_quantity}")
    else:
        print("Item not found")

def get_total_value():
    total = 0
    for category in inventory:
        for item, details in inventory[category].items():
            total += details['price'] * details['quantity']
    return total

def low_stock_report(threshold=10):
    print(f"\n=== LOW STOCK REPORT (< {threshold}) ===")
    for category, items in inventory.items():
        for item, details in items.items():
            if details['quantity'] < threshold:
                print(f"{category.title()}: {item} - {details['quantity']} left")

# Usage
add_item('office', 'desk_chair', 199.99, 8, 'OfficeSupply')
update_quantity('electronics', 'laptop', 3)
print(f"Total inventory value: ${get_total_value():.2f}")
low_stock_report()
```

### Example 2: Menu Ordering System

```python
menu = {
    'appetizers': {
        'spring_rolls': {'price': 8.99, 'ingredients': ['vegetables', 'wrapper', 'sauce']},
        'chicken_wings': {'price': 12.99, 'ingredients': ['chicken', 'sauce', 'celery']}
    },
    'main_courses': {
        'burger': {'price': 15.99, 'ingredients': ['beef', 'bun', 'lettuce', 'tomato']},
        'pasta': {'price': 14.99, 'ingredients': ['pasta', 'sauce', 'cheese']},
        'salad': {'price': 11.99, 'ingredients': ['lettuce', 'tomato', 'cucumber', 'dressing']}
    },
    'beverages': {
        'soda': {'price': 2.99, 'ingredients': ['carbonated_water', 'syrup']},
        'coffee': {'price': 3.99, 'ingredients': ['coffee_beans', 'water']}
    }
}

class Order:
    def __init__(self, customer_name):
        self.customer_name = customer_name
        self.items = {}
        self.total = 0

    def add_item(self, category, item_name, quantity=1):
        if category in menu and item_name in menu[category]:
            if item_name not in self.items:
                self.items[item_name] = {
                    'quantity': 0,
                    'price': menu[category][item_name]['price'],
                    'category': category
                }

            self.items[item_name]['quantity'] += quantity
            self.calculate_total()
            print(f"Added {quantity} x {item_name} to order")
        else:
            print("Item not found in menu")

    def remove_item(self, item_name):
        if item_name in self.items:
            del self.items[item_name]
            self.calculate_total()
            print(f"Removed {item_name} from order")
        else:
            print("Item not in order")

    def calculate_total(self):
        self.total = sum(
            item['price'] * item['quantity']
            for item in self.items.values()


def print_receipt(self):
        print(f"\n=== ORDER RECEIPT FOR {self.customer_name.upper()} ===")
        print("-" * 50)

        for item_name, details in self.items.items():
            price = details['price']
            quantity = details['quantity']
            subtotal = price * quantity
            print(f"{item_name.replace('_', ' ').title():<20} ${price:>6.2f} x {quantity} = ${subtotal:>7.2f}")

        print("-" * 50)
        print(f"{'TOTAL':<20} ${self.total:>22.2f}")
        print("=" * 50)

# Usage
def display_menu():
    print("\n=== RESTAURANT MENU ===")
    for category, items in menu.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item_name, details in items.items():
            print(f"  {item_name.replace('_', ' ').title():<15} - ${details['price']:.2f}")

# Create and manage orders
display_menu()

order1 = Order("John Doe")
order1.add_item('appetizers', 'spring_rolls', 2)
order1.add_item('main_courses', 'burger', 1)
order1.add_item('beverages', 'soda', 2)
order1.print_receipt()

order2 = Order("Jane Smith")
order2.add_item('main_courses', 'pasta', 1)
order2.add_item('main_courses', 'salad', 1)
order2.add_item('beverages', 'coffee', 1)
order2.remove_item('salad')
order2.print_receipt()
```

### Example 3: Configuration Manager

```python
class ConfigManager:
    def __init__(self):
        self.config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'username': 'admin',
                'password': None,
                'database_name': 'myapp'
            },
            'app_settings': {
                'debug_mode': True,
                'log_level': 'INFO',
                'max_connections': 100,
                'timeout': 30
            },
            'features': {
                'user_registration': True,
                'email_notifications': True,
                'file_upload': False,
                'premium_features': False
            }
        }

    def get_setting(self, category, key, default=None):
        """Get a specific setting with optional default value"""
        return self.config.get(category, {}).get(key, default)

    def set_setting(self, category, key, value):
        """Set a specific setting"""
        if category not in self.config:
            self.config[category] = {}
        self.config[category][key] = value
        print(f"Updated {category}.{key} = {value}")

    def get_category(self, category):
        """Get all settings in a category"""
        return self.config.get(category, {})

    def validate_config(self):
        """Validate configuration and return issues"""
        issues = []

        # Check required database settings
        db_config = self.config.get('database', {})
        required_db_keys = ['host', 'port', 'username', 'database_name']
        for key in required_db_keys:
            if not db_config.get(key):
                issues.append(f"Database setting '{key}' is missing or empty")

        # Check port is valid number
        port = db_config.get('port')
        if port and not isinstance(port, int):
            issues.append("Database port must be an integer")

        # Check app settings
        app_config = self.config.get('app_settings', {})
        max_conn = app_config.get('max_connections')
        if max_conn and (not isinstance(max_conn, int) or max_conn <= 0):
            issues.append("max_connections must be a positive integer")

        return issues

    def export_config(self, category=None):
        """Export configuration as a formatted string"""
        if category:
            data = {category: self.config.get(category, {})}
        else:
            data = self.config

        result = []
        for cat, settings in data.items():
            result.append(f"[{cat.upper()}]")
            for key, value in settings.items():
                result.append(f"{key} = {value}")
            result.append("")  # Empty line between categories

        return "\n".join(result)

# Usage
config = ConfigManager()

# Get individual settings
print("Database host:", config.get_setting('database', 'host'))
print("Debug mode:", config.get_setting('app_settings', 'debug_mode'))
print("Unknown setting:", config.get_setting('unknown', 'key', 'default_value'))

# Update settings
config.set_setting('database', 'password', 'secret123')
config.set_setting('app_settings', 'debug_mode', False)

# Validate configuration
issues = config.validate_config()
if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid!")

# Export configuration
print("\nDatabase Configuration:")
print(config.export_config('database'))
```

---

## Best Practices

### 1. Use Clear and Meaningful Keys

```python
# ❌ Poor key names
data = {'a': 'John', 'b': 25, 'c': 'Engineer'}

# ✅ Clear key names
person = {'name': 'John', 'age': 25, 'occupation': 'Engineer'}
```

### 2. Use get() for Safe Access

```python
person = {'name': 'Alice', 'age': 25}

# ❌ Can raise KeyError
# salary = person['salary']

# ✅ Safe access with default
salary = person.get('salary', 0)
annual_income = person.get('salary', 50000) * 12
```

### 3. Use Dictionary Comprehensions for Simple Transformations

```python
numbers = [1, 2, 3, 4, 5]

# ❌ Verbose approach
squares = {}
for num in numbers:
    squares[num] = num ** 2

# ✅ Concise comprehension
squares = {num: num ** 2 for num in numbers}
```

### 4. Use setdefault() for Default Values

```python
# ❌ Manual checking
if 'categories' not in data:
    data['categories'] = []
data['categories'].append('new_category')

# ✅ Using setdefault
data.setdefault('categories', []).append('new_category')
```

### 5. Use items() for Key-Value Iteration

```python
person = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# ❌ Less efficient
for key in person:
    value = person[key]
    print(f"{key}: {value}")

# ✅ More efficient
for key, value in person.items():
    print(f"{key}: {value}")
```

---

## Performance Tips

### 1. Dictionary Lookups are O(1)

```python
# ✅ Fast lookup using dictionary
user_permissions = {
    'admin': ['read', 'write', 'delete'],
    'editor': ['read', 'write'],
    'viewer': ['read']
}

def check_permission(user_role, action):
    return action in user_permissions.get(user_role, [])

# ❌ Slower alternative using lists
user_list = [
    ('admin', ['read', 'write', 'delete']),
    ('editor', ['read', 'write']),
    ('viewer', ['read'])
]
```

### 2. Use dict.get() Instead of Exception Handling

```python
person = {'name': 'Alice', 'age': 25}

# ❌ Slower with exception handling
try:
    salary = person['salary']
except KeyError:
    salary = 0

# ✅ Faster with get()
salary = person.get('salary', 0)
```

### 3. Prefer Dictionary Comprehensions for Simple Cases

```python
data = ['apple', 'banana', 'cherry']

# ✅ Fast comprehension
lengths = {item: len(item) for item in data}

# ❌ Slower loop
lengths = {}
for item in data:
    lengths[item] = len(item)
```

---

## Common Pitfalls

### 1. Mutable Default Arguments

```python
# ❌ Dangerous - mutable default argument
def add_item(item, inventory={}):
    inventory[item] = inventory.get(item, 0) + 1
    return inventory

# This creates shared state between function calls!

# ✅ Safe approach
def add_item(item, inventory=None):
    if inventory is None:
        inventory = {}
    inventory[item] = inventory.get(item, 0) + 1
    return inventory
```

### 2. Modifying Dictionary During Iteration

```python
data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# ❌ Don't modify during iteration
# for key in data:
#     if data[key] % 2 == 0:
#         del data[key]  # RuntimeError!

# ✅ Safe approaches
# Option 1: Create list of keys first
for key in list(data.keys()):
    if data[key] % 2 == 0:
        del data[key]

# Option 2: Build new dictionary
data = {k: v for k, v in data.items() if v % 2 != 0}
```

### 3. Using Mutable Objects as Keys

```python
# ❌ Lists are mutable - can't be keys
# my_dict = {[1, 2]: 'value'}  # TypeError!

# ✅ Use immutable objects as keys
my_dict = {(1, 2): 'value'}  # Tuples are immutable
my_dict = {'key': 'value'}   # Strings are immutable
my_dict = {1: 'value'}       # Numbers are immutable
```

### 4. Assuming Dictionary Order (Pre-Python 3.7)

```python
# ⚠️ Only guaranteed in Python 3.7+
data = {'first': 1, 'second': 2, 'third': 3}

# If order matters and you need compatibility with older Python versions
from collections import OrderedDict
ordered_data = OrderedDict([('first', 1), ('second', 2), ('third', 3)])
```

---

## Advanced Patterns

### 1. Dictionary as Switch Statement

```python
def calculate(operation, x, y):
    operations = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / b if b != 0 else None
    }

    return operations.get(operation, lambda a, b: None)(x, y)

# Usage
print(calculate('add', 5, 3))      # 8
print(calculate('divide', 10, 2))  # 5.0
print(calculate('invalid', 1, 2))  # None
```

### 2. Caching with Dictionaries (Memoization)

```python
def fibonacci_with_cache():
    cache = {}

    def fibonacci(n):
        if n in cache:
            return cache[n]

        if n <= 1:
            result = n
        else:
            result = fibonacci(n-1) + fibonacci(n-2)

        cache[n] = result
        return result

    return fibonacci

# Usage
fib = fibonacci_with_cache()
print(fib(10))  # 55
print(fib(50))  # Very fast due to caching
```

### 3. Dictionary for State Machines

```python
class TrafficLight:
    def __init__(self):
        self.state = 'red'
        self.transitions = {
            'red': 'green',
            'green': 'yellow',
            'yellow': 'red'
        }
        self.durations = {
            'red': 30,
            'green': 25,
            'yellow': 5
        }

    def next_state(self):
        self.state = self.transitions[self.state]
        return self.state

    def get_duration(self):
        return self.durations[self.state]

    def get_status(self):
        return {
            'current_state': self.state,
            'duration': self.get_duration(),
            'next_state': self.transitions[self.state]
        }

# Usage
light = TrafficLight()
print(light.get_status())  # {'current_state': 'red', 'duration': 30, 'next_state': 'green'}
light.next_state()
print(light.get_status())  # {'current_state': 'green', 'duration': 25, 'next_state': 'yellow'}
```

---

## Conclusion

Dictionaries are one of Python's most powerful and versatile data structures. They provide:

- **Fast O(1) lookups** for efficient data access
- **Flexible key-value mapping** for representing relationships
- **Rich built-in methods** for manipulation and analysis
- **Memory efficiency** compared to other data structures
- **Readability** through meaningful key names

### Key Takeaways:

1. **Use dictionaries when you need fast lookups by key**
2. **Choose meaningful key names for better code readability**
3. **Use `get()` method for safe access with defaults**
4. **Leverage dictionary comprehensions for concise code**
5. **Be aware of mutable vs immutable key requirements**
6. **Consider `defaultdict` and `Counter` for specialized use cases**

### Next Steps:

- Practice with the examples provided
- Explore `collections` module for specialized dictionary types
- Learn about `dataclasses` as an alternative to dictionaries for structured data
- Study JSON handling, as it maps naturally to Python dictionaries
- Investigate database ORMs that use dictionary-like interfaces

Happy coding with Python dictionaries! 🐍📚

---

## Contributing

Feel free to contribute to this guide by:

- Adding more examples
- Improving explanations
- Fixing typos or errors
- Suggesting new sections

## License

This guide is open source and available under the MIT License.
