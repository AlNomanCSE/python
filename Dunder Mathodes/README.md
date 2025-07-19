Dunder methods (double underscore methods) in Python, also known as magic methods or special methods, are used to define how objects of a class behave with respect to Python’s built-in operations. They are surrounded by double underscores (e.g., `__init__`, `__str__`). Below is a concise overview of the most important dunder methods, grouped by their purpose, along with simple code examples to illustrate their usage. I’ll keep the explanations brief and focus on practical examples, assuming you’re familiar with basic Python concepts.

### 1. **Object Initialization and Construction**
- **`__init__(self, ...)`**: Initializes a new instance of a class.
- **`__new__(cls, ...)`**: Creates a new instance (called before `__init__`).
- **`__del__(self)`**: Called when an object is about to be destroyed (garbage collection).

**Example**:
```python
class Person:
    def __new__(cls, name):
        print("Creating instance")
        return super().__new__(cls)
    
    def __init__(self, name):
        print("Initializing instance")
        self.name = name
    
    def __del__(self):
        print(f"{self.name} is being destroyed")

p = Person("Alice")  # Output: Creating instance, Initializing instance
del p  # Output: Alice is being destroyed
```

### 2. **String Representation**
- **`__str__(self)`**: Defines the string representation for `print()` or `str()`.
- **`__repr__(self)`**: Defines the “official” string representation, often for debugging.

**Example**:
```python
class Book:
    def __init__(self, title):
        self.title = title
    
    def __str__(self):
        return f"Book: {self.title}"
    
    def __repr__(self):
        return f"Book('{self.title}')"

book = Book("Python 101")
print(str(book))  # Output: Book: Python 101
print(repr(book))  # Output: Book('Python 101')
```

### 3. **Comparison Operators**
- **`__eq__(self, other)`**: Defines behavior for `==`.
- **`__lt__(self, other)`**: Defines behavior for `<`.
- **`__le__(self, other)`**: Defines behavior for `<=`.
- **`__gt__(self, other)`**: Defines behavior for `>`.
- **`__ge__(self, other)`**: Defines behavior for `>=`.
- **`__ne__(self, other)`**: Defines behavior for `!=`.

**Example**:
```python
class Number:
    def __init__(self, value):
        self.value = value
    
    def __eq__(self, other):
        return self.value == other.value
    
    def __lt__(self, other):
        return self.value < other.value

n1 = Number(5)
n2 = Number(10)
print(n1 == n2)  # Output: False
print(n1 < n2)   # Output: True
```

### 4. **Arithmetic Operations**
- **`__add__(self, other)`**: Defines behavior for `+`.
- **`__sub__(self, other)`**: Defines behavior for `-`.
- **`__mul__(self, other)`**: Defines behavior for `*`.
- **`__truediv__(self, other)`**: Defines behavior for `/`.
- **`__floordiv__(self, other)`**: Defines behavior for `//`.
- **`__mod__(self, other)`**: Defines behavior for `%`.

**Example**:
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2
print(v3)  # Output: Vector(4, 6)
```

### 5. **Attribute Access**
- **`__getattr__(self, name)`**: Called when an attribute is not found.
- **`__setattr__(self, name, value)`**: Called when an attribute is set.
- **`__delattr__(self, name)`**: Called when an attribute is deleted.

**Example**:
```python
class Dynamic:
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        return self._data.get(name, f"No attribute {name}")
    
    def __setattr__(self, name, value):
        self.__dict__["_data"] = {name: value}

d = Dynamic()
d.name = "Alice"
print(d.name)  # Output: Alice
print(d.age)   # Output: No attribute age
```

### 6. **Container Methods**
- **`__len__(self)`**: Defines behavior for `len()`.
- **`__getitem__(self, key)`**: Defines behavior for indexing (`obj[key]`).
- **`__setitem__(self, key, value)`**: Defines behavior for setting an item.
- **`__delitem__(self, key)`**: Defines behavior for deleting an item.
- **`__contains__(self, item)`**: Defines behavior for `in` operator.

**Example**:
```python
class MyList:
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value
    
    def __contains__(self, item):
        return item in self.items

ml = MyList([1, 2, 3])
print(len(ml))     # Output: 3
print(ml[0])       # Output: 1
ml[0] = 10
print(ml[0])       # Output: 10
print(2 in ml)     # Output: True
```

### 7. **Callable Objects**
- **`__call__(self, ...)`**: Allows an object to be called like a function.

**Example**:
```python
class Greeter:
    def __init__(self, greeting):
        self.greeting = greeting
    
    def __call__(self, name):
        return f"{self.greeting}, {name}!"

g = Greeter("Hello")
print(g("Alice"))  # Output: Hello, Alice!
```

### 8. **Context Managers**
- **`__enter__(self)`**: Defines what happens when entering a `with` block.
- **`__exit__(self, exc_type, exc_value, traceback)`**: Defines what happens when exiting a `with` block.

**Example**:
```python
class Resource:
    def __enter__(self):
        print("Resource acquired")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Resource released")

with Resource() as r:
    print("Using resource")
# Output:
# Resource acquired
# Using resource
# Resource released
```

### 9. **Iterator Methods**
- **`__iter__(self)`**: Returns an iterator object.
- **`__next__(self)`**: Defines the next item in iteration.

**Example**:
```python
class Counter:
    def __init__(self, max):
        self.max = max
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.max:
            self.current += 1
            return self.current
        raise StopIteration

c = Counter(3)
for num in c:
    print(num)  # Output: 1, 2, 3
```

### 10. **Other Useful Dunder Methods**
- **`__hash__(self)`**: Defines behavior for `hash()` (used in sets/dictionaries).
- **`__bool__(self)`**: Defines behavior for `bool()` (truth value).

**Example**:
```python
class Item:
    def __init__(self, name):
        self.name = name
    
    def __hash__(self):
        return hash(self.name)
    
    def __bool__(self):
        return bool(self.name)

item = Item("Book")
print(hash(item))  # Output: Some hash value based on "Book"
print(bool(item))  # Output: True
item2 = Item("")
print(bool(item2))  # Output: False
```

### Notes
- **Why These Methods?**: These are the most commonly used dunder methods that allow you to customize object behavior for Python’s built-in operations. There are others (e.g., `__format__`, `__copy__`), but these cover the core functionality.
- **Using `uv` on Your MacBook Air**: Since you mentioned using `uv`, ensure your virtual environment is active (`source my_env/bin/activate`) before running these examples. The libraries (`pandas`, `matplotlib`, `numpy`) from your previous query aren’t needed here, as these examples use only Python’s standard library.
- **Running the Code**: Save any example as, e.g., `dunder.py` and run it with:
  ```bash
  uv run python dunder.py
  ```
  This ensures the script runs in your `uv`-managed virtual environment.
- **Customization**: If you want examples tailored to a specific use case (e.g., integrating with your Iris dataset from the previous query), let me know, and I can provide a more targeted example (e.g., a custom class for Iris data with dunder methods).

If you have questions about specific dunder methods or want more examples, feel free to ask!