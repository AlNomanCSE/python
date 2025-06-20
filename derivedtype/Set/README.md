# Python Sets - Complete Guide

## What is a Set?

A **Set** in Python is an unordered collection of unique elements. Sets are mutable (can be changed after creation) and do not allow duplicate values. They are similar to mathematical sets and are particularly useful when you need to store unique items and perform set operations like union, intersection, and difference.

## Key Characteristics

- **Unordered**: Elements have no defined order
- **Unique**: No duplicate elements allowed
- **Mutable**: Can add/remove elements after creation
- **Iterable**: Can loop through elements
- **No indexing**: Cannot access elements by index

## Declaration and Creation

### 1. Empty Set
```python
# Using set() constructor
empty_set = set()
print(empty_set)  # set()

# Note: {} creates an empty dictionary, not a set
```

### 2. Set with Initial Values
```python
# Using curly braces
fruits = {"apple", "banana", "orange"}
print(fruits)  # {'banana', 'apple', 'orange'}

# Using set() constructor with iterable
numbers = set([1, 2, 3, 4, 5])
print(numbers)  # {1, 2, 3, 4, 5}

# From string (each character becomes an element)
char_set = set("hello")
print(char_set)  # {'h', 'e', 'l', 'o'}
```

### 3. Set from Other Data Types
```python
# From list (duplicates removed automatically)
list_data = [1, 2, 2, 3, 3, 4]
unique_numbers = set(list_data)
print(unique_numbers)  # {1, 2, 3, 4}

# From tuple
tuple_data = (1, 2, 3, 2, 1)
set_from_tuple = set(tuple_data)
print(set_from_tuple)  # {1, 2, 3}
```

## Basic Operations

### Adding Elements
```python
fruits = {"apple", "banana"}

# Add single element
fruits.add("orange")
print(fruits)  # {'apple', 'banana', 'orange'}

# Add multiple elements
fruits.update(["grape", "mango"])
print(fruits)  # {'apple', 'banana', 'orange', 'grape', 'mango'}

# Update with another set
more_fruits = {"kiwi", "pear"}
fruits.update(more_fruits)
```

### Removing Elements
```python
fruits = {"apple", "banana", "orange"}

# Remove element (raises KeyError if not found)
fruits.remove("banana")
print(fruits)  # {'apple', 'orange'}

# Discard element (no error if not found)
fruits.discard("grape")  # No error even though 'grape' doesn't exist

# Pop random element
removed_fruit = fruits.pop()
print(f"Removed: {removed_fruit}")

# Clear all elements
fruits.clear()
print(fruits)  # set()
```

## Set Operations

### Union (|)
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# Using | operator
union_result = set1 | set2
print(union_result)  # {1, 2, 3, 4, 5}

# Using union() method
union_result = set1.union(set2)
print(union_result)  # {1, 2, 3, 4, 5}
```

### Intersection (&)
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Using & operator
intersection_result = set1 & set2
print(intersection_result)  # {3, 4}

# Using intersection() method
intersection_result = set1.intersection(set2)
print(intersection_result)  # {3, 4}
```

### Difference (-)
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Elements in set1 but not in set2
difference_result = set1 - set2
print(difference_result)  # {1, 2}

# Using difference() method
difference_result = set1.difference(set2)
print(difference_result)  # {1, 2}
```

### Symmetric Difference (^)
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Elements in either set, but not in both
sym_diff_result = set1 ^ set2
print(sym_diff_result)  # {1, 2, 5, 6}

# Using symmetric_difference() method
sym_diff_result = set1.symmetric_difference(set2)
print(sym_diff_result)  # {1, 2, 5, 6}
```

## Set Comparison Methods

```python
set1 = {1, 2, 3}
set2 = {1, 2, 3, 4, 5}
set3 = {4, 5, 6}

# Check if subset
print(set1.issubset(set2))  # True
print(set1 <= set2)         # True (alternative syntax)

# Check if superset
print(set2.issuperset(set1))  # True
print(set2 >= set1)           # True (alternative syntax)

# Check if disjoint (no common elements)
print(set1.isdisjoint(set3))  # True
```

## Why Use Sets?

### 1. **Remove Duplicates**
```python
# Remove duplicates from a list
numbers = [1, 2, 2, 3, 3, 4, 4, 5]
unique_numbers = list(set(numbers))
print(unique_numbers)  # [1, 2, 3, 4, 5]
```

### 2. **Fast Membership Testing**
```python
# O(1) average time complexity for membership testing
large_set = set(range(1000000))
print(999999 in large_set)  # Very fast lookup
```

### 3. **Mathematical Set Operations**
```python
# Find common interests
alice_interests = {"reading", "swimming", "coding"}
bob_interests = {"swimming", "gaming", "coding"}

common_interests = alice_interests & bob_interests
print(common_interests)  # {'swimming', 'coding'}
```

### 4. **Data Analysis and Filtering**
```python
# Find unique visitors across different days
monday_visitors = {"Alice", "Bob", "Charlie"}
tuesday_visitors = {"Bob", "David", "Eve"}

all_unique_visitors = monday_visitors | tuesday_visitors
returning_visitors = monday_visitors & tuesday_visitors
new_visitors = tuesday_visitors - monday_visitors

print(f"All unique visitors: {all_unique_visitors}")
print(f"Returning visitors: {returning_visitors}")
print(f"New visitors: {new_visitors}")
```

## Practical Examples

### Example 1: Finding Common Elements
```python
def find_common_elements(list1, list2):
    """Find common elements between two lists"""
    return list(set(list1) & set(list2))

students_math = ["Alice", "Bob", "Charlie", "David"]
students_science = ["Bob", "Charlie", "Eve", "Frank"]

common_students = find_common_elements(students_math, students_science)
print(common_students)  # ['Bob', 'Charlie']
```

### Example 2: Tag Management System
```python
class TagManager:
    def __init__(self):
        self.tags = set()
    
    def add_tags(self, new_tags):
        """Add multiple tags"""
        self.tags.update(new_tags)
    
    def remove_tag(self, tag):
        """Remove a tag safely"""
        self.tags.discard(tag)
    
    def has_tag(self, tag):
        """Check if tag exists"""
        return tag in self.tags
    
    def get_common_tags(self, other_tags):
        """Find common tags with another set"""
        return self.tags & set(other_tags)

# Usage
tag_manager = TagManager()
tag_manager.add_tags(["python", "programming", "tutorial"])
print(tag_manager.tags)  # {'python', 'programming', 'tutorial'}
```

## Set Comprehensions

```python
# Create set using comprehension
squares = {x**2 for x in range(10)}
print(squares)  # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

# With condition
even_squares = {x**2 for x in range(10) if x % 2 == 0}
print(even_squares)  # {0, 4, 16, 36, 64}
```

## Frozen Sets

```python
# Immutable version of set
frozen_fruits = frozenset(["apple", "banana", "orange"])
print(frozen_fruits)  # frozenset({'apple', 'banana', 'orange'})

# Can be used as dictionary keys (unlike regular sets)
fruit_colors = {
    frozenset(["apple", "cherry"]): "red",
    frozenset(["banana", "lemon"]): "yellow"
}
```

## Performance Considerations

| Operation | Average Time Complexity |
|-----------|------------------------|
| Add       | O(1)                   |
| Remove    | O(1)                   |
| Contains  | O(1)                   |
| Union     | O(len(s1) + len(s2))   |
| Intersection | O(min(len(s1), len(s2))) |

## Common Use Cases

1. **Removing duplicates from data**
2. **Finding unique elements**
3. **Set operations (union, intersection, difference)**
4. **Fast membership testing**
5. **Mathematical computations**
6. **Data filtering and analysis**
7. **Permission and role management**
8. **Tag systems**

## Best Practices

1. Use sets when you need unique elements
2. Prefer sets over lists for membership testing with large datasets
3. Use set operations instead of loops when possible
4. Remember that sets are unordered - don't rely on element order
5. Use frozenset when you need an immutable set
6. Consider memory usage for very large sets

## Conclusion

Python sets are powerful data structures that provide efficient ways to work with unique collections of data. They're essential for mathematical operations, data deduplication, and fast lookups. Understanding sets will make your Python code more efficient and elegant.

---

**Happy Coding! 🐍**