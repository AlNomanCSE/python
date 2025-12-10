"""
PYTHON NUMBERS - COMPLETE GUIDE
================================
This file contains comprehensive examples of working with numbers in Python.
"""

# ============================================================================
# 1. TYPES OF NUMBERS IN PYTHON
# ============================================================================

print("=" * 60)
print("1. TYPES OF NUMBERS")
print("=" * 60)

# Integer (int) - whole numbers
age = 25
temperature = -5
big_number = 1000000
print(f"Integer examples: {age}, {temperature}, {big_number}")
print(f"Type: {type(age)}\n")

# Float - decimal numbers
price = 19.99
pi = 3.14159
negative_float = -7.5
print(f"Float examples: {price}, {pi}, {negative_float}")
print(f"Type: {type(price)}\n")

# Complex numbers (rarely used in basic programming)
complex_num = 3 + 4j
print(f"Complex number: {complex_num}")
print(f"Type: {type(complex_num)}\n")


# ============================================================================
# 2. BASIC ARITHMETIC OPERATIONS
# ============================================================================

print("=" * 60)
print("2. BASIC ARITHMETIC OPERATIONS")
print("=" * 60)

a = 10
b = 3

# Addition
result_add = a + b
print(f"{a} + {b} = {result_add}")

# Subtraction
result_sub = a - b
print(f"{a} - {b} = {result_sub}")

# Multiplication
result_mul = a * b
print(f"{a} * {b} = {result_mul}")

# Division (always returns float)
result_div = a / b
print(f"{a} / {b} = {result_div}")

# Floor Division (returns integer, rounds down)
result_floor = a // b
print(f"{a} // {b} = {result_floor}")

# Modulus (remainder)
result_mod = a % b
print(f"{a} % {b} = {result_mod}")

# Exponentiation (power)
result_pow = a**b
print(f"{a} ** {b} = {result_pow}\n")


# ============================================================================
# 3. NUMBER CONVERSION
# ============================================================================

print("=" * 60)
print("3. NUMBER CONVERSION")
print("=" * 60)

# String to Integer
str_num = "42"
int_num = int(str_num)
print(f"String '{str_num}' to int: {int_num} (type: {type(int_num)})")

# String to Float
str_float = "3.14"
float_num = float(str_float)
print(f"String '{str_float}' to float: {float_num} (type: {type(float_num)})")

# Integer to Float
int_to_float = float(10)
print(f"Int 10 to float: {int_to_float} (type: {type(int_to_float)})")

# Float to Integer (truncates decimal part)
float_to_int = int(9.99)
print(f"Float 9.99 to int: {float_to_int} (type: {type(float_to_int)})")

# Number to String
num_to_str = str(123)
print(f"Number 123 to string: '{num_to_str}' (type: {type(num_to_str)})\n")


# ============================================================================
# 4. USEFUL BUILT-IN FUNCTIONS
# ============================================================================

print("=" * 60)
print("4. USEFUL BUILT-IN FUNCTIONS")
print("=" * 60)

numbers = [5, -3, 12, 8, -1, 0, 15]

# Absolute value
print(f"Absolute value of -42: {abs(-42)}")

# Round
print(f"Round 3.14159 to 2 decimals: {round(3.14159, 2)}")
print(f"Round 7.5: {round(7.5)}")

# Max and Min
print(f"Maximum of {numbers}: {max(numbers)}")
print(f"Minimum of {numbers}: {min(numbers)}")

# Sum
print(f"Sum of {numbers}: {sum(numbers)}")

# Power (alternative to **)
print(f"pow(2, 8): {pow(2, 8)}\n")


# ============================================================================
# 5. MATH MODULE - ADVANCED OPERATIONS
# ============================================================================

print("=" * 60)
print("5. MATH MODULE")
print("=" * 60)

import math

# Square root
print(f"Square root of 16: {math.sqrt(16)}")

# Ceiling and Floor
print(f"Ceiling of 4.3: {math.ceil(4.3)}")
print(f"Floor of 4.9: {math.floor(4.9)}")

# Trigonometric functions
print(f"Sin of 90 degrees: {math.sin(math.radians(90))}")
print(f"Cos of 0 degrees: {math.cos(math.radians(0))}")

# Constants
print(f"Pi: {math.pi}")
print(f"Euler's number (e): {math.e}")

# Factorial
print(f"Factorial of 5: {math.factorial(5)}")

# Logarithm
print(f"Natural log of 10: {math.log(10)}")
print(f"Log base 10 of 100: {math.log10(100)}\n")


# ============================================================================
# 6. COMPARISON OPERATORS
# ============================================================================

print("=" * 60)
print("6. COMPARISON OPERATORS")
print("=" * 60)

x = 10
y = 20

print(f"x = {x}, y = {y}")
print(f"x == y: {x == y}")  # Equal to
print(f"x != y: {x != y}")  # Not equal to
print(f"x > y: {x > y}")  # Greater than
print(f"x < y: {x < y}")  # Less than
print(f"x >= y: {x >= y}")  # Greater than or equal to
print(f"x <= y: {x <= y}\n")  # Less than or equal to


# ============================================================================
# 7. ASSIGNMENT OPERATORS
# ============================================================================

print("=" * 60)
print("7. ASSIGNMENT OPERATORS")
print("=" * 60)

num = 10
print(f"Initial value: {num}")

num += 5  # Same as num = num + 5
print(f"After += 5: {num}")

num -= 3  # Same as num = num - 3
print(f"After -= 3: {num}")

num *= 2  # Same as num = num * 2
print(f"After *= 2: {num}")

num /= 4  # Same as num = num / 4
print(f"After /= 4: {num}")

num //= 2  # Same as num = num // 2
print(f"After //= 2: {num}")

num %= 3  # Same as num = num % 3
print(f"After %= 3: {num}")

num **= 3  # Same as num = num ** 3
print(f"After **= 3: {num}\n")


# ============================================================================
# 8. RANDOM NUMBERS
# ============================================================================

print("=" * 60)
print("8. RANDOM NUMBERS")
print("=" * 60)

import random

# Random float between 0 and 1
print(f"Random float [0, 1): {random.random()}")

# Random integer in a range
print(f"Random int [1, 10]: {random.randint(1, 10)}")

# Random choice from a list
choices = [10, 20, 30, 40, 50]
print(f"Random choice from {choices}: {random.choice(choices)}")

# Random float in a range
print(f"Random float [5.0, 10.0]: {random.uniform(5.0, 10.0)}\n")


# ============================================================================
# 9. NUMBER FORMATTING
# ============================================================================

print("=" * 60)
print("9. NUMBER FORMATTING")
print("=" * 60)

value = 1234.56789

# Basic formatting
print(f"Default: {value}")
print(f"2 decimal places: {value:.2f}")
print(f"No decimals: {value:.0f}")

# Thousands separator
big_num = 1000000
print(f"With comma separator: {big_num:,}")

# Percentage
percentage = 0.856
print(f"As percentage: {percentage:.1%}")

# Scientific notation
scientific = 12345678
print(f"Scientific notation: {scientific:.2e}")

# Padding
num = 42
print(f"Padded with zeros (5 digits): {num:05d}")
print(f"Right aligned (10 spaces): {num:>10}")
print(f"Left aligned (10 spaces): {num:<10}\n")


# ============================================================================
# 10. PRACTICAL EXAMPLES
# ============================================================================

print("=" * 60)
print("10. PRACTICAL EXAMPLES")
print("=" * 60)

# Example 1: Calculate circle area
radius = 5
area = math.pi * radius**2
print(f"Circle with radius {radius} has area: {area:.2f}")

# Example 2: Temperature conversion
celsius = 25
fahrenheit = (celsius * 9 / 5) + 32
print(f"{celsius}°C = {fahrenheit}°F")

# Example 3: Calculate average
scores = [85, 92, 78, 90, 88]
average = sum(scores) / len(scores)
print(f"Average score: {average:.2f}")

# Example 4: Check if number is even or odd
number = 17
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")

# Example 5: Calculate compound interest
principal = 1000  # Initial amount
rate = 0.05  # 5% interest rate
time = 3  # 3 years
amount = principal * (1 + rate) ** time
print(f"Investment: ${principal}, Rate: {rate * 100}%, Time: {time} years")
print(f"Final amount: ${amount:.2f}")

# Example 6: Distance between two points
x1, y1 = 0, 0
x2, y2 = 3, 4
distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
print(f"Distance between ({x1},{y1}) and ({x2},{y2}): {distance:.2f}")


# ============================================================================
# 11. COMMON PITFALLS AND TIPS
# ============================================================================

print("\n" + "=" * 60)
print("11. COMMON PITFALLS AND TIPS")
print("=" * 60)

# Division always returns float
print(f"10 / 2 = {10 / 2} (type: {type(10 / 2)})")
print(f"10 // 2 = {10 // 2} (type: {type(10 // 2)})")

# Float precision issues
print(f"0.1 + 0.2 = {0.1 + 0.2} (not exactly 0.3!)")
print(f"Use round: {round(0.1 + 0.2, 1)}")

# Integer division with negative numbers
print(f"7 // 2 = {7 // 2}")
print(f"-7 // 2 = {-7 // 2} (rounds down to more negative)")

# Underscores in numbers (Python 3.6+)
million = 1_000_000
print(f"Readable large number: {million}")

print("\n" + "=" * 60)
print("END OF NUMBERS GUIDE")
print("=" * 60)
