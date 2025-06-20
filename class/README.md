# Python Object-Oriented Programming (OOP) Guide 🐍

Welcome to your comprehensive guide to Object-Oriented Programming in Python! This README will walk you through the fundamentals of classes, constructors, and the four pillars of OOP with practical examples.

## Table of Contents
- [What is a Class?](#what-is-a-class)
- [Constructors in Python](#constructors-in-python)
- [Types of Methods](#types-of-methods)
- [Decorators in OOP](#decorators-in-oop)
- [The Four Pillars of OOP](#the-four-pillars-of-oop)
  - [1. Encapsulation](#1-encapsulation)
  - [2. Inheritance](#2-inheritance)
  - [3. Polymorphism](#3-polymorphism)
  - [4. Abstraction](#4-abstraction)
- [Multiple Inheritance](#multiple-inheritance)
- [isinstance() and Type Checking](#isinstance-and-type-checking)
- [Practice Examples](#practice-examples)
- [Quick Reference](#quick-reference)

---

## What is a Class?

A **class** is like a blueprint or template for creating objects. Think of it as a cookie cutter - it defines the shape and properties, but you can make many cookies (objects) from the same cutter.

```python
class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    # Constructor method
    def __init__(self, name, age, breed):
        # Instance attributes (unique to each object)
        self.name = name
        self.age = age
        self.breed = breed
    
    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"
    
    def info(self):
        return f"{self.name} is a {self.age} year old {self.breed}"

# Creating objects (instances) from the class
my_dog = Dog("Buddy", 3, "Golden Retriever")
your_dog = Dog("Max", 5, "German Shepherd")

print(my_dog.bark())  # Output: Buddy says Woof!
print(your_dog.info())  # Output: Max is a 5 year old German Shepherd
```

---

## Constructors in Python

The **constructor** is a special method that gets called automatically when you create a new object. In Python, we use the `__init__` method as our constructor.

### How Constructors Work:

```python
class Person:
    def __init__(self, name, age):
        """
        Constructor method - called when creating a new Person object
        
        Args:
            name (str): The person's name
            age (int): The person's age
        """
        print(f"Creating a new Person object...")
        self.name = name  # Setting instance attribute
        self.age = age    # Setting instance attribute
        self.id = self.generate_id()  # You can call other methods too!
    
    def generate_id(self):
        import random
        return random.randint(1000, 9999)
    
    def introduce(self):
        return f"Hi, I'm {self.name}, {self.age} years old. My ID is {self.id}"

# When you create an object, __init__ is automatically called
person1 = Person("Alice", 25)  # Output: Creating a new Person object...
print(person1.introduce())     # Output: Hi, I'm Alice, 25 years old. My ID is 1234
```

### Constructor with Default Values:

```python
class Car:
    def __init__(self, make, model, year=2023, color="White"):
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.mileage = 0
    
    def drive(self, miles):
        self.mileage += miles
        return f"Drove {miles} miles. Total mileage: {self.mileage}"

# Different ways to create Car objects
car1 = Car("Toyota", "Camry")  # Uses default year and color
car2 = Car("Honda", "Civic", 2022, "Blue")  # Custom values
car3 = Car("Ford", "Mustang", color="Red")  # Mix of default and custom
```

---

## Types of Methods

Python classes can have three types of methods: **instance methods**, **class methods**, and **static methods**. Let's explore each one!

### Instance Methods (Regular Methods)

These are the most common methods that work with instance data. They automatically receive `self` as the first parameter.

```python
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Instance method - works with instance data"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        """Instance method - accesses instance attribute"""
        return self.history

calc = Calculator()
print(calc.add(5, 3))  # Output: 8
print(calc.get_history())  # Output: ['5 + 3 = 8']
```

### Class Methods

Class methods work with class data rather than instance data. They use the `@classmethod` decorator and receive `cls` (the class itself) as the first parameter.

```python
class Student:
    # Class attributes
    total_students = 0
    school_name = "Python High School"
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Student.total_students += 1
    
    @classmethod
    def get_total_students(cls):
        """Class method - works with class data"""
        return f"Total students enrolled: {cls.total_students}"
    
    @classmethod
    def set_school_name(cls, name):
        """Class method - modifies class data"""
        cls.school_name = name
        return f"School name changed to: {cls.school_name}"
    
    @classmethod
    def create_honor_student(cls, name, age):
        """Class method - alternative constructor"""
        student = cls(name, age)
        student.honor_status = True
        return student

# Usage
student1 = Student("Alice", 16)
student2 = Student("Bob", 17)

print(Student.get_total_students())  # Output: Total students enrolled: 2
print(Student.set_school_name("Advanced Python Academy"))

# Using class method as alternative constructor
honor_student = Student.create_honor_student("Charlie", 18)
print(f"{honor_student.name} is an honor student: {honor_student.honor_status}")
```

### Static Methods

Static methods don't work with instance or class data. They're like regular functions but belong to the class namespace. They use the `@staticmethod` decorator.

```python
class MathUtility:
    @staticmethod
    def is_prime(number):
        """Static method - doesn't need class or instance data"""
        if number < 2:
            return False
        for i in range(2, int(number ** 0.5) + 1):
            if number % i == 0:
                return False
        return True
    
    @staticmethod
    def factorial(n):
        """Static method - pure function"""
        if n <= 1:
            return 1
        return n * MathUtility.factorial(n - 1)
    
    @staticmethod
    def celsius_to_fahrenheit(celsius):
        """Static method - utility function"""
        return (celsius * 9/5) + 32
    
    @staticmethod
    def validate_email(email):
        """Static method - validation function"""
        return "@" in email and "." in email

# Usage - can call without creating an instance
print(MathUtility.is_prime(17))  # Output: True
print(MathUtility.factorial(5))  # Output: 120
print(MathUtility.celsius_to_fahrenheit(25))  # Output: 77.0
print(MathUtility.validate_email("user@example.com"))  # Output: True

# You can also call from an instance (but it's not recommended)
math_util = MathUtility()
print(math_util.is_prime(13))  # Output: True (but better to use MathUtility.is_prime(13))
```

### Comparison of Method Types

```python
class Example:
    class_var = "I'm a class variable"
    
    def __init__(self, value):
        self.instance_var = value
    
    def instance_method(self):
        """Can access both instance and class variables"""
        return f"Instance: {self.instance_var}, Class: {self.class_var}"
    
    @classmethod
    def class_method(cls):
        """Can access class variables, but not instance variables"""
        return f"Class method accessing: {cls.class_var}"
    
    @staticmethod
    def static_method():
        """Cannot access class or instance variables directly"""
        return "Static method - independent of class/instance data"

# Demonstration
obj = Example("instance value")

print(obj.instance_method())  # Instance: instance value, Class: I'm a class variable
print(Example.class_method())  # Class method accessing: I'm a class variable
print(Example.static_method())  # Static method - independent of class/instance data
```

---

## Decorators in OOP

Decorators are a powerful feature that allows you to modify or enhance functions and methods. Here are common decorators used in OOP:

### Property Decorator

The `@property` decorator allows you to define methods that can be accessed like attributes.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Getter method"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Setter method with validation"""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        """Computed property"""
        return 3.14159 * self._radius ** 2
    
    @property
    def diameter(self):
        """Computed property"""
        return 2 * self._radius

# Usage
circle = Circle(5)
print(circle.radius)    # Output: 5 (using getter)
print(circle.area)      # Output: 78.53975 (computed)
print(circle.diameter)  # Output: 10 (computed)

circle.radius = 3       # Using setter
print(circle.area)      # Output: 28.274309999999996 (automatically updated)

# circle.radius = -1    # Would raise ValueError
```

### Custom Decorators

You can create your own decorators for OOP:

```python
def log_method_calls(func):
    """Decorator to log method calls"""
    def wrapper(self, *args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(self, *args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

def validate_positive(func):
    """Decorator to validate that arguments are positive"""
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError(f"Argument {arg} must be positive")
        return func(self, *args, **kwargs)
    return wrapper

class BankAccount:
    def __init__(self, balance=0):
        self._balance = balance
    
    @log_method_calls
    @validate_positive
    def deposit(self, amount):
        self._balance += amount
        return self._balance
    
    @log_method_calls
    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        return self._balance
    
    @property
    def balance(self):
        return self._balance

# Usage
account = BankAccount(100)
account.deposit(50)   # Logs the method call and validates positive amount
# Output: Calling deposit with args: (50,), kwargs: {}
#         deposit returned: 150

account.withdraw(25)  # Logs the method call
# Output: Calling withdraw with args: (25,), kwargs: {}
#         withdraw returned: 125
```

---

## The Four Pillars of OOP

### 1. Encapsulation 🔒

**Encapsulation** means bundling data (attributes) and methods that work on that data within a single unit (class), and controlling access to them.

```python
class BankAccount:
    def __init__(self, account_holder, initial_balance=0):
        self.account_holder = account_holder
        self._balance = initial_balance  # Protected attribute (convention)
        self.__account_number = self._generate_account_number()  # Private attribute
    
    def _generate_account_number(self):
        """Protected method - internal use"""
        import random
        return f"ACC{random.randint(100000, 999999)}"
    
    def deposit(self, amount):
        """Public method to deposit money"""
        if amount > 0:
            self._balance += amount
            return f"Deposited ${amount}. New balance: ${self._balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        """Public method to withdraw money"""
        if 0 < amount <= self._balance:
            self._balance -= amount
            return f"Withdrew ${amount}. New balance: ${self._balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        """Public method to check balance"""
        return f"Current balance: ${self._balance}"
    
    def get_account_info(self):
        """Public method to get account info"""
        return f"Account Holder: {self.account_holder}, Account Number: {self.__account_number}"

# Usage
account = BankAccount("John Doe", 1000)
print(account.deposit(500))        # Output: Deposited $500. New balance: $1500
print(account.withdraw(200))       # Output: Withdrew $200. New balance: $1300
print(account.get_balance())       # Output: Current balance: $1300

# Direct access to private attributes is discouraged
# print(account.__account_number)  # This would cause an AttributeError
```

### 2. Inheritance 👨‍👩‍👧‍👦

**Inheritance** allows a class to inherit attributes and methods from another class. The child class can use, modify, or extend the parent class's functionality.

```python
# Parent class (Base class)
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.is_alive = True
    
    def eat(self):
        return f"{self.name} is eating"
    
    def sleep(self):
        return f"{self.name} is sleeping"
    
    def make_sound(self):
        return f"{self.name} makes a sound"

# Child class (Derived class)
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Canine")  # Call parent constructor
        self.breed = breed
    
    # Override parent method
    def make_sound(self):
        return f"{self.name} barks: Woof! Woof!"
    
    # New method specific to Dog
    def fetch(self):
        return f"{self.name} is fetching the ball"

class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, "Feline")
        self.is_indoor = indoor
    
    # Override parent method
    def make_sound(self):
        return f"{self.name} meows: Meow!"
    
    # New method specific to Cat
    def climb(self):
        return f"{self.name} is climbing"

# Usage
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", indoor=True)

print(dog.eat())         # Inherited method: Buddy is eating
print(dog.make_sound())  # Overridden method: Buddy barks: Woof! Woof!
print(dog.fetch())       # Dog-specific method: Buddy is fetching the ball

print(cat.sleep())       # Inherited method: Whiskers is sleeping
print(cat.make_sound())  # Overridden method: Whiskers meows: Meow!
print(cat.climb())       # Cat-specific method: Whiskers is climbing
```

### 3. Polymorphism 🎭

**Polymorphism** means "many forms" - the same method name can behave differently depending on the object that calls it.

```python
class Shape:
    def __init__(self, name):
        self.name = name
    
    def area(self):
        pass
    
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

class Triangle(Shape):
    def __init__(self, base, height, side1, side2, side3):
        super().__init__("Triangle")
        self.base = base
        self.height = height
        self.sides = [side1, side2, side3]
    
    def area(self):
        return 0.5 * self.base * self.height
    
    def perimeter(self):
        return sum(self.sides)

# Polymorphism in action - same method, different behavior
shapes = [
    Rectangle(5, 3),
    Circle(4),
    Triangle(6, 4, 5, 6, 7)
]

print("Shape calculations:")
for shape in shapes:
    print(f"{shape.name}:")
    print(f"  Area: {shape.area()}")
    print(f"  Perimeter: {shape.perimeter()}")
    print()

# Output:
# Rectangle:
#   Area: 15
#   Perimeter: 16
# Circle:
#   Area: 50.26544
#   Perimeter: 25.13272
# Triangle:
#   Area: 12.0
#   Perimeter: 18
```

### 4. Abstraction 🎨

**Abstraction** hides complex implementation details and shows only the essential features of an object. In Python, we can use abstract base classes.

```python
from abc import ABC, abstractmethod

# Abstract base class
class Vehicle(ABC):
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.is_running = False
    
    @abstractmethod
    def start_engine(self):
        """Abstract method - must be implemented by child classes"""
        pass
    
    @abstractmethod
    def stop_engine(self):
        """Abstract method - must be implemented by child classes"""
        pass
    
    # Concrete method (can be used as-is by child classes)
    def honk(self):
        return "Beep! Beep!"
    
    def get_info(self):
        return f"{self.make} {self.model}"

class Car(Vehicle):
    def __init__(self, make, model, fuel_type="Gasoline"):
        super().__init__(make, model)
        self.fuel_type = fuel_type
    
    def start_engine(self):
        if not self.is_running:
            self.is_running = True
            return f"{self.get_info()} engine started with {self.fuel_type}"
        return "Engine is already running"
    
    def stop_engine(self):
        if self.is_running:
            self.is_running = False
            return f"{self.get_info()} engine stopped"
        return "Engine is already off"

class ElectricCar(Vehicle):
    def __init__(self, make, model, battery_capacity):
        super().__init__(make, model)
        self.battery_capacity = battery_capacity
        self.charge_level = 100
    
    def start_engine(self):
        if self.charge_level > 0 and not self.is_running:
            self.is_running = True
            return f"{self.get_info()} electric motor started silently"
        elif self.charge_level == 0:
            return "Cannot start - battery is empty"
        return "Motor is already running"
    
    def stop_engine(self):
        if self.is_running:
            self.is_running = False
            return f"{self.get_info()} electric motor stopped"
        return "Motor is already off"
    
    def charge_battery(self):
        self.charge_level = 100
        return "Battery fully charged!"

# Usage
car = Car("Toyota", "Camry")
electric_car = ElectricCar("Tesla", "Model 3", 75)

print(car.start_engine())          # Toyota Camry engine started with Gasoline
print(electric_car.start_engine()) # Tesla Model 3 electric motor started silently
print(car.honk())                  # Beep! Beep! (inherited method)

# You cannot create an instance of the abstract Vehicle class directly
# vehicle = Vehicle("Generic", "Car")  # This would raise TypeError
```

---

## Multiple Inheritance

**Multiple inheritance** allows a class to inherit from more than one parent class. Python supports this feature, but it should be used carefully.

### Basic Multiple Inheritance

```python
class Flyable:
    def __init__(self):
        self.can_fly = True
    
    def fly(self):
        return "Flying through the air!"
    
    def land(self):
        return "Landing safely"

class Swimmable:
    def __init__(self):
        self.can_swim = True
    
    def swim(self):
        return "Swimming in the water!"
    
    def dive(self):
        return "Diving underwater"

class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def eat(self):
        return f"{self.name} is eating"
    
    def sleep(self):
        return f"{self.name} is sleeping"

# Multiple inheritance - Duck inherits from all three classes
class Duck(Animal, Flyable, Swimmable):
    def __init__(self, name):
        # Call constructors of all parent classes
        Animal.__init__(self, name, "Duck")
        Flyable.__init__(self)
        Swimmable.__init__(self)
    
    def quack(self):
        return f"{self.name} says: Quack! Quack!"

# Usage
duck = Duck("Donald")
print(duck.eat())    # From Animal: Donald is eating
print(duck.fly())    # From Flyable: Flying through the air!
print(duck.swim())   # From Swimmable: Swimming in the water!
print(duck.quack())  # From Duck: Donald says: Quack! Quack!
print(f"Can fly: {duck.can_fly}, Can swim: {duck.can_swim}")  # True, True
```

### Method Resolution Order (MRO)

When multiple parent classes have methods with the same name, Python uses the **Method Resolution Order** to determine which method to call.

```python
class A:
    def method(self):
        return "Method from A"

class B:
    def method(self):
        return "Method from B"

class C(A, B):  # C inherits from both A and B
    pass

class D(B, A):  # D inherits from both B and A (different order)
    pass

# Check Method Resolution Order
print(C.__mro__)  # Shows the order: C -> A -> B -> object
print(D.__mro__)  # Shows the order: D -> B -> A -> object

obj_c = C()
obj_d = D()

print(obj_c.method())  # Output: "Method from A" (A comes first in MRO)
print(obj_d.method())  # Output: "Method from B" (B comes first in MRO)
```

### Diamond Problem and Super()

The diamond problem occurs when a class inherits from two classes that have a common parent. Use `super()` to handle this properly.

```python
class Vehicle:
    def __init__(self, brand):
        print(f"Vehicle.__init__ called with brand: {brand}")
        self.brand = brand
    
    def start(self):
        return f"{self.brand} vehicle starting"

class Car(Vehicle):
    def __init__(self, brand, doors):
        print(f"Car.__init__ called")
        super().__init__(brand)  # Call Vehicle.__init__
        self.doors = doors
    
    def start(self):
        return f"{self.brand} car starting with {self.doors} doors"

class Boat(Vehicle):
    def __init__(self, brand, length):
        print(f"Boat.__init__ called")
        super().__init__(brand)  # Call Vehicle.__init__
        self.length = length
    
    def start(self):
        return f"{self.brand} boat starting, length: {self.length}ft"

# Amphibious vehicle inherits from both Car and Boat
class AmphibiousVehicle(Car, Boat):
    def __init__(self, brand, doors, length):
        print(f"AmphibiousVehicle.__init__ called")
        # Use super() to properly handle multiple inheritance
        Car.__init__(self, brand, doors)
        Boat.__init__(self, brand, length)
    
    def start(self):
        return f"{self.brand} amphibious vehicle starting (doors: {self.doors}, length: {self.length}ft)"

# Usage
amphibious = AmphibiousVehicle("Amphicar", 2, 15)
print(amphibious.start())
print(f"MRO: {AmphibiousVehicle.__mro__}")
```

### Mixin Classes

**Mixins** are classes designed to be inherited from, providing specific functionality without being a complete class on their own.

```python
class TimestampMixin:
    """Mixin to add timestamp functionality"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import datetime
        self.created_at = datetime.datetime.now()
    
    def get_age(self):
        import datetime
        return datetime.datetime.now() - self.created_at

class SerializableMixin:
    """Mixin to add serialization functionality"""
    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)

class User(TimestampMixin, SerializableMixin):
    def __init__(self, username, email):
        super().__init__()
        self.username = username
        self.email = email
    
    def __str__(self):
        return f"User: {self.username} ({self.email})"

# Usage
user = User("john_doe", "john@example.com")
print(user)  # User: john_doe (john@example.com)
print(f"Account age: {user.get_age()}")  # Account age: 0:00:00.000123
print(f"User data: {user.to_dict()}")  # User data: {'created_at': datetime..., 'username': 'john_doe', 'email': 'john@example.com'}
```

---

## isinstance() and Type Checking

The `isinstance()` function is used to check if an object is an instance of a particular class or any of its parent classes.

### Basic isinstance() Usage

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def bark(self):
        return f"{self.name} barks!"

class Cat(Animal):
    def meow(self):
        return f"{self.name} meows!"

# Create instances
dog = Dog("Buddy")
cat = Cat("Whiskers")
number = 42
text = "Hello"

# Check types
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True (Dog inherits from Animal)
print(isinstance(cat, Dog))     # False
print(isinstance(number, int))  # True
print(isinstance(text, str))    # True

# Check multiple types
print(isinstance(dog, (Dog, Cat)))     # True (dog is a Dog)
print(isinstance(cat, (Dog, Cat)))     # True (cat is a Cat)
print(isinstance(number, (int, float))) # True (number is an int)
```

### Type Checking in Practice

```python
class Shape:
    def __init__(self, name):
        self.name = name

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2

def calculate_total_area(shapes):
    """Calculate total area of different shapes"""
    total = 0
    for shape in shapes:
        if isinstance(shape, Shape):
            total += shape.area()
        else:
            print(f"Skipping {shape} - not a Shape")
    return total

def describe_shape(shape):
    """Describe a shape based on its type"""
    if isinstance(shape, Rectangle):
        return f"Rectangle: {shape.width} x {shape.height}, Area: {shape.area()}"
    elif isinstance(shape, Circle):
        return f"Circle: radius {shape.radius}, Area: {shape.area():.2f}"
    elif isinstance(shape, Shape):
        return f"Generic shape: {shape.name}"
    else:
        return f"Not a shape: {type(shape).__name__}"

# Usage
shapes = [
    Rectangle(5, 3),
    Circle(4),
    "not a shape",
    Rectangle(2, 8)
]

print("Shape descriptions:")
for shape in shapes:
    print(describe_shape(shape))

print(f"\nTotal area of valid shapes: {calculate_total_area(shapes)}")
```

### Type Checking with hasattr()

Sometimes you want to check if an object has certain attributes or methods rather than checking its type:

```python
class Flyable:
    def fly(self):
        return "Flying!"

class Bird(Flyable):
    def __init__(self, name):
        self.name = name

class Penguin:
    def __init__(self, name):
        self.name = name
    
    def swim(self):
        return f"{self.name} is swimming!"

def make_it_fly(animal):
    """Make an animal fly if it can"""
    # Duck typing - check if it has the method rather than the type
    if hasattr(animal, 'fly') and callable(getattr(animal, 'fly')):
        return animal.fly()
    else:
        return f"{animal.name} cannot fly"

# Usage
eagle = Bird("Eagle")
penguin = Penguin("Pingu")

print(make_it_fly(eagle))    # Flying!
print(make_it_fly(penguin))  # Pingu cannot fly

# Check what methods/attributes an object has
print(f"Eagle attributes: {[attr for attr in dir(eagle) if not attr.startswith('_')]}")
print(f"Penguin attributes: {[attr for attr in dir(penguin) if not attr.startswith('_')]}")
```

### Advanced Type Checking

```python
from typing import Union, List, Optional

class NumberProcessor:
    @staticmethod
    def process_number(value: Union[int, float]) -> str:
        """Process different types of numbers"""
        if isinstance(value, int):
            return f"Integer: {value}, Square: {value ** 2}"
        elif isinstance(value, float):
            return f"Float: {value:.2f}, Square Root: {value ** 0.5:.2f}"
        else:
            raise TypeError(f"Expected int or float, got {type(value).__name__}")
    
    @staticmethod
    def process_list(items: List[Union[int, float]]) -> List[str]:
        """Process a list of numbers"""
        results = []
        for item in items:
            if isinstance(item, (int, float)):
                results.append(NumberProcessor.process_number(item))
            else:
                results.append(f"Skipped: {item} (type: {type(item).__name__})")
        return results

# Usage
processor = NumberProcessor()
numbers = [5, 3.14, "hello", 7, 2.5]

results = processor.process_list(numbers)
for result in results:
    print(result)

# Output:
# Integer: 5, Square: 25
# Float: 3.14, Square Root: 1.77
# Skipped: hello (type: str)
# Integer: 7, Square: 49
# Float: 2.50, Square Root: 1.58
```

---

## Practice Examples

Here's a complete example that demonstrates all concepts:

```python
class Student:
    # Class attribute
    total_students = 0
    
    def __init__(self, name, age, student_id):
        # Instance attributes
        self.name = name
        self.age = age
        self.student_id = student_id
        self.grades = []
        self._gpa = 0.0  # Protected attribute
        
        # Increment class attribute
        Student.total_students += 1
        print(f"Student {self.name} enrolled! Total students: {Student.total_students}")
    
    def add_grade(self, subject, grade):
        """Add a grade for a subject"""
        if 0 <= grade <= 100:
            self.grades.append({"subject": subject, "grade": grade})
            self._calculate_gpa()
            return f"Grade {grade} added for {subject}"
        return "Invalid grade. Must be between 0 and 100"
    
    def _calculate_gpa(self):
        """Protected method to calculate GPA"""
        if self.grades:
            total = sum(grade["grade"] for grade in self.grades)
            self._gpa = total / len(self.grades) / 25  # Simple GPA calculation
    
    def get_gpa(self):
        """Get current GPA"""
        return round(self._gpa, 2)
    
    def get_transcript(self):
        """Get student transcript"""
        transcript = f"\n--- Transcript for {self.name} ---\n"
        transcript += f"Student ID: {self.student_id}\n"
        transcript += f"Age: {self.age}\n\n"
        
        if self.grades:
            transcript += "Grades:\n"
            for grade in self.grades:
                transcript += f"  {grade['subject']}: {grade['grade']}\n"
            transcript += f"\nGPA: {self.get_gpa()}\n"
        else:
            transcript += "No grades recorded yet.\n"
        
        return transcript

# Usage
student1 = Student("Alice Johnson", 20, "STU001")
student2 = Student("Bob Smith", 19, "STU002")

# Add grades
print(student1.add_grade("Mathematics", 95))
print(student1.add_grade("Physics", 88))
print(student1.add_grade("Chemistry", 92))

print(student2.add_grade("Mathematics", 78))
print(student2.add_grade("History", 85))

# Get transcripts
print(student1.get_transcript())
print(student2.get_transcript())

print(f"Total students enrolled: {Student.total_students}")
```

---

## Quick Reference

### Class Definition Syntax:
```python
class ClassName:
    # Class attributes
    class_variable = "shared by all instances"
    
    def __init__(self, parameter1, parameter2):
        """Constructor"""
        self.instance_variable1 = parameter1
        self.instance_variable2 = parameter2
    
    def method_name(self):
        """Instance method"""
        return "method result"
    
    @classmethod
    def class_method(cls):
        """Class method"""
        return "class method result"
    
    @staticmethod
    def static_method():
        """Static method"""
        return "static method result"
```

### The Four Pillars Summary:
1. **Encapsulation**: Bundle data and methods, control access
2. **Inheritance**: Child classes inherit from parent classes
3. **Polymorphism**: Same method name, different behaviors
4. **Abstraction**: Hide complexity, show only essential features

### Key Points to Remember:
- `self` refers to the current instance of the class
- `__init__` is the constructor method
- Methods defined in a class automatically receive `self` as the first parameter
- Use `super()` to call parent class methods
- Private attributes start with `__` (double underscore)
- Protected attributes start with `_` (single underscore)

---

Happy coding! 🚀 Remember, practice makes perfect. Try creating your own classes and experimenting with these concepts.