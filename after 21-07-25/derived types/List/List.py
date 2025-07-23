planets = [
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]

short_planet = [planet for planet in planets if len(planet) < 6]
print(f"{short_planet}")

import random

L = [random.random() for i in range(10)]
data = [1, 2, 3]

data.append(4)
print(data)
data.extend([10, 11, 12])
print(data.pop())

print("Mars" in planets)
planets.sort(reverse=True)
print(planets)
dontKnow = [32 for planet in planets]
print(dontKnow)
# name[:3] = ["Alice", "Bob", "Charlie", "David",]
# print(name)
planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars',]
print(planets)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [x for x in numbers if x%2==0]
print(evens)