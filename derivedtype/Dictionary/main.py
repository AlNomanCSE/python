empty_dict = {}

person = {
    "Name": "Abdullah Al Noman",
    "Age": 29,
    "City": "Khulna",
}
print(person)
from pprint import pprint

mixed = {
    "string_key": "Hello",
    42: "number key",
    "list_value": [1, 2, 3],
    ("x", "y"): (12, 13),
    "nested_dict": {"inner": "value"},
}

person = dict(
    [
        ("Greetings", "Hello"),
        (42, "NUmberValue"),
        ("List Value", [1, 2, 3, 4]),
        (("x", "y"), (12, 13)),
        ("nested_dect", {"inner value": "NAN"}),
    ]
)


pairs = [("apple", 5), ("orange", 10), (("a", "b"), (12, 13))]
fruits = dict(pairs)

keys = ("name", "age", "city")
values = ("Charlie", 35, "Paris")
person = dict(zip(keys, values))
pprint(person, indent=4)
squres = {x: x**2 for x in range(1, 11) if x % 3 == 0}
print(squres)
words = ["hello", "world", "python"]
words_lenght = {word: len(word) for word in words}
pprint(words_lenght, indent=4)

person = {
    "name": "Alice",
    "age": 25,
    "city": "New York",
    "hobbies": ["reading", "swimming"],
}

print("name" in person)
if person.get("salary") is not None:
    print("salary exists")
else:
    person["salary"] = 25000
    print("salary" in person)
person.update(salary=30000, experience=1)
pprint(person)

del person["salary"]
person.pop("hobbies")
print(person.pop("salary","Not Found"))
pprint(person)
person.clear()
pprint(person)