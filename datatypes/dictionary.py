user = {
    "id": 101,
    "name": "Alice",
    "role": "Admin",
    "id": 999,  # This overwrites the previous "id" key
}

print(user.get("name"))
print(type(user.keys()))
print(type(user.values()))
myList = list(user.items())
print(myList)

user.setdefault("d",6)
newDict = dict.fromkeys(["a","b"],100)
user.update(newDict)



del user["role"]
print("Admin" in user)
