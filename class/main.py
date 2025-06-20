class Dog:
    def __init__(self, name, age, breed):
        # Instance attributes (unique to each object)
        self.name = name
        self.age = age
        self.breed = breed

    def brak(self):
        return f"{self.name} says Woof!"

    def info(self):
        return f"{self.name} is a {self.age} year old {self.breed}"


my_dog = Dog("Buddy", 3, "Golden Retriever")
your_dog = Dog("Max", 5, "German Shepherd")

print(my_dog.info())
