import copy

my_shopping_list = [3]
# my_shopping_list = ["milk", 3.50, "eggs"]
my_shopping_list.extend([1, 2, 34])

newListOne = copy.deepcopy(my_shopping_list)
newListTwo = my_shopping_list[:]
newListThree = list(my_shopping_list)
print("ID : ", id(my_shopping_list))
print("ID : ", id(newListOne))
print("ID : ", id(newListTwo))
print("ID : ", id(newListThree))
print("🚨 ", max(my_shopping_list))
