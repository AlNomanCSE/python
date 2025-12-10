# import string

# myStr = " I love my country. "
# print(f"myStr : {len(myStr)}")
# print(f"{myStr.upper()}")
# myStr = myStr.strip()
# print(len(myStr))
# # myStr=myStr.split()
# print(type(myStr))
# myStr = myStr.replace("I", "you")
# print(myStr.find("L"))

# pi = 3.14159
# print(f"{myStr[::-1]}")

# my_tuple = (1, 2, "hello", 3.14)

# (_, two, three, *_) = my_tuple
# myone, my_two = "hello", "Abdullah Al Noman"

# newTuple = my_tuple + (30,);
# print(newTuple[:4]+my_tuple[::-1])

# list

my_shopping_list = [3]
my_shopping_list = ["milk", 3.50, "eggs"]
my_shopping_list.append(4)
my_shopping_list.extend([1, 2, 34])

print(my_shopping_list)
my_shopping_list.insert(4, "Abdullah al noman")
print(my_shopping_list)
data = [1, 2, 3, 2, 4]

try:
    my_shopping_list.sort()
    my_shopping_list.reverse()
    print(my_shopping_list)
    print("💕 length : ", len(my_shopping_list))
    print("🚨 index: ", my_shopping_list)
    new_List = my_shopping_list.copy()
    mylist = my_shopping_list
    print(new_List, mylist)
except ValueError as e:
    print("Error", str(e))
    print("Args", e.args)
    print("Name", type(e).__name__)
    print("class", e.__class__)
    print("Traceback", e.__traceback__)
