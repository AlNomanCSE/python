empty_set = {12, "banana", "orange"}
print(type(empty_set))

char_set = set("hello")
char_tuple = tuple("hello")
char_list = list("hello")
print(char_set, char_tuple, char_list)

# From list (duplicates removed automatically)
list_data = [1, 2, 2, 3, 3, 4]
unique_numbers = set(list_data)
print(unique_numbers)  # {1, 2, 3, 4}

# From tuple
tuple_data = (1, 2, 3, 2, 1)
set_from_tuple = set(tuple_data)
print(set_from_tuple)  # {1, 2, 3}

print(f"Union : {unique_numbers & set_from_tuple}")
print(f"Uniointersection : {unique_numbers | set_from_tuple}")
print(f"difference : {unique_numbers - set_from_tuple}")

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(set1 ^ set2)

set1 = {1, 2, 3}
set2 = {1, 2, 3, 4, 5}
set3 = {4, 5, 6}

print(set1.issubset(set2))
print(set1 <= set2)

print(set2 >= set1)
print(set2.issuperset(set1))

print(set1.isdisjoint(set2))
import math
print(abs(math.ceil(-123.12)))