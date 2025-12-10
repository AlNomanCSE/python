A = {1, 2, 3, 4}
B = {3, 4, 5, 6}


print("Union --->: ", A | B)
print("Intersection --->: ", A & B)
print("diffrence --->: ", A - B)
print("diffrence --->: ", B - A)
print(f"{1 in A}")
a = frozenset(A | B)
A.add(12)
B.update((7, 8, 9))
# B.remove(8)
# B.discard(10)
B.pop()
print(B)
