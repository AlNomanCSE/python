temperatures_f = [32, 68, 86, 104, 122]
temperatures_c = [(f - 32) * 5 / 9 for f in temperatures_f]
print(f"Celsius:{[round(temp,2) for temp in temperatures_c]}")

numbers = range(-5, 6)
positive_squire = [x**2 for x in numbers if x > 2]
print(positive_squire)

new_list = [**temperatures_f]
Mylist = [x for x in range(1, 11)]
print(Mylist)