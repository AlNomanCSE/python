import numpy as np # type: ignore

new_list1 = list([x for x in range(1, 6)])
new_list2 = list([x for x in range(6, 11)])

a = np.array(new_list1)
b = np.array(new_list2)
print(new_list1)
print(type(a))
mixed = np.array([a, b])
print(mixed)


print(np.arange(12).reshape(4, 3))

print(np.std(mixed))
print(np.mean(mixed))

# a = np.array([[1], [2], [3]])
# b = np.array([10, 20, 30])
# print(a + b)

A = np.array([[1, 2], [3, 4]])
B = np.linalg.inv(A)

print(np.dot(A, B))
print("---------------")
np.random.seed(0)
print(np.random.rand(2, 4))
np.random.seed(3)
print(np.random.rand(2, 4))
print("🥴🥴🥴🥴🥴🥴🥴🥴")
print(mixed)

print(mixed[:1])
print(np.concatenate([a,b])[:2])

newArray = np.arange(12).reshape(4, 3);
print(newArray)

# *row
print(newArray[2,:])
# * column
print(newArray[:,2])

print(newArray[1:3,0:2])
print(newArray[2:,1:])

print(np.zeros((5,3)))

print(np.random.randint(10,100,6).reshape(3,2))

print(np.random.randn(5,6))