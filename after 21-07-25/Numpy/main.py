import numpy as np

a = np.array([[1,2,3,4,5],[10,9,8,7,6]])
print(a.size)

b = np.arange(0,10).reshape(5,2);
print(b)

sqrt_numpy = np.sqrt(a);
print(type(sqrt_numpy>2))