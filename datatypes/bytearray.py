
try:
    a = [10, 20, 30, 40, 50]
    b = bytes(a)
except ValueError as e:
    print(str(e))

data  = bytearray(b"Hello");
print(data)