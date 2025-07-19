import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Iris.csv")

# list of columns
print(df.columns.values)
print(df.info())

# undestrand data shape
print(df.shape)
print(df["Species"].value_counts())

plt.figure(figsize=(10, 6))
plt.plot(df["SepalLengthCm"], label="Sepal Length")
plt.plot(df["SepalWidthCm"], label="Sepal Width")
plt.plot(df["PetalLengthCm"], label="Petal Length")
plt.plot(df["PetalWidthCm"], label="Petal Width")
plt.xlabel("Sample Index")
plt.ylabel("Masuremnet (cm)")
plt.legend()
plt.grid(True, alpha=.3)
plt.show()
