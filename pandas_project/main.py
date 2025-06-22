import pandas as pd  # type: ignore
import numpy as np  # type: ignore

new_2Darray = np.arange(0, 20).reshape(5, 4)
print(new_2Darray)
rowLable = [f"Row-{i+1}" for i in range(new_2Darray.shape[0])]
columnLable = [f"Column*{i+1}" for i in range(new_2Darray.shape[1])]
df = pd.DataFrame(data=new_2Darray, index=rowLable, columns=columnLable)

print(type(df))
print(df.info())
print(df.describe())

print(df.head())
print(type(df[['Column*2','Column*1']]))
print(type(df['Column*1']))