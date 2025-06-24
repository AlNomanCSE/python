Great! You've uploaded the **Iris dataset** — one of the most popular datasets for data visualization and machine learning. Let's build a **beginner to advanced guide for Matplotlib**, using **Iris.csv** every step of the way.

---

# 🌼 Matplotlib Guide (Beginner → Advanced) using the Iris Dataset

---

### 📦 Step 0: Load the Data

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('/mnt/data/Iris.csv')

# Preview the data
df.head()
```

---

## 🐣 Part 1: Beginner — Basic Plots

### ✅ Q1: What’s the distribution of species?

```python
import matplotlib.pyplot as plt

df['Species'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Number of Samples per Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
```

---

### ✅ Q2: What does a histogram of petal length look like?

```python
plt.hist(df['PetalLengthCm'], bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()
```

---

### ✅ Q3: How do I plot a simple scatter plot?

```python
plt.scatter(df['SepalLengthCm'], df['PetalLengthCm'], alpha=0.7)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()
```

---

## 🌱 Part 2: Intermediate — Grouped and Colored Plots

### ✅ Q4: Scatter Plot by Species (colored)

```python
species = df['Species'].unique()
colors = ['red', 'green', 'blue']

for i, sp in enumerate(species):
    subset = df[df['Species'] == sp]
    plt.scatter(subset['SepalLengthCm'], subset['PetalLengthCm'],
                label=sp, color=colors[i])

plt.title('Sepal vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()
```

---

### ✅ Q5: Boxplot of Sepal Width by Species

```python
df.boxplot(column='SepalWidthCm', by='Species', grid=False)
plt.title('Sepal Width by Species')
plt.suptitle('')  # Remove default title
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.show()
```

---

### ✅ Q6: Subplots (multiple charts together)

```python
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(df['SepalLengthCm'], color='skyblue', bins=15)
axs[0].set_title('Sepal Length Distribution')

axs[1].hist(df['PetalLengthCm'], color='lightcoral', bins=15)
axs[1].set_title('Petal Length Distribution')

plt.tight_layout()
plt.show()
```

---

## 🌳 Part 3: Advanced — Fine-tuning, Annotations, Advanced Layouts

### ✅ Q7: Annotated Scatter Plot

```python
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], color='teal', alpha=0.6)

# Annotate first 5 points
for i in range(5):
    plt.annotate(df['Species'][i], (df['SepalLengthCm'][i], df['SepalWidthCm'][i]))

plt.title('Sepal Dimensions with Annotations')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
```

---

### ✅ Q8: Pairwise Comparison with Pandas + Matplotlib

```python
pd.plotting.scatter_matrix(df.iloc[:, 1:5], figsize=(10, 10), diagonal='hist', alpha=0.8, color='purple')
plt.suptitle('Scatter Matrix of Features')
plt.show()
```

---

### ✅ Q9: Save plots to a file

```python
plt.hist(df['PetalWidthCm'], bins=15, color='green')
plt.title('Petal Width Histogram')
plt.savefig('petal_width_histogram.png')  # Saved to working directory
```

---

## 🚀 Bonus: Useful Styling Tricks

```python
plt.style.use('seaborn-v0_8-darkgrid')  # Try 'ggplot', 'classic', etc.
```

---

## 🎓 Final Challenges (for Practice)

1. Plot average petal width per species using a bar chart.
2. Create a violin plot (using `seaborn`) of petal length.
3. Make a subplot grid of all four features (2x2).

---

Would you like me to turn this into a **Jupyter notebook** or give **exercises** for each level to practice?
