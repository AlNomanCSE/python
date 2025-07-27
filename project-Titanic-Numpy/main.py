import numpy as np

data = np.genfromtxt(
    "gender_submission.csv",
    delimiter=",",
    skip_header=1,
    usecols=('Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'),
    missing_values="",
    filling_values=np.nan,
)
data[:,2]
