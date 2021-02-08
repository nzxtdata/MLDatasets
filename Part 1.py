import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

# Load the wine dataset from scikit-learn
wine = load_wine()
# Convert the dataset into a Pandas DataFrame for simpler analysis
# Reference: (numpy.c_ — NumPy v1.19 Manual, 2020) Retrieved from 'https://numpy.org/doc/stable/reference/generated/numpy.c_.html'
df = pd.DataFrame(data=np.c_[wine['target'], wine['data']],
                  columns=['class'] + wine['feature_names'])

# Preview the first 10 rows in the dataset
print(df.head(10))
print("")

# Key aspects of the dataset
print("Features:", wine.feature_names)
print("Targets:", wine.target_names)
print("Total number of values:", df['class'].count())
print("Dimensions of the dataset:", df.shape)
print("")
print("Number of values for each class:\n", df.groupby('class').count())
print("")

# Summary statistics
print("Summary Statistics:")
print(df.describe())
print("")

# Check the data types for each column
print(df.dtypes)
print("")

# Check for null values
print(df.isnull().sum())
print("")

# Preparing the dataset to be used for further machine learning
X = df.drop(['class'], axis=1)
print(X.head())
Y = df.iloc[:, :1]
print(Y.head())