import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


data = pd.read_csv("Datasets/Salary_Data.csv")
X = data.iloc[ : , :-1 ].values
Y = data.iloc[ : , -1 ].values



print(X)
print(Y)

