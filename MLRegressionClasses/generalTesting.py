import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



datasetFilepath = "Datasets/50_Startups.csv"
dataset = pd.read_csv( datasetFilepath )
print(dataset)
def importTypes(typeFilepath):
    with open( typeFilepath ) as csvFile: types = [ i for i in csv.reader( csvFile ) ][ 0 ]
    return types[ :-1 ] , types[ -1 ]
xTypes, ytype = importTypes("Datasets/50_Startup_categories.csv")

categories = [i for i, value in enumerate(xTypes) if value == 'categorical']
X = dataset.iloc[ : , :-1 ]
dum = pd.get_dummies(X,columns=['State'])
