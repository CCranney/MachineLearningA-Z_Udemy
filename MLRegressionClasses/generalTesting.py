import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
#df = "Datasets/50_Startups.csv"
df = "Datasets/Data.csv"
#df = "Datasets/Churn_Modelling.csv"
#tf = "Datasets/50_Startup_categories.csv"
tf = "Datasets/Data_types.csv"
#tf = "Datasets/Churn_categories.csv"


def importDatasets( datasetFilepath ):
    dataset = pd.read_csv( datasetFilepath )
    return dataset.iloc[ : , :-1 ] , dataset.iloc[ :, -1 ]

def importTypes(typeFilepath):
    with open( typeFilepath ) as csvFile: types = [ i for i in csv.reader( csvFile ) ][ 0 ]
    return types[ :-1 ] , types[ -1 ]

def getTypes( col , xTypes , type ):
    return [ col[ i ] for i in range( len( xTypes ) ) if xTypes[ i ] == type ]

def encodeXCategories( X , xTypes ):
    # categorizing columns by type
    numeric = getTypes( X.columns.to_list() , xTypes , 'numeric' )
    binary = getTypes( X.columns.to_list() , xTypes , 'binary' )
    categorical = getTypes( X.columns.to_list() , xTypes , 'categorical' )

    # encoding binary, adjusting column names
    X = pd.get_dummies( X , columns = binary , drop_first = True )
    binary = [ v for v in X.columns.to_list() if (v not in numeric and v not in categorical) ]

    # encoding categories, adjusting column names
    X = pd.get_dummies( X , columns = categorical )
    categorical = [ v for v in X.columns.to_list() if (v not in numeric and v not in binary) ]

    return numeric , binary , categorical , X


#X , y , z = importDatasets( df )
#xTypes , yType = importTypes( tf )
#num , bin , cate , X = encodeXCategories(X,xTypes)
#print(X)

# Load data
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
     index=['cobra', 'viper', 'sidewinder'],
     columns=['max_speed', 'shield', 'extra'])
df.loc[:,['max_speed','shield']] = 20

print(df)
