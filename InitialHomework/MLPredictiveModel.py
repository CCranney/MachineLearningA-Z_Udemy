import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class MLPredictiveModel:
    def __init__( self , datasetFilepath , typeFilepath ):
        # dataset
        self.dataset = pd.read_csv( datasetFilepath )
        X = self.dataset.iloc[ : , :-1 ].values
        y = self.dataset.iloc[ :, -1 ].values

        # types
        with open( typeFilepath ) as csvFile: types = [ i for i in csv.reader( csvFile ) ][ 0 ]
        xTypes = types[ :-1 ]
        yType = types[ -1 ]

        # encoding categorical data
        categories = [ i for i , value in enumerate( xTypes ) if value == 'categorical' ]
        ct = ColumnTransformer( transformers = [ ( 'encoder' , OneHotEncoder() , [3] ) ] , remainder = 'passthrough' )
        X = np.array( ct.fit_transform( X ) )

        # split the dataset into training and testing
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split( X , y , test_size = 0.2 , random_state = 0 )

        # train model, using the trainModel() function as defined by the child
        self.model = self.trainModel()

    def trainModel( self ):
        return

    def testModel( self ):
        pass

    def visualize( self ):
        pass