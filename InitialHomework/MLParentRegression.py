import numpy as np
import pandas as pd
import csv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def importTypes(typeFilepath):
    with open( typeFilepath ) as csvFile: types = [ i for i in csv.reader( csvFile ) ][ 0 ]
    return types[ :-1 ] , types[ -1 ]


def encodeXCategories(X, xTypes):
    categories = [ i for i , value in enumerate( xTypes ) if value == 'categorical' ]
    ct = ColumnTransformer( transformers = [ ( 'encoder' , OneHotEncoder() , categories ) ] , remainder = 'passthrough' )
    return np.array( ct.fit_transform( X ) )


class MLParentRegression:
    def __init__( self , datasetFilepath , typeFilepath ):

        # import datasets
        X , y = self.importDatasets( datasetFilepath )

        # import types (for automation)
        xTypes , yType = importTypes( typeFilepath )

        # encoding categorical data
        X = encodeXCategories( X , xTypes )

        # split the dataset into training and testing
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split( X , y , test_size = 0.2 , random_state = 0 )

        # perform Feature Scaling, if necessary
        self.featureScaling()

        # create model
        self.model = self.trainModel()

        # create predictions for test data based on model
        self.y_pred = self.testModel()

    def importDatasets( self , datasetFilepath ):
        dataset = pd.read_csv( datasetFilepath )
        return dataset.iloc[ : , :-1 ].values , dataset.iloc[ :, -1 ].values

    def trainModel( self ):
        return

    def featureScaling( self ):
        pass

    def testModel( self ):
        return self.model.predict(self.X_test)

    def predTestPrintCompare( self ):
        np.set_printoptions(precision=2)
        print(np.concatenate((self.y_pred.reshape(len(self.y_pred), 1), self.y_test.reshape(len(self.y_test), 1)), 1))

    def evaluatePerformance( self ):
        return r2_score( self.y_test , self.y_pred )