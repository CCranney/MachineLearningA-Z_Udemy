import numpy as np
import pandas as pd
import csv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
pd.set_option('chained_assignment',None) # removing a false positive error


def importTypes( typeFilepath ):
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
    binary = [ v for v in X.columns.to_list() if ( v not in numeric and v not in categorical ) ]

    # encoding categories, adjusting column names
    X = pd.get_dummies( X , columns = categorical )
    categorical = [ v for v in X.columns.to_list() if ( v not in numeric and v not in binary ) ]

    return numeric , binary , categorical , X

class MLParentRegression:
    def __init__( self , datasetFilepath , typeFilepath ):

        # import datasets
        X , y = self.importDatasets( datasetFilepath )

        # import types (for automation)
        xTypes , yType = importTypes( typeFilepath )

        # encoding binary/categorical data
        self.numeric , self.binary , self.categorical , X = encodeXCategories( X , xTypes )

        # cleaning missing data
        imputer = SimpleImputer( missing_values = np.nan , strategy = 'mean' )
        imputer.fit( X[ self.numeric ] )
        X.loc[ : , self.numeric ] = imputer.transform( X[ self.numeric ].values )

        # split the dataset into training and testing
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split( X , y , test_size = 0.2 , random_state = 0 )

        # removing false positive errors

        # perform Feature Scaling, if necessary
        self.featureScaling()

        # create model
        self.model = self.trainModel()

        # create predictions for test data based on model
        self.y_pred = self.testModel()

    def importDatasets( self , datasetFilepath ):
        dataset = pd.read_csv( datasetFilepath )
        return dataset.iloc[ : , :-1 ] , dataset.iloc[ : , -1 ]

    def trainModel( self ):
        return

    def featureScaling( self ):
        self.sc_X = StandardScaler()
        self.X_train.is_copy=False

        self.X_train.loc[ : , self.numeric ] = self.sc_X.fit_transform( self.X_train[ self.numeric ].values )

    def testModel( self ):
        return self.model.predict( self.X_test )

    def predTestPrintCompare( self ):
        np.set_printoptions( precision = 2 )
        print( np.concatenate( ( self.y_pred.reshape( len( self.y_pred ) , 1 ) , self.y_test.reshape( len( self.y_test ) , 1 ) ) , 1 ) )

    def evaluatePerformance( self ):
        return r2_score( self.y_test , self.y_pred )
