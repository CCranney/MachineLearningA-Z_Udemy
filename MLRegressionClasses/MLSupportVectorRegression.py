from MLParentRegression import MLParentRegression
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class MLSupportVectorRegression( MLParentRegression ):

    def trainModel( self ):
        self.sc_y = StandardScaler()
        self.y_train = self.y_train.values.reshape( len( self.y_train ) , 1 )
        self.y_train = self.sc_y.fit_transform( self.y_train )

        regressor = SVR( kernel = 'rbf' )
        regressor.fit( self.X_train , self.y_train.ravel() )
        return regressor

    def testModel( self ):
        self.X_test.is_copy = False
        self.X_test.loc[ : , self.numeric ] = self.sc_X.transform( self.X_test[ self.numeric ].values )
        y = pd.DataFrame( data = self.model.predict( self.X_test ) )
        return self.sc_y.inverse_transform( y.values )
