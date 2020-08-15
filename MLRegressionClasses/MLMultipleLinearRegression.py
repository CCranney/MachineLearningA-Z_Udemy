from MLParentRegression import MLParentRegression
from sklearn.linear_model import LinearRegression


class MLMultipleLinearRegression( MLParentRegression ):

    def trainModel( self ):
        regressor = LinearRegression()
        regressor.fit( self.X_train , self.y_train )
        return regressor

    def featureScaling( self , n ):
        pass
