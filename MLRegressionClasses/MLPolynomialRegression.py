from MLParentRegression import MLParentRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd

def addDataFrames( dfList ):
    for v in dfList: v[ 'key' ] = 0
    df = pd.concat( dfList , axis=1 , join='outer' )
    return df.drop( columns = [ 'key' ] )

class MLPolynomialRegression( MLParentRegression ):

    def trainModel( self ):
        self.poly_reg = PolynomialFeatures( degree = 4 )
        X_poly = self.poly_reg.fit_transform( self.X_train )
        regressor = LinearRegression()
        regressor.fit( X_poly , self.y_train )
        return regressor

    def testModel(self):
        X_poly = pd.DataFrame( data = self.poly_reg.transform( self.X_test ) )
        return self.model.predict( X_poly )

    def featureScaling( self ):
        pass
