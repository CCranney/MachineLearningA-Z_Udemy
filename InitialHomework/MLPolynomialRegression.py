from MLPredictiveModel import MLPredictiveModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class MLPolynomialRegression( MLPredictiveModel ):

    def trainModel( self ):
        self.poly_reg = PolynomialFeatures(degree=4)
        X_poly = self.poly_reg.fit_transform( self.X_train )
        regressor = LinearRegression()
        regressor.fit( X_poly , self.y_train )
        return regressor

    def testModel(self):
        return self.model.predict(self.poly_reg.transform(self.X_test))


plr = MLPolynomialRegression( "Datasets/Data.csv" , "Datasets/Data_types.csv" )
plr.predTestPrintCompare()
print(plr.evaluatePerformance())
