from MLPredictiveModel import MLPredictiveModel
from sklearn.linear_model import LinearRegression


class MLMultipleLinearRegression( MLPredictiveModel ):

    def trainModel( self ):
        regressor = LinearRegression()
        regressor.fit( self.X_train , self.y_train )
        return regressor


mlr = MLMultipleLinearRegression( "Datasets/Data.csv" , "Datasets/Data_types.csv" )
mlr.predTestPrintCompare()
print(mlr.evaluatePerformance())
