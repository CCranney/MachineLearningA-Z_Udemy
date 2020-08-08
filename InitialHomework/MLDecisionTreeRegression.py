from MLPredictiveModel import MLPredictiveModel
from sklearn.tree import DecisionTreeRegressor


class MLDecisionTreeRegression( MLPredictiveModel ):

    def trainModel( self ):
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit( self.X_train, self.y_train )
        return regressor


dlr = MLDecisionTreeRegression( "Datasets/Data.csv" , "Datasets/Data_types.csv" )
dlr.predTestPrintCompare()
print(dlr.evaluatePerformance())
