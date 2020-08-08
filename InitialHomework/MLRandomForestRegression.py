from MLPredictiveModel import MLPredictiveModel
from sklearn.ensemble import RandomForestRegressor

class MLDecisionTreeRegression( MLPredictiveModel ):

    def trainModel( self ):
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(self.X_train, self.y_train)
        return regressor


dlr = MLDecisionTreeRegression( "Datasets/Data.csv" , "Datasets/Data_types.csv" )
dlr.testModel()
print(dlr.evaluatePerformance())
