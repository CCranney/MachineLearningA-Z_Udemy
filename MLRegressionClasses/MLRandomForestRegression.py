from MLParentRegression import MLParentRegression
from sklearn.ensemble import RandomForestRegressor

class MLRandomForestRegression( MLParentRegression ):

    def trainModel( self ):
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def featureScaling( self , n ):
        pass
