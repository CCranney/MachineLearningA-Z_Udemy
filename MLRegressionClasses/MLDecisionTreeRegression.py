from MLParentRegression import MLParentRegression
from sklearn.tree import DecisionTreeRegressor


class MLDecisionTreeRegression( MLParentRegression ):

    def trainModel( self ):
        regressor = DecisionTreeRegressor( random_state = 0 )
        regressor.fit( self.X_train , self.y_train )
        return regressor

    def featureScaling( self ):
        pass
