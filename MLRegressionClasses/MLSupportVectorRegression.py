from MLParentRegression import MLParentRegression
from sklearn.svm import SVR


class MLSupportVectorRegression( MLParentRegression ):

    def importDatasets( self , datasetFilepath ):
        X , y = super().importDatasets( datasetFilepath )
        y = y.reshape( len( y ) , 1 )
        return X , y

    def trainModel( self ):
        regressor = SVR(kernel='rbf')
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def testModel( self ):
        return self.sc_y.inverse_transform(self.model.predict(self.sc_X.transform(self.X_test)))
