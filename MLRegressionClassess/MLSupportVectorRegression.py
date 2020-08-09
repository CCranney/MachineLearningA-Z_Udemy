from MLParentRegression import MLParentRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class MLSupportVectorRegression( MLParentRegression ):

    def importDatasets( self , datasetFilepath ):
        X , y = super().importDatasets( datasetFilepath )
        y = y.reshape( len( y ) , 1 )
        return X , y

    def featureScaling( self ):
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X_train = self.sc_X.fit_transform(self.X_train)
        self.y_train = self.sc_y.fit_transform(self.y_train)

    def trainModel( self ):
        regressor = SVR(kernel='rbf')
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def testModel(self):
        return self.sc_y.inverse_transform(self.model.predict(self.sc_X.transform(self.X_test)))
