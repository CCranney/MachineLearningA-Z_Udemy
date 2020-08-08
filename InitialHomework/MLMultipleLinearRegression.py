from MLPredictiveModel import MLPredictiveModel

import numpy as np
from sklearn.linear_model import LinearRegression


class MLMultipleLinearRegression(MLPredictiveModel):

    def trainModel(self):
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def testModel(self):
        y_pred = self.model.predict(self.X_test)
        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred), 1), self.y_test.reshape(len(self.y_test), 1)), 1))


mlr = MLMultipleLinearRegression("Datasets/50_Startups.csv", "Datasets/50_Startup_categories.csv")
mlr.testModel()
