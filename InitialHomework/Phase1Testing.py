from MLMultipleLinearRegression import MLMultipleLinearRegression
from MLPolynomialRegression import MLPolynomialRegression
from MLSupportVectorRegression import MLSupportVectorRegression
from MLDecisionTreeRegression import MLDecisionTreeRegression
from MLRandomForestRegression import MLRandomForestRegression

mr = MLMultipleLinearRegression("Datasets/Data.csv", "Datasets/Data_types.csv")
pr = MLPolynomialRegression("Datasets/Data.csv", "Datasets/Data_types.csv")
sr = MLSupportVectorRegression("Datasets/Data.csv", "Datasets/Data_types.csv")
dr = MLDecisionTreeRegression("Datasets/Data.csv", "Datasets/Data_types.csv")
rr = MLRandomForestRegression("Datasets/Data.csv", "Datasets/Data_types.csv")
mrEval = mr.evaluatePerformance()
prEval = pr.evaluatePerformance()
srEval = sr.evaluatePerformance()
drEval = dr.evaluatePerformance()
rrEval = rr.evaluatePerformance()

template = "{0:30}{1:10}"
print(template.format("Multiple Linear Regression:" , str(mrEval)))
print(template.format("Polynomial Linear Regression:" , str(prEval)))
print(template.format("Support Vector Regression:" , str(srEval)))
print(template.format("Decision Tree Regression:" , str(drEval)))
print(template.format("Random Forest Regression:" , str(rrEval)))

