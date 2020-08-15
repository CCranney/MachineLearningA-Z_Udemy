from MLMultipleLinearRegression import MLMultipleLinearRegression
from MLPolynomialRegression import MLPolynomialRegression
from MLSupportVectorRegression import MLSupportVectorRegression
from MLDecisionTreeRegression import MLDecisionTreeRegression
from MLRandomForestRegression import MLRandomForestRegression
d = "50_Startups.csv"
#d = "Data.csv"
t = "50_Startup_categories.csv"
#t = "Data_types.csv"

mr = MLMultipleLinearRegression("Datasets/" + d, "Datasets/" + t)
pr = MLPolynomialRegression("Datasets/" + d, "Datasets/" + t)
sr = MLSupportVectorRegression("Datasets/" + d, "Datasets/" + t)
dr = MLDecisionTreeRegression("Datasets/" + d, "Datasets/" + t)
rr = MLRandomForestRegression("Datasets/" + d, "Datasets/" + t)
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
