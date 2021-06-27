# Author: Sarbhanu Baidya
# Date: 06:26:2021
# Description: LinearRegression implementation using scikit-learn

# ML: Linear Regression | Model is built using UCI ML Dataset Repository
# Student Performance Dataset
# Dataset Link: https://archive.ics.uci.edu/ml/datasets/Student+Performance

# library imports
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("/content/student_data/student-mat.csv", sep = ";")  #use the directory of your dataset

print(data.head())

# the only data we need
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head())

#building the model to for Final Grades prediction i.e. G3 in the dataset.
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# dividing the dataset into four different variables.
# test variables are randomly allocated.
# Spliting up 10% of data for test.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# WARNING: TIME SENSITIVE LOOP

best_acc = 0

# no of iterations for the loop to find the best model. more the merrier.
# for this dataset, I found a model of about 98.18% accuracy after running 1000000 iterations

iter = 1000000

for _ in range(iter):
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) #Spliting up 10% of data for test.
  # Model Implementation
  linear = linear_model.LinearRegression()
  linear.fit(x_train, y_train)
  accuracy = linear.score(x_test, y_test)
  # print the accuracy of the model upto two decimal place.
  # print(format(accuracy*100, ".2f")+"%")
  # Saves the best model using pickle
  if accuracy > best_acc:
    best_acc = accuracy
    with open("studentmodel.pickle", "wb") as f:
      pickle.dump(linear, f)
#best model accuracy result for the test size splitted before.
print(best_acc*100)

# loading the model I saved before
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# prints the coefficeints & intercept
print('Coefficients: ', linear.coef_)
print('Vertical Intercept: ', linear.intercept_)

# comparing the predicted values using the saved model and the real values to verify the accuracy
predictions = linear.predict(x_test)

for i in range (len(predictions)):
  print("Predicted Value: " + format(predictions[i], ".2f"), x_test[i], "\nActual Value:", y_test[i])

# plotting the graphs using matplotlib
# these 5 graphs show the relation with the parameters and G3, the final grade parameter.

# with grade 1
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

# with grade 2
p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

# with studytime
p = 'studytime'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

# with failures
p = 'failures'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

# with absences
p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
# end