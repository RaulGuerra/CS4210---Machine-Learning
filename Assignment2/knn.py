#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv
import array

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

iterator = 0
wrongPredictions = 0

wholeSet = [2, 1], [4, 1], [3, 2], [0, 3], [3, 3], [4, 3], [1, 4], [2, 4], [4, 4], [0, 5]
wholeClasses = [1,1,2,1,2,2,1,2,2,1]
#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X and remove the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning messages

    #transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...]. Convert values to float to avoid warning messages

    #--> add your Python code here
    X = [(2, 1), (4, 1), (3, 2), (0, 3), (3, 3), (4, 3), (1, 4), (2, 4), (4, 4), (0, 5)]
    X.pop(iterator)

    Y = [1,1,2,1,2,2,1,2,2,1]
    Y.pop(iterator)

    testSample = wholeSet[iterator]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    prediction = class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    if prediction != wholeClasses[iterator]:
        wrongPredictions += 1

    iterator += 1

#print the error rate
#--> add your Python code here
errorRate = wrongPredictions/10
print("Error Rate: ", errorRate)





