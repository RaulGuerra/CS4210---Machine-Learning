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
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

X = []
Y = []

dbTraining1 = []
with open(dataSets[0], 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining1.append (row)

dbTraining2 = []
with open(dataSets[1], 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTraining2.append(row)

dbTraining3 = []
with open(dataSets[2], 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTraining3.append(row)

#read the test data and add this data to dbTest
#--> add your Python code here
# dbTest =
dbTest = []
with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTest.append(row)

testData = [1, 1, 2, 1], [1, 2, 2, 1], [1, 1, 2, 2], [3, 2, 2, 2], [3, 1, 2, 1], [3, 1, 1, 2], [2, 1, 1, 1], [2, 1, 2, 2]
testTrueLabel = [1,1,2,2,2,2,1,2]


# transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
# transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]

# Training set 1:
X1 = [1, 1, 1, 1], [1, 2, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [2, 2, 2, 2], [2, 2, 1, 1], [3, 1, 1, 1], [3, 2, 1,
                                                                                                               1]
# Training 1 Classes:
Y1 = [1, 1, 1, 1, 2, 2, 1, 2]

# Training set 2:
X2 = [1, 1, 2, 2], [1, 1, 2, 1], [1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 1], [2, 1, 2, 1], [2, 2, 1, 1], [3, 1, 2, 1], [3, 1, 1, 1], [3, 2, 2, 2], [3, 2, 2, 1], [3, 2, 1, 1]

# Training 2 Classes:
Y2 = [2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2]

# Training set 3:
X3 = [1, 1, 2, 2], [1, 1, 2, 1], [1, 1, 1, 2], [1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 1], [1, 2, 1, 2], [1, 2, 1, 1], [2, 1, 2, 2], [2, 1, 2, 1], [2, 1, 1, 2], [2, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 1], [2, 2, 1, 2], [2, 2, 1, 1], [3, 1, 2, 2], [3, 1, 2, 1], [3, 1, 1, 2], [3, 1, 1, 1], [3, 2, 2, 2], [3, 2, 2, 1], [3, 2, 1, 2], [3, 2, 1, 1]

# Training 3 Classes:
Y3 = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2]



#----------------------------- Training Set 1 -----------------------------


lowestAccuracy = 1

#loop your training and test tasks 10 times here
for i in range (10):

    #fitting the decision tree to the data setting max_depth=3
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
    clf = clf.fit(X1, Y1)
    accuratePredictions = 0
    totalPredictions = 0
    iterator = 0

    for data in dbTest:
        #transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        #--> add your Python code here

        class_predicted = clf.predict([testData[iterator]])[0]


        #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        #--> add your Python code here

        if class_predicted == testTrueLabel[iterator]:
            accuratePredictions += 1

        totalPredictions += 1
        iterator += 1

    #find the lowest accuracy of this model during the 10 runs (training and test set)
    #--> add your Python code here
    currentAccuracy = accuratePredictions/totalPredictions

    if lowestAccuracy > currentAccuracy:
        lowestAccuracy = currentAccuracy

#print the lowest accuracy of this model during the 10 runs (training and test set).
#your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
#--> add your Python code here

print("final accuracy when training on contact_lens_training_1.csv: ", lowestAccuracy)




#----------------------------- Training Set 2 -----------------------------



lowestAccuracy = 1

#loop your training and test tasks 10 times here
for i in range (10):

    #fitting the decision tree to the data setting max_depth=3
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
    clf = clf.fit(X2, Y2)
    accuratePredictions = 0
    totalPredictions = 0
    iterator = 0

    for data in dbTest:
        #transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        #--> add your Python code here

        class_predicted = clf.predict([testData[iterator]])[0]


        #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        #--> add your Python code here

        if class_predicted == testTrueLabel[iterator]:
            accuratePredictions += 1

        totalPredictions += 1
        iterator += 1

    #find the lowest accuracy of this model during the 10 runs (training and test set)
    #--> add your Python code here
    currentAccuracy = accuratePredictions/totalPredictions

    if lowestAccuracy > currentAccuracy:
        lowestAccuracy = currentAccuracy

#print the lowest accuracy of this model during the 10 runs (training and test set).
#your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
#--> add your Python code here

print("final accuracy when training on contact_lens_training_2.csv: ", lowestAccuracy)



#----------------------------- Training Set 3 -----------------------------



lowestAccuracy = 1

#loop your training and test tasks 10 times here
for i in range (10):

    #fitting the decision tree to the data setting max_depth=3
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
    clf = clf.fit(X3, Y3)
    accuratePredictions = 0
    totalPredictions = 0
    iterator = 0

    for data in dbTest:
        #transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        #--> add your Python code here

        class_predicted = clf.predict([testData[iterator]])[0]


        #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        #--> add your Python code here

        if class_predicted == testTrueLabel[iterator]:
            accuratePredictions += 1

        totalPredictions += 1
        iterator += 1

    #find the lowest accuracy of this model during the 10 runs (training and test set)
    #--> add your Python code here
    currentAccuracy = accuratePredictions/totalPredictions

    if lowestAccuracy > currentAccuracy:
        lowestAccuracy = currentAccuracy

#print the lowest accuracy of this model during the 10 runs (training and test set).
#your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
#--> add your Python code here

print("final accuracy when training on contact_lens_training_3.csv: ", lowestAccuracy)
