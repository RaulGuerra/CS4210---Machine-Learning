#-------------------------------------------------------------------------
# AUTHOR: Raul Guerra
# FILENAME: naive_bayes.py
# SPECIFICATION: makes prediction using naive bayes
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv
import sys

#reading the training data in a csv file
#--> add your Python code here

trainingDB = []

with open("./weather_training.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    trainingDB.append(row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = [1, 1, 1, 1], [1, 1, 1, 2], [2, 1, 1, 1], [3, 2, 1, 1], [3, 3, 2, 1], [3, 3, 2, 2], [2, 3, 2, 2], [1, 2, 1, 1], [1, 3, 2, 1], [3, 2, 2, 1], [1, 2, 2, 2], [2, 2, 1, 2], [2, 1, 2, 1], [3, 2, 1, 2]

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = [1,1,2,2,2,1,2,1,2,2,2,2,2,1]


#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here

testDB = []

with open("./weather_test.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    testDB.append(row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
#print (testDB, clf.predict([[1,2,2,1]])[0], clf.predict_proba([[1,2,2,1]])[0])

testArray = []
testArray = [1, 1, 2, 1], [1, 1, 2, 2], [1, 3, 1, 1], [2, 1, 1, 2], [2, 3, 1, 1], [2, 3, 1, 2], [3, 2, 2, 2], [3, 1, 2, 2], [3, 1, 1, 1], [3, 2, 1, 2]

counter = 0

for i in testDB[1:]:
  for j in i:
    for k in j:
      for l in k:
        for m in l:
          for n in m:
            if n != "?":
              print(n, end="")
            else:
              predict = clf.predict([testArray[counter-1]])[0]
              if predict == 1:
                print("No", end="")
              else:
                print("Yes", end="")
              print("".ljust(12), end='')
              print(clf.predict_proba([testArray[counter-1]])[0], end="")
    print("".ljust(12), end = '')
  print("\n")
  counter += 1

