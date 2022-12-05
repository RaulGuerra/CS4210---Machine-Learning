#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
import labels as labels
#importing some Python libraries
import sklearn.cluster
from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import sklearn.metrics
from sklearn import metrics
import csv

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix

X_training = []

with open('training_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        X_training.append ([eval(i) for i in row])


#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

currentScore = 0
k_plot = []
silhouette_plot = []
tempScore = ""

for k in range(2, 21):
    k_plot.append(k)
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
    # find which k maximizes the silhouette_coefficient
    # --> add your Python code here
    tempScore = silhouette_score(X_training, kmeans.labels_)
    silhouette_plot.append(tempScore)
    if tempScore > currentScore:
        currentScore = tempScore
        print("************Best score so far: ", currentScore, ", by K: ", k)
        bestK = k

# plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
# --> add your Python code here
plt.plot(k_plot, silhouette_plot)
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
tf = pd.read_csv('testing_data.csv', sep=',', header=None) #reading the testing data by using Pandas library
X_testing = []
with open('testing_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        tempInt = ([eval(i) for i in row])
        X_testing.append(tempInt[0])


#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
np.array(df.values).reshape(1,244672)[0]

kmeans = KMeans(n_clusters=bestK, random_state=0)
kmeans.fit(X_training)

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(X_testing, kmeans.labels_).__str__())
#--> add your Python code here
