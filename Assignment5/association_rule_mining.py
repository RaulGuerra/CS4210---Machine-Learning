#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
import math

import numpy as numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import math

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
TempSet = list(itemset)


def isNaN(num):
    return num != num


for index, row in df.iterrows():
    labels = [0,0,0,0,0,0,0,0,0]
    for i in range(row.size):
        exists = -1
        if isNaN(row[i]):
            doNothing = 1
        else:
            exists = TempSet.index(row[i])
            labels[exists] = 1
    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)


    #iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below
for i in range(rules.shape[0]):
    print("-------Rule " + str(i+1) + "-------")
    anteFrozen = rules.loc[i]['antecedents']
    temp = list(anteFrozen)
    ante = []
    for j in range(len(temp)):
        ante.append(TempSet[temp[j]])

    conseFrozen = rules.loc[i]['consequents']
    temp = list(conseFrozen)
    conse = []
    for k in range(len(temp)):
        conse.append(TempSet[temp[k]])

    print(list(ante), " -> ", list(conse))
    print("Support: ", rules.loc[i]['support'])
    print("Confidence: ", rules.loc[i]['confidence'])
    supportCount = sum(row[temp[0]] for row in encoded_vals)
    prior = supportCount/len(encoded_vals)
    print("Prior: ", prior)
    gainConfi = 100*((rules.loc[i]['confidence']-prior)/prior)
    print("Gain in Confidence: ", gainConfi)

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


