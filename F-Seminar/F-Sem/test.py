import numpy as np
import pandas as pd
import csv
import itertools
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import random

#region Attribute

columnTypeDictionary = {
    "numerical": [],
    "categorie": []
}

evaAttributes = []

#endregion Attribute


#region Methoden

def buildTable(df, dict, stepRange):
    valueDict = {}
    header = []

    for c in df:
        if str(c) in dict["numerical"]:
            valueDict[str(c)] = range(df[str(c)].min(), df[str(c)].max() +1)
        else:
            valueDict[str(c)] = df[str(c)].unique().tolist()

        header.append(str(c))

    with open('generated.csv', 'w', newline="") as gen:
        writer = csv.writer(gen)
        writer.writerow(header)

        data = []
        keys, values = zip(*valueDict.items())
        for x in itertools.product(*values):
            writer.writerow(x)
            # print(str(x))


def addColumnWithValues(df, predictions):
    df["predictionResult"] = 0
    df["compareValue"] = 0
    for i in range(len(df["predictionResult"])):
        df["predictionResult"][i] = predictions[i]
        #df["compareValue"][i] =  df["Verkaeufe"][i] /  df["Monate beschaeftigt"][i]


    #region Test
    smap = {0: 'm', 1: 'w'}
    emap = {0: 'weiss', 1: 'afroamerikanisch', 2: 'asiatisch'}
    pmap = {0: "No", 1: "Yes"}

    #df['Geschlecht'] = df['Geschlecht'].map(smap)
    #df['Ethnie'] = df['Ethnie'].map(emap)
    #df['predictionResult'] = df['predictionResult'].map(pmap)
    #endregion Test

    df.to_csv("generated.csv")


def evaluateResults(df):
    evaDict = {}
    for ea in evaAttributes:
        evaDict[ea] = []
        for uni in df[ea].unique().tolist():
            evaDict[ea].append(uni)

    tdf = df["predictionResult"] == 1
    befDF = df[tdf]

    #print(evaDict)

    itemsToSumUpAA = []
    itemsToSumUpAU = []

    for ed in evaDict:
        minimumAA = []
        allAU = []
        for listElement in evaDict[ed]:
            #df[y] == 1: Alle, die befördert werden sollen
            tempDf = befDF[ed] == listElement
            filtDf = befDF[tempDf]

            print(ed + "/" + str(listElement) + ": " + str(len(filtDf)) + " wurden befördert")
            print(ed + "/" + str(listElement) + ": " + str(len(filtDf) / len(befDF)) + " = Anzahl an Beförderungen aus der Gruppe / Alle Beförderungen")
            #print(ed + "/" + str(listElement) + ": " + str(filtDf.compareValue.min()) + " ist die niedrigste Verkäufe pro Monat beschäftigt-Ratio, um befördert zu werden")
            print("")

            #minimumAA.append(filtDf.compareValue.min())
            allAU.append((len(filtDf) / len(befDF)) * 100)
        #itemsToSumUpAA.append(max(minimumAA) - min(minimumAA))
        itemsToSumUpAU.append(max(allAU) - min(allAU))

    #print("Anforderungsabstand: "+ str(sum(itemsToSumUpAA)/len(evaDict)))
    #print(sum(itemsToSumUpAA))
    print("Anteilsunterschied: " + str(sum(itemsToSumUpAU) / (len(evaDict) * 100)))


#endregion Methoden


filename = "CreditSimuUnbalanced.csv"
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 10 #desired sample size
skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

df = pd.read_csv(filename, skiprows=skip)

MLdf = pd.read_csv(filename)
#df = pd.read_csv("TestDatenHardBias.CSV", )


#============ Kategorische Werte werden auf Zahlen gemappt ============

#sMap = {'m' : 0, 'w': 1}
#eMap = {'weiss' : 0, 'afroamerikanisch' : 1, 'asiatisch' : 2}
#pMap = {"n" : 0, "j" : 1}

#df['Geschlecht'] = df['Geschlecht'].map(sMap)
#df['Ethnie'] = df['Ethnie'].map(eMap)
#df['Gehaltserhoehung'] = df['Gehaltserhoehung'].map(pMap)

features = ["Mortgage","Balance","Amount Past Due","Delinquency Status","Credit Inquiry","Open Trade","Utilization","Gender","Race"] #Die Features sind die unabhängigen Variablen


#============ Dieses Dict enthält Metainformationen über das Dataset ============


#columnTypeDictionary["numerical"].extend(['Verkaeufe', 'Monate beschaeftigt'])
columnTypeDictionary["categorie"].extend(features)

evaAttributes.append('Race')
evaAttributes.append("Gender")


#============ Hier kommt der AI Spaß ============

x_table = df[features]

x = MLdf[features]
y = MLdf['Status']

xTrain, xTest, yTrain, yTest =  train_test_split(x,y, test_size = 0.3, random_state = 100)



dtree = DecisionTreeClassifier()
dtree = dtree.fit(xTrain,yTrain)

y_prediction = dtree.predict(xTest)


print("Accuracy is " + str(accuracy_score(yTest, y_prediction)*100))


buildTable(x_table,columnTypeDictionary, 1)
generated = pd.read_csv("generated.csv")
result = dtree.predict(generated)


addColumnWithValues(generated, result)

finalDF = pd.read_csv("generated.csv")

evaluateResults(finalDF)