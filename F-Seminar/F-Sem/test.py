import numpy as np
import pandas as pd
import csv
import itertools
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

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


    data = []
    keys, values = zip(*valueDict.items())
    for x in itertools.product(*values):
        data.append(x)
        #print(str(x))

    with open('generated.csv', 'w', newline="") as gen:
        writer = csv.writer(gen)

        writer.writerow(header)
        writer.writerows(data)


def addColumnWithValues(df, predictions):
    df["predictionResult"] = 0
    for i in range(len(df["predictionResult"])):
        df["predictionResult"][i] = predictions[i]

    #region Test
    smap = {0: 'm', 1: 'w'}
    emap = {0: 'weiss', 1: 'afroamerikanisch', 2: 'asiatisch'}
    pmap = {0: "No", 1: "Yes"}

    df['Geschlecht'] = df['Geschlecht'].map(smap)
    df['Ethnie'] = df['Ethnie'].map(emap)
    df['predictionResult'] = df['predictionResult'].map(pmap)
    #endregion Test

    df.to_csv("generated.csv")


def evaluateResults(df):
    evaDict = {}
    for ea in evaAttributes:
        evaDict[ea] = []
        for uni in df[ea].unique().tolist():
            evaDict[ea].append(uni)

    tdf = df["predictionResult"] == "Yes"
    befDF = df[tdf]

    #print(evaDict)

    #Das st jetzt hart gecodet, ist insgesamt ein bisschen hier, aber das ist egal, da wir den Code nicht abgeben. Das jetzt schön und generell zu bauen wäre jetzt ein Abfuck
    metricCalcDict ={
        "Verkaeufe" : [],
        "Priv" : "",
        "Unpriv": "",
    }

    for ed in evaDict:
        for listElement in evaDict[ed]:
            #df[y] == 1: Alle, die befördert werden sollen
            tempDf = befDF[ed] == listElement
            filtDf = befDF[tempDf]

            print(ed + "/" + listElement + ": " + str(len(filtDf)) + " wurden befördert")
            print(ed + "/" + listElement + ": " + str(len(filtDf) / len(befDF)) + " = Anzahl an Beförderungen aus der Gruppe / Alle Beförderungen")
            print(ed + "/" + listElement + ": " + str(filtDf.Verkaeufe.min()) + " ist die niedrigste Anzahl an Verkäufen, um in dieser Gruppe befördert zu werden")
            print("")

            metricCalcDict["Verkaeufe"].append(filtDf.Verkaeufe.min())

    print(max(metricCalcDict["Verkaeufe"]) - min(metricCalcDict["Verkaeufe"]))

#endregion Methoden


df = pd.read_csv("TestDatenNonBias.CSV", )

#============ Kategorische Werte werden auf Zahlen gemappt ============

sMap = {'m' : 0, 'w': 1}
eMap = {'weiss' : 0, 'afroamerikanisch' : 1, 'asiatisch' : 2}
pMap = {"n" : 0, "j" : 1}

df['Geschlecht'] = df['Geschlecht'].map(sMap)
df['Ethnie'] = df['Ethnie'].map(eMap)
df['Gehaltserhoehung'] = df['Gehaltserhoehung'].map(pMap)

features = ['Geschlecht', 'Ethnie', 'Verkaeufe', 'Monate beschaeftigt'] #Die Features sind die unabhängigen Variablen


#============ Dieses Dict enthält Metainformationen über das Dataset ============


columnTypeDictionary["numerical"].append(['Verkaeufe', 'Monate beschaeftigt'])
columnTypeDictionary["categorie"].extend(['Ethnie', 'Geschlecht'])

evaAttributes.append('Ethnie')
evaAttributes.append("Geschlecht")


#============ Hier kommt der AI Spaß ============

x = df[features]
y = df['Gehaltserhoehung']

xTrain, xTest, yTrain, yTest =  train_test_split(x,y, test_size = 0.3, random_state = 100)



dtree = DecisionTreeClassifier()
dtree = dtree.fit(xTrain,yTrain)

y_prediction = dtree.predict(xTest)


print("Accuracy is " + str(accuracy_score(yTest, y_prediction)*100))


buildTable(x,columnTypeDictionary, 1)
generated = pd.read_csv("generated.csv")
result = dtree.predict(generated)


addColumnWithValues(generated, result)

finalDF = pd.read_csv("generated.csv")

evaluateResults(finalDF)