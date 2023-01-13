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
    print("Table wird gebuildet")
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
        "Sales" : [],
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
            print(ed + "/" + listElement + ": " + str(filtDf.Sales.min()) + " ist die niedrigste Anzahl an Verkäufen, um in dieser Gruppe befördert zu werden")
            print("")

            metricCalcDict["Sales"].append(filtDf.Sales.min())

    print(max(metricCalcDict["Sales"]) - min(metricCalcDict["Sales"]))

#endregion Methoden


df = pd.read_csv("FairnessCreditSimu.csv")

#print(df[['RecSupervisionLevel', 'RecSupervisionLevelText']].to_string()) # Das sind die Evaluationsspalten, das ist wichtig

#print(df.to_string())

#print("")

#for col in df:
    #print(str(col))



features = ['Gender', 'Race', 'Mortgage', 'Balance', 'Amount Past Due', 'Credit Inquiry'] #Die Features sind die unabhängigen Variablen


#============ Dieses Dict enthält Metainformationen über das Dataset ============


#columnTypeDictionary["numerical"].append('Sales')
columnTypeDictionary["categorie"].extend(['Gender', 'Race', 'Mortgage', 'Balance', 'Amount Past Due', 'Credit Inquiry'])

evaAttributes.append('Ethnie')
evaAttributes.append("Sex")


#============ Hier kommt der AI Spaß ============

x = df[features]
y = df['Approved']

xTrain, xTest, yTrain, yTest =  train_test_split(x,y, test_size = 0.5, random_state = 100)



dtree = DecisionTreeClassifier()
dtree = dtree.fit(xTrain,yTrain)

y_prediction = dtree.predict(xTest)
#print(xTest)
#print(y_prediction)

print("Accuracy is " + str(accuracy_score(yTest, y_prediction)*100))

#TargetValue = df.values[19]
#df.drop([19, 20])
#Features = df.values

#dtree = DecisionTreeClassifier()
#dtree = dtree.fit(Features, TargetValue)

#tree.plot_tree(dtree, feature_names=Features)

#print(type(df[features]))


buildTable(x,columnTypeDictionary, 1)
generated = pd.read_csv("generated.csv")
result = dtree.predict(generated)

#print(result)

addColumnWithValues(generated, result)

finalDF = pd.read_csv("generated.csv")
#print(len(finalDF))

#evaluateResults(finalDF)