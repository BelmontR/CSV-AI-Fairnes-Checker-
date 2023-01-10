import pandas as pd
import csv
import itertools

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



    #with open('generated.csv', 'w') as gen:
     #   writer = csv.writer(gen)

      #  writer.writerow(header)


