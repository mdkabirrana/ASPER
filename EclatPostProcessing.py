import pandas as pd
import numpy as np

df = pd.read_csv("Eclat_Edges_test.csv", header = None)

values = df.values

#values = [v for v in values if v[-1]>=0.2 and len(v[-2].split("%,,%")) > 2]

values = [v for v in values if v[-1]>=0.09]

distinct_maximal_edges = []

for i,v in enumerate(values):
    pattern = v[-2].split("%,,%")
    xExists = False
    yExists = False
    thatExists = False
    interestingExists = False
    for p in pattern:
        components = p.split("!!")
        if(components[0] == "x"):
            xExists = True
        if(components[0] == "y"):
            yExists = True
        if(components[0] == "that"):
            thatExists = True
        if(components[0] == "interesting"):
            interestingExists = True
        if(components[3] == "x"):
            xExists = True
        if(components[3] == "y"):
            yExists = True
        if(components[3] == "that"):
            thatExists = True
        if(components[3] == "interesting"):
            interestingExists = True
    if(xExists == True and yExists == True and thatExists == False and interestingExists == False):
        isCandidate = True
        for edges in distinct_maximal_edges:
            replaceDistinct = True
            for e in edges:
                if(e not in pattern):
                    replaceDistinct = False
                    break
            if(replaceDistinct == True and len(pattern) > len(edges)):
                distinct_maximal_edges.remove(edges)
                distinct_maximal_edges.append(pattern)
                isCandidate = False
                break
        if(isCandidate == True):
            distinct_maximal_edges.append(pattern)
subsets = []

for d in distinct_maximal_edges:
    for d2 in distinct_maximal_edges:
        if(frozenset(d).issubset(d2) and len(d)<len(d2)):
            subsets.append(d)

for s in subsets:
    if(s in distinct_maximal_edges):
        distinct_maximal_edges.remove(s)

finalValues = []
stopEdges = ["architecture","towada","country","parish","county","province","region","and", "district","state", "rural","located","romanized","small","in","civil","thirsk","east","yorkshire","harrogate","east","contents"]

for d in distinct_maximal_edges:
    for v in values:
        pattern = v[-2]
        if(pattern == "%,,%".join(d)):
            filteredEdges = []
            for k in d:
                found = False
                for s in stopEdges:
                    if s in k:
                        found = True
                        break
                if found == False:
                    filteredEdges.append(k)

            v[-2] = "%,,%".join(filteredEdges)
            
            finalValues.append(v)
            break

df = pd.DataFrame(data = finalValues)

df.to_csv("PostProcessedEdges_test.csv",header =False, index = False)






