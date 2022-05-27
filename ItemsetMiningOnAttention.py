import pandas as pd
import numpy as np
from decimal import Decimal
import os

def unique(list1): 
  
    unique_list = [] 
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

df = pd.read_csv("test_result_all.csv", header = None)
df2 = pd.read_csv("test_result.csv", header = None)

df = df.loc[(df[5] == 1.0) & (df[6] >= 0.50)]
df2 = df2.loc[(df2[2] == 1.0) & (df2[3] >= 0.50)]

path_column = df2.values[:,0]
edited_column = []

for i,p in enumerate(path_column):
    row = df.values[i]
    hyponym = row[0]
    hypernym = row[1]
    p = p.replace(hyponym, "x")
    p = p.replace(hypernym, "y")
    edited_column.append(p)

path_column = edited_column

lengths = [len(a.split('%,,%')) for a in path_column]

attention_column = df.values[:,4]

attention = [[np.log(float(b)) for b in a.split(",")][:lengths[i]] for i,a  in enumerate(attention_column)]
weight = [[b-min(a) for b in a] for a in attention]
normalized_weight = [[b/sum(a) for b in a] for a in weight]

paths = [[b.split('!!') for b in a.split('%,,%')] for a in path_column]

frequent_edges = []

for i, weights in enumerate(normalized_weight):
    max_weight = max(weights)
    for j,w in enumerate(weights):
        if(df.values[i][5] >= 0.9 and df.values[i][6] >= 0.5 and w >= 0.00001 * max_weight):
            p = '%,,%'.join(paths[i][j])
            p = p.replace(df.values[i][0],"x")
            p = p.replace(df.values[i][1],"y")
            p = p.split('%,,%')
            paths[i][j] = p
            frequent_edges.append(p)

path_indices = []
frequent_edges =unique(frequent_edges)

for i,pt in enumerate(paths):
    temp = []
    for j,p in enumerate(pt):
        if(df.values[i][5] >= 0.9 and df.values[i][6] >= 0.5 and p in frequent_edges):
            temp.append(frequent_edges.index(p))
    if(len(temp)>0):
        path_indices.append(temp)

f2 = open('input.txt','w')

for indices in path_indices:
    myString = ",".join(map(str,indices))
    f2.writelines(myString+'\n')

f2.close()
os.system("eclat.exe -s0.87 -c0.89 input.txt output.txt")
f3 = open('output.txt','r')

lines = f3.readlines()

eclat_frequent_edges = []
supports = []
sentences = []
hyponyms = []
hypernyms = []

for line in lines:
    strpline = line.rstrip()
    #print(strpline)
    arr = strpline.split(' ')

    st = float(''.join(c for c in arr[-1] if c.isdigit() or c == '.'))

    #if(st<2.0):
    #    continue

    edges = [];
    support = [];
    for s in arr:
            if "(" not in s:
                edges.append(frequent_edges[int(s)])
            else:
                st = float(''.join(c for c in s if c.isdigit() or c == '.'))
                support.append(st)

    edges_check = ["!!".join(e) for e in edges]

    for i,a in enumerate(path_column):
        k = 0
        for b in edges_check:
            if(a.find(b) >=0 and df.values[i][5] >= 0.9 and df.values[i][5] >= 0.6):
                k = k + 1
        if(k == len(edges_check)):
            row = df.values[i]
            hyponyms.append(row[0])
            hypernyms.append(row[1])
            sentences.append(row[2])
            break
            
            
    eclat_frequent_edges.append('%,,%'.join(["!!".join(e) for e in edges]))
    supports.append("!!".join([str(a) for a in support]))

f3.close()

df = pd.DataFrame(data = {'Hyponyms' : hyponyms, 'Hypernyms' : hypernyms, 'Sentences': sentences ,'Eclat_Edges':eclat_frequent_edges, 'Supports':supports})

df.to_csv("Eclat_Edges_test.csv",header =False, index = False)


