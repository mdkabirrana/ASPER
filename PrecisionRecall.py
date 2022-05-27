import pandas as pd
import numpy as np
import re

df = pd.read_csv("kabir_generated_data.csv", header = None)

df = df.loc[df[3] == 1]

#df = df.loc[df[4] == True]

positive_pattern = []

positive_pattern += ["Y, such as X",
                        "Y, including X",
                        "Y, especially X",
                        "Y, like X",
                        "Y, called X"]
positive_pattern += ["unlike most Y, X",
                        "unlike all Y, X",
                        "unlike any Y, X",
                        "unlike other Y, X",
                        "like most Y, X",
                        "like all Y, X",
                        "like any Y, X",
                        "like other Y, X"
                        ]

positive_pattern += ["X and any other Y",
                        "X and some other Y",
                        "X and many other Y",
                        "X and all other Y",
                        "X or any other Y",
                        "X or some other Y",
                        "X or many other Y",
                        "X or all other Y"]
positive_pattern += ["X, which is a Y",
                        "X, an example of Y",
                        "X, a class of Y",
                        "X, a kind of Y",
                        "X, a type of Y"]
positive_pattern += ["X is a Y",
                        "X is an example of Y",
                        "X is a class of Y",
                        "X is a kind of Y",
                        "X is a type of Y"]
positive_pattern += ["a Y, X",
                        "an example of Y, X",
                        "a class of Y, X",
                        "a kind of Y, X",
                        "a type of Y, X"]

patterns = []
sentences = []

print(len(positive_pattern))

for val in df.values:
    t = val
    val[2] = val[2].lower()
    val[0] = val[0].lower()
    val[1] = val[1].lower()
    val[2] = val[2].replace(val[0], "X")
    val[2] = val[2].replace(val[1], "Y")
    splitted_target =  re.split(' |,',val[2])
    splitted_target = [a for a in splitted_target if len(a) > 0]

    for pat in positive_pattern:
        current = re.split(' |,',pat)
        current = [a for a in current if len(a) > 0]
        count = 0

        try:
            index = splitted_target.index(current[0])
        except:
            index = -1
            continue

        for i,c in enumerate(current):
            try:
                if(splitted_target[index].find(c)> -1):
                    index = index + 1
                    count = count + 1
            except:
                break
        if count >= len(current):
            patterns.append(pat)
            positive_pattern.remove(pat)
            sentences.append(val[2])

print(len(patterns))


