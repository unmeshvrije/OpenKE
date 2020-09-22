import sys
import os

inputdir = sys.argv[1]
classifier = "lstm" # Can be also mlp
model = "transe" # Can be complex, rotate
pred = "tail" # Can be also head

prefixToSearch = "Validation set : accuracy: "

ranking = []
for dirName, subdirList, fileList in os.walk(inputdir):
    for f in fileList:
        if f.endswith('.log') and classifier in f and model in f and pred in f:
            for line in open(inputdir + "/" + f, 'r'):
                if prefixToSearch in line:
                    accuracy = float(line[len(prefixToSearch):-2])
                    ranking.append((f, accuracy))

ranking.sort(key=lambda tup: tup[1])
for r in ranking:
    print(r)
