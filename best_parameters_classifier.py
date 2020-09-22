import sys
import os

inputdir = sys.argv[1]
classifier = "lstm" # Can be also mlp
model = "transe" # Can be complex, rotate

ranking = []
for dirName, subdirList, fileList in os.walk(inputdir):
    print(fileList)