import pickle as plt
import numpy as np
import sys

inputFile = sys.argv[1]
outputFile = sys.argv[2]

data = plt.load(open(inputFile, 'rb'))

def postprocess(data, suffix):
    x_data = data['x_' + suffix]
    newdata = np.zeros(shape=(len(x_data), 600), dtype=np.float64)
    for i in range(len(x_data)):
        newdata[i]  = x_data[i][5:]
    return newdata

newdata = {}
for mode in {'raw', 'fil'}:
    newdata['x_tail_' + mode] = postprocess(data, 'tail_' + mode)
    newdata['x_head_' + mode] = postprocess(data, 'head_' + mode)
    newdata['y_tail_' + mode] = data['y_tail_' + mode]
    newdata['y_head_' + mode] = data['y_head_' + mode]
plt.dump(newdata, open(outputFile, 'wb'))