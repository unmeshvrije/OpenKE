import pickle as plt
import numpy as np
import sys

inputFile = sys.argv[1]
outputFile = sys.argv[2]
dim = int(sys.argv[3])

data = plt.load(open(inputFile, 'rb'))

def postprocess(data, suffix):
    x_data = data['x_' + suffix]
    newdata = np.zeros(shape=(len(x_data), dim), dtype=np.float64)
    for i in range(len(x_data)):
        newdata[i]  = x_data[i][5:]
    return newdata

newdata = {}
newdata['x_tail'] = postprocess(data, 'tail')
newdata['x_head'] = postprocess(data, 'head')
newdata['y_tail'] = data['y_tail']
newdata['y_head'] = data['y_head']
plt.dump(newdata, open(outputFile, 'wb'))
