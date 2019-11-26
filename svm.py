import numpy as np
import sys
import json
from sklearn.svm import SVC

input_file = sys.argv[1]

print("Loading training data...", end = " ")
with open(input_file, "r") as fin:
    data = json.loads(fin.read())

print("DONE")
all_data_x = data['x_tail']
all_data_y = data['y_tail']

print("Splitting data...", end = " ")
percent_test = 10
N_TOTAL = len(all_data_x)
N_TEST = int(N_TOTAL * percent_test / 100)
N_TRAIN = int(N_TOTAL - N_TEST)
print(N_TOTAL)
print(N_TRAIN)
print(N_TEST)
print("#"*80)

x_train = all_data_x[:N_TRAIN]
y_train = all_data_y[:N_TRAIN]

labels = np.unique(y_train)
print("yrai[0], ", y_train[0])

print("Training : ", len(x_train))
x_test = all_data_x[N_TRAIN:]
y_test = all_data_y[N_TRAIN:]

print("Test ", len(x_test))

print("*** ", labels)
print("DONE")
model = SVC()
print("Fitting SVM...", end = " ")
model.fit(x_train, y_train)

print("DONE")
print(model.score(x_test, y_test))

a,b = (model.predict(x_test) == y_test).sum() , len(y_test)
print(a, b)

