from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import numpy as np
import sys
import json

input_file = sys.argv[1]

print("Loading training data...", end = " ")
with open(input_file, "r") as fin:
    data = json.loads(fin.read())

print("DONE")
TOPK = 10

SAMPLE_SIZE = 10000

all_data_x = data['x_tail'][:SAMPLE_SIZE]
all_data_y = data['y_tail'][:SAMPLE_SIZE]

print("Splitting data...", end = " ")
#percent_test = 10

print("x data shape ", np.shape(all_data_x))
print("y data shape ", np.shape(all_data_y))
N_TOTAL = len(all_data_x)

print("N_TOTAL = ", N_TOTAL)
N_TEST = 100 #int(N_TOTAL * percent_test / 100)
N_TRAIN = int(N_TOTAL - N_TEST)

print("N_TRAIN = ", N_TRAIN)
print("N_TEST  =", N_TEST)
print("#"*80)

x_train = np.array(all_data_x[:N_TRAIN])
y_train = np.array(all_data_y[:N_TRAIN])

print("Training x shape: ", np.shape(x_train))
x_test = np.array(all_data_x[N_TRAIN:])
y_test = np.array(all_data_y[N_TRAIN:])

print("Test x length ", len(x_test))
print("DONE")

DIM = 200

# each query has topK answers (sequence of 50 answers)
# each answer has 200 features
# 1 for score, 1 for rank, 3 for (h,r,t) ids and 1 for answer entity
N_FEATURES = len(x_train[0])


x_train = np.reshape(x_train, (N_TRAIN//TOPK, TOPK, N_FEATURES))
y_train = np.reshape(y_train, (N_TRAIN//TOPK, TOPK, 1))
#x_train = np.random.random((N_TRAIN, TOPK, N_FEATURES))
#y_train = ((np.random.random((N_TRAIN, TOPK, 1)) * 42) % 2).astype(int)

#x_test = np.random.random((N_TEST, TOPK, N_FEATURES))
#y_test = ((np.random.random((N_TEST, TOPK, 1)) * 42) % 2).astype(int)
x_test = np.reshape(x_test, (N_TEST//TOPK, TOPK, N_FEATURES))
y_test = np.reshape(y_test, (N_TEST//TOPK, TOPK, 1))

model = Sequential()
model.add(LSTM(1, input_shape=(TOPK, N_FEATURES),return_sequences=True))

model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2,validation_data=(x_test, y_test))
predict = model.predict(x_test)
