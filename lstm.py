from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.models import model_from_json

import numpy as np
import sys
import json

input_file = sys.argv[1]
mode = sys.argv[2]

print("Loading training data...", end = " ")
with open(input_file, "r") as fin:
    data = json.loads(fin.read())

print("DONE")
TOPK = 10

SAMPLE_SIZE = 50000

all_data_x = data['x_tail'][:SAMPLE_SIZE]
all_data_y = data['y_tail'][:SAMPLE_SIZE]

print("Splitting data...", end = " ")
#percent_test = 10

print("x data shape ", np.shape(all_data_x))
print("y data shape ", np.shape(all_data_y))
N_TOTAL = len(all_data_x)

print("N_TOTAL = ", N_TOTAL)
N_TEST = 500 #int(N_TOTAL * percent_test / 100)
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

N_FEATURES = len(x_train[0])

x_train = np.reshape(x_train, (N_TRAIN//TOPK, TOPK, N_FEATURES))
y_train = np.reshape(y_train, (N_TRAIN//TOPK, TOPK, 1))
#x_train = np.random.random((N_TRAIN, TOPK, N_FEATURES))
#y_train = ((np.random.random((N_TRAIN, TOPK, 1)) * 42) % 2).astype(int)

#x_test = np.random.random((N_TEST, TOPK, N_FEATURES))
#y_test = ((np.random.random((N_TEST, TOPK, 1)) * 42) % 2).astype(int)
x_test = np.reshape(x_test, (N_TEST//TOPK, TOPK, N_FEATURES))
y_test = np.reshape(y_test, (N_TEST//TOPK, TOPK, 1))


RESULT_PATH = "/var/scratch2/uji300/OpenKE-results/"
if mode == "train":
    model = Sequential()
    model.add(LSTM(1, input_shape=(TOPK, N_FEATURES),return_sequences=True))

    model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)#,validation_data=(x_test, y_test))
    #predict = model.predict(x_test)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    json_model = model.to_json()
    with open(RESULT_PATH +"fb15k237-lstm-model.json", 'w') as fout:
        fout.write(json_model)

    model.save_weights(RESULT_PATH + "fb15k237-lstm-weights.h5")
    print("Saved model to disk")
elif mode == "test":
    with open(RESULT_PATH + "fb15k237-lstm-model.json", 'r') as fin:
        file_model = fin.read()

    loaded_model = model_from_json(file_model)
    loaded_model.load_weights(RESULT_PATH+"fb15k237-lstm-weights.h5")
    loaded_model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
    score = loaded_model.evaluate(x_test, y_test, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
else:
    print("Options are \"train\" and \"test\"")
