from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
import os
import numpy as np
import sys
import json
import argparse
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--infile', dest ='infile', required = True, type = str, help = 'File containing training/test data with labels.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--units', dest = 'units', type = int, default = 100)
    parser.add_argument('-bs', '--batch-size', dest = 'batch_size', type = int)
    parser.add_argument('--epochs', dest = 'epochs', type = int, default = 50)
    parser.add_argument('--dropout', dest = 'dropout', type = float, default = 0.5)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--model', required = True, dest = 'model_str', type = str, default = "lstm")
    parser.add_argument('--pred', dest ='pred', required = True, type = str, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    return parser.parse_args()

args = parse_args()

# DIM = 200 for each entity/relation
# Paths
db_path = "./benchmarks/" + args.db + "/"
result_dir =  args.result_dir + args.db + "/"
os.makedirs(result_dir, exist_ok = True)

input_file = args.infile
topk = int(args.topk)
type_prediction = args.pred
n_units = int(args.units)
dropout = float(args.dropout)
model_str = args.model_str
n_epochs = args.epochs
if args.batch_size == None:
    batch_size = int(args.topk) * 10
else:
    batch_size = int(args.batch_size)

print("Loading training data...", input_file, end = " ")
with open(input_file, "rb") as fin:
    training_data = pickle.load(fin)
print("DONE")

SAMPLE_SIZE = len(training_data['x_' + type_prediction])
print("Sample size for the total data = ", SAMPLE_SIZE)
all_data_x = training_data['x_' + type_prediction][:SAMPLE_SIZE]
all_data_y = training_data['y_' + type_prediction][:SAMPLE_SIZE]
all_data_y = np.array(all_data_y)
N_TOTAL = len(all_data_x)
N_VALID = int(N_TOTAL * 0.20)
N_TRAIN = int(N_TOTAL - N_VALID)

print("N_TOTAL = ", N_TOTAL)
print("N_TRAIN = ", N_TRAIN)
print("N_VALID = ", N_VALID)
print("#"*80)

randomChoice = np.random.choice(N_TOTAL // args.topk, N_VALID // args.topk, replace=False)
randomIdxs = np.zeros(N_TOTAL, dtype=np.bool)
for i in range(len(randomChoice)):
    startIdx = randomChoice[i] * args.topk
    for j in range(args.topk):
        randomIdxs[startIdx + j] = True

print("rand indexes : ", len(randomIdxs))
x_train = np.array(all_data_x[~randomIdxs])
print("x_Train len : ", len(x_train))
y_train = np.array(all_data_y[~randomIdxs], dtype=np.int32)

x_valid = np.array(all_data_x[randomIdxs])
y_valid = np.array(all_data_y[randomIdxs], dtype=np.int32)

N_FEATURES = len(x_train[0])

print("N_FEATURES = ", N_FEATURES)
x_train = np.reshape(x_train, (N_TRAIN//topk, topk, N_FEATURES))
y_train = np.reshape(y_train, (N_TRAIN//topk, topk, 1))

x_valid = np.reshape(x_valid, (N_VALID//topk, topk, N_FEATURES))
y_valid = np.reshape(y_valid, (N_VALID//topk, topk, 1))

# Model
model = Sequential()
if model_str == "lstm":
    model.add(LSTM(n_units, input_shape=(topk, N_FEATURES), return_sequences = True))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = 'sigmoid'))
elif model_str == "mlp":
    model.add(Dense(n_units, input_shape=(topk, N_FEATURES)));
    model.add(Dropout(dropout))
    model.add(Dense(n_units))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = 'sigmoid'))

# For classification problem (with 2 classes), binary cross entropy is used.
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs = n_epochs, batch_size = batch_size, verbose = 2, validation_data=(x_valid, y_valid))
score = model.evaluate(x_valid, y_valid, verbose=1)
print("Validation set : %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#Saving of model and weights
json_model = model.to_json()

base_name = os.path.basename(input_file).split('.')[0]
model_file_name = result_dir + "models/" + base_name + "-" + type_prediction + "-model-"+ model_str + "-units-"+str(n_units) + \
"-dropout-" + str(dropout) +".json"
with open(model_file_name, 'w') as fout:
    fout.write(json_model)

model_weights_file_name = result_dir + "models/" + base_name + "-" + type_prediction + "-weights-"+ model_str + "-units-"+str(n_units) + \
"-dropout-" + str(dropout) +".h5"

model.save_weights(model_weights_file_name)
print("Saved model to disk")
