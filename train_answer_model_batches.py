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

from data_generator import DataGenerator
import glob

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--infile', dest ='infile', required = True, type = str, help = 'File containing training/test data with labels.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--units', dest = 'units', type = int, default = 100)
    parser.add_argument('-bs', '--batch-size', dest = 'batch_size', type = int, default=2)
    parser.add_argument('--epochs', dest = 'epochs', type = int, default = 100)
    parser.add_argument('--dropout', dest = 'dropout', type = float, default = 0.5)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--emb-model', required = True, dest = 'emb_model', type = str, default = "transe")
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
emb_model = args.emb_model

all_files = glob.glob(input_file + "*training-topk-"+str(topk) +"*" + type_prediction + "*.pkl")
num_files = len(all_files)
num_valid_files = max((int)(0.05 * num_files), 2)
marker = num_files - num_valid_files

print("num_files = ", num_files)
print("valid_file = ", num_valid_files)
print("marker = ", marker)
id_list = np.arange(1, num_files + 1)

train_ids = id_list[:marker]
valid_ids = id_list[marker:]

#TODO: deduce from data
N_FEATURES=605

params = {
    'db'        : args.db,
    'emb_model' : emb_model,
    'topk'      : topk,
    'dim_x'     : (1000, topk, N_FEATURES),
    'dim_y'     : (1000, topk, 1),
    'batch_size': int(args.batch_size),
    'n_classes' : 2,
    'shuffle'   : True
    }

partition = {}
partition['train'] = train_ids
partition['valid'] = valid_ids

training_generator = DataGenerator(partition['train'], input_file, type_prediction, **params)
valid_generator    = DataGenerator(partition['valid'], input_file, type_prediction, **params)

# Model
model = Sequential();
if model_str == "lstm":
    model.add(LSTM(n_units, input_shape=(topk, N_FEATURES), batch_input_shape=(1000 * int(args.batch_size), topk, N_FEATURES), return_sequences = True));
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


model.fit_generator(generator=training_generator, validation_data = valid_generator, use_multiprocessing = True, epochs = n_epochs, workers = 8)

#Saving of model and weights
json_model = model.to_json()

base_name = args.db + "-" + args.emb_model + "-training-topk-" + str(args.topk)
model_file_name = result_dir + "models/" + base_name + "-" + type_prediction + "-model-"+ model_str + "-units-"+str(n_units) + \
"-dropout-" + str(dropout) +".json"
with open(model_file_name, 'w') as fout:
    fout.write(json_model)

model_weights_file_name = result_dir + "models/" + base_name + "-" + type_prediction + "-weights-"+ model_str + "-units-"+str(n_units) + \
"-dropout-" + str(dropout) +".h5"

model.save_weights(model_weights_file_name)
print("Saved model to disk")
