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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--infile', dest ='infile', required = True, type = str, help = 'File containing training/test data with labels.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--units', dest = 'units', type = int, default = 100)
    parser.add_argument('-bs', '--batch-size', dest = 'batch_size', type = int)
    parser.add_argument('--epochs', dest = 'epochs', type = int, default = 100)
    parser.add_argument('--dropout', dest = 'dropout', type = float, default = 0.5)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--model', required = True, dest = 'model_str', type = str, default = "lstm")
    parser.add_argument('--mode', required = True, dest = 'mode', type = str, default = "test")
    parser.add_argument('--pred', dest ='pred', type = str, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    return parser.parse_args()

args = parse_args()

# DIM = 200 for each entity/relation
# Paths
db_path = "./benchmarks/" + args.db + "/"
result_dir =  args.result_dir + args.db + "/"
os.makedirs(result_dir, exist_ok = True)

input_file = args.infile
mode = args.mode
topk = int(args.topk)
type_prediction = args.pred
n_units = int(args.units)
dropout = float(args.dropout)
model_str = args.model
n_epochs = args.epochs
if args.batch_size == None:
    batch_size = int(args.topk) * 10
else:
    batch_size = int(args.batch_size)

def percentify(num):
    return str(round(float(num)*100, 2)) + "%"

if mode == "train":
    print("Loading training data...", end = " ")
    with open(input_file, "r") as fin:
        training_data = json.loads(fin.read())
    print("DONE")

    SAMPLE_SIZE = len(training_data['x_' + type_prediction])
    print("Sample size for the total data = ", SAMPLE_SIZE)
    all_data_x = training_data['x_' + type_prediction][:SAMPLE_SIZE]
    all_data_y = training_data['y_' + type_prediction][:SAMPLE_SIZE]
    N_TOTAL = len(all_data_x)
    N_VALID = 10000
    N_TRAIN = int(N_TOTAL - N_VALID)

    print("N_TOTAL = ", N_TOTAL)
    print("N_TRAIN = ", N_TRAIN)
    print("N_TEST  =", N_VALID)
    print("#"*80)

    x_train = np.array(all_data_x[:N_TRAIN])
    y_train = np.array(all_data_y[:N_TRAIN], dtype=np.int32)

    x_valid = np.array(all_data_x[N_TRAIN:])
    y_valid = np.array(all_data_y[N_TRAIN:], dtype=np.int32)

    N_FEATURES = len(x_train[0])

    print("N_FEATURES = ", N_FEATURES)
    x_train = np.reshape(x_train, (N_TRAIN//topk, topk, N_FEATURES))
    y_train = np.reshape(y_train, (N_TRAIN//topk, topk, 1))

    x_valid = np.reshape(x_valid, (N_VALID//topk, topk, N_FEATURES))
    y_valid = np.reshape(y_valid, (N_VALID//topk, topk, 1))

    # Model
    model = Sequential();
    if model_str == "lstm":
        model.add(LSTM(n_units, input_shape=(topk, N_FEATURES), return_sequences = True));
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
    model_file_name = result_dir + args.db + "-" + model_str + "-model-"+str(topk)+"-"+type_prediction+str(n_units) + \
    "-units"+ "dropout" + str(dropout) +".json"
    with open(model_file_name, 'w') as fout:
        fout.write(json_model)

    model_weights_file_name = result_dir + args.db + "-" + model_str + "-weights-"+str(topk)+"-"+type_prediction+".h5"
    model.save_weights(model_weights_file_name)
    print("Saved model to disk")
elif mode == "test":
    print("Loading test data...", end = " ")
    with open(input_file, "r") as fin:
        data = json.loads(fin.read())
    print("DONE")
    print(type(data))
    print(data.keys())
    SAMPLE_SIZE = len(data['x_' + type_prediction])
    print("Size of all test triples = ", len(data['x_'+ type_prediction]))
    all_data_x = data['x_' + type_prediction][:SAMPLE_SIZE]
    all_data_y = data['y_' + type_prediction][:SAMPLE_SIZE]
    N_TEST = len(all_data_x)

    model_file_name = result_dir + args.db + "-" + model_str + "-model-"+str(topk)+"-"+type_prediction+str(n_units) + \
    "-units"+ "dropout" + str(dropout) +".json"
    with open(model_file_name, 'r') as fin:
        file_model = fin.read()

    loaded_model = model_from_json(file_model)
    model_weights_file_name = result_dir + args.db + "-" + model_str + "-weights-"+str(topk)+"-"+type_prediction+".h5"
    base_file_name = os.path.basename(input_file).split('.')[0]
    result_file_name = result_dir + base_file_name + "-" + type_prediction + "-units-"+str(n_units)+"-dropout-"+str(dropout)+".out"
    result_file = open(result_file_name, "w")
    loaded_model.load_weights(model_weights_file_name)
    # Compiling a model is mandatory before training/testing (Runtime error otherwise)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    x_test = np.array(all_data_x)
    y_test = np.array(all_data_y, dtype=np.int32)
    N_FEATURES = len(x_test[0])
    x_test = np.reshape(x_test, (N_TEST//topk, topk, N_FEATURES))
    y_test = np.reshape(y_test, (N_TEST//topk, topk, 1))

    predicted = loaded_model.predict_classes(x_test)
    fp_predicted = predicted.flatten().astype(np.int32)
    fpy = y_test.flatten().astype(np.int32)

    a = (fp_predicted[np.array(fpy) == 1] == 1).sum()
    b = (np.array(fpy) == 1).sum()
    c = (fp_predicted[np.array(fpy) == 0] == 0).sum()
    d = (np.array(fpy) == 0).sum()

    print("# of 1s predicted by the model : ", list(fp_predicted).count(1), file = result_file)
    print("# of 0s predicted by the model : ", list(fp_predicted).count(0), file = result_file)
    print("Predicting 1s   : ", a, " / ", b, " ( ", percentify(a/b), " ) ", file = result_file)
    print("Predicting 0s   : ", c, " / ", d, " ( ", percentify(c/d), " ) ", file = result_file)
    print("Predicting both : ", a+c, " / ", b+d, " ( ", percentify((a+c)/(b+d)), " ) ", file = result_file)
    print(confusion_matrix(fpy, fp_predicted), file = result_file)
    print(classification_report(fpy, fp_predicted), file = result_file)
else:
    print("Options are \"train\" and \"test\"")
