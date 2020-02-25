from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

import numpy as np
import sys
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--infile', dest ='infile', required = True, type = str, help = 'File containing training/test data with labels.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--mode', required = True, dest = 'mode', type = str, default = "test")
    parser.add_argument('--pred', dest ='pred', type = str, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    return parser.parse_args()

args = parse_args()

# Paths
db_path = "./benchmarks/" + args.db + "/"
result_dir      = "/var/scratch2/uji300/OpenKE-results/" + args.db + "/"

input_file = args.infile
mode = args.mode
topk = args.topk
type_prediction = args.pred

if mode == "train":
    '''
    # Working code:
    model = Sequential()
    model.add(LSTM(1, input_shape=(topk, N_FEATURES),return_sequences=True))

    model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)#,validation_data=(x_test, y_test))
    #predict = model.predict(x_test)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    '''
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

    # DIM = 200 for each entity/relation

    N_FEATURES = len(x_train[0])

    print("N_FEATURES = ", N_FEATURES)
    x_train = np.reshape(x_train, (N_TRAIN//topk, topk, N_FEATURES))
    y_train = np.reshape(y_train, (N_TRAIN//topk, topk, 1))

    x_valid = np.reshape(x_valid, (N_VALID//topk, topk, N_FEATURES))
    y_valid = np.reshape(y_valid, (N_VALID//topk, topk, 1))

    # Model
    model = Sequential();
    model.add(LSTM(100, input_shape=(topk, N_FEATURES), return_sequences = True));
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = 'sigmoid'))
    model.add(Dropout(0.1))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 2, validation_data=(x_valid, y_valid))

    #Saving of model and weights
    json_model = model.to_json()
    model_file_name = result_dir + args.db + "-lstm-model-"+str(topk)+"-"+type_prediction+".json"
    with open(model_file_name, 'w') as fout:
        fout.write(json_model)

    model_weights_file_name = result_dir + args.db + "-lstm-weights-"+str(topk)+"-"+type_prediction+".h5"
    model.save_weights(model_weights_file_name)
    print("Saved model to disk")
elif mode == "test":
    print("Loading test data...", end = " ")
    with open(input_test_file, "r") as fin:
        data = json.loads(fin.read())
    print("DONE")
    print(type(data))
    print(data.keys())
    SAMPLE_SIZE = len(data['x_' + type_prediction])
    print("Size of all test triples = ", len(data['x_'+ type_prediction]))
    all_data_x = data['x_' + type_prediction][:SAMPLE_SIZE]
    all_data_y = data['y_' + type_prediction][:SAMPLE_SIZE]
    N_TEST = len(all_data_x)

    model_file_name = result_dir + args.db + "-lstm-model-"+str(topk)+"-"+type_prediction+".json"
    with open(model_file_name, 'r') as fin:
        file_model = fin.read()

    loaded_model = model_from_json(file_model)
    model_weights_file_name = result_dir + args.db + "-lstm-weights-"+str(topk)+"-"+type_prediction+".h5"
    loaded_model.load_weights(model_weights_file_name)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    x_test = np.array(all_data_x)
    y_test = np.array(all_data_y, dtype=np.int32)
    N_FEATURES = len(x_test[0])
    x_test = np.reshape(x_test, (N_TEST//topk, topk, N_FEATURES))
    y_test = np.reshape(y_test, (N_TEST//topk, topk, 1))
    score = loaded_model.evaluate(x_test, y_test, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    print(loaded_model.metrics_names)
    predicted = loaded_model.predict_classes(x_test)
    fp_predicted = predicted.flatten().astype(np.int32)
    print("fp_predicted :", len(fp_predicted))
    print(fp_predicted)
    fpy = y_test.flatten().astype(np.int32)
    a = (fp_predicted[np.array(fpy) == 1] == 1).sum()
    print("# of 1s guessed : ", list(fp_predicted).count(1))
    print("# of 0s guessed : ", list(fp_predicted).count(0))
    b = (np.array(fpy) == 1).sum()
    c = (fp_predicted[np.array(fpy) == 0] == 0).sum()
    d = (np.array(fpy) == 0).sum()
    print("Predicting 1s   : ", a, " / ", b, a/b)
    print("Predicting 0s   : ", c, " / ", d, c/d)
    print("Predicting both : ", a+c, b+d, (a+c)/(b+d))
else:
    print("Options are \"train\" and \"test\"")
