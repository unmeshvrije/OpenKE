from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

import numpy as np
import sys
import json

input_train_file = sys.argv[1]
input_test_file  = sys.argv[1]
#lstm_training_file = "/var/scratch2/uji300/OpenKE-results/fb15k237-training-topk-"+str(topk)+".json"
mode = sys.argv[2]
TOPK = int(sys.argv[3]) #int(input_file.split('-')[4].split('.')[0])
type_prediction = sys.argv[4] # can only be head or tail

TRAINING_SAMPLE_SIZE = 50000
TEST_SAMPLE_SIZE = 50000
percent_validation = 10
RESULT_PATH = "/var/scratch2/uji300/OpenKE-results/"


if mode == "train":
    '''
    # Working code:
    model = Sequential()
    model.add(LSTM(1, input_shape=(TOPK, N_FEATURES),return_sequences=True))

    model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)#,validation_data=(x_test, y_test))
    #predict = model.predict(x_test)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    '''
    print("Loading training data...", end = " ")
    with open(input_train_file, "r") as fin:
        training_data = json.loads(fin.read())
    print("DONE")

    SAMPLE_SIZE = min(TRAINING_SAMPLE_SIZE, len(training_data['x_' + type_prediction]))
    print("Sample size for the total data = ", SAMPLE_SIZE)
    all_data_x = training_data['x_' + type_prediction][:SAMPLE_SIZE]
    all_data_y = training_data['y_' + type_prediction][:SAMPLE_SIZE]
    N_TOTAL = len(all_data_x)
    N_VALID = int(N_TOTAL * percent_validation / 100)
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
    x_train = np.reshape(x_train, (N_TRAIN//TOPK, TOPK, N_FEATURES))
    y_train = np.reshape(y_train, (N_TRAIN//TOPK, TOPK, 1))

    x_valid = np.reshape(x_valid, (N_VALID//TOPK, TOPK, N_FEATURES))
    y_valid = np.reshape(y_valid, (N_VALID//TOPK, TOPK, 1))


    # Model
    model = Sequential();
    model.add(LSTM(100, input_shape=(TOPK, N_FEATURES), return_sequences = True));
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = 'sigmoid'))
    model.add(Dropout(0.1))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 2, validation_data=(x_valid, y_valid))
    '''
    score = model.evaluate(x_test, y_test, verbose = 1)
    predicted = model.predict_classes(x_test)
    fp_predicted = predicted.flatten().astype(np.int32)
    print("fp_predicted :", len(fp_predicted))
    print(fp_predicted)
    fpy = y_test.flatten().astype(np.int32)
    a = (fp_predicted[np.array(fpy) == 1] == 1).sum()
    print("# of 1s guessed : ", list(fp_predicted).count(1))
    print("# of 1s expected: ", list(fpy).count(1))
    b = (np.array(fpy) == 1).sum()
    c = (fp_predicted[np.array(fpy) == 0] == 0).sum()
    d = (np.array(fpy) == 0).sum()
    print("Predicting 1s   : ", a, b, a/b)
    print("Predicting 0s   : ", c, d, c/d)
    print("Predicting both : ", a+c, b+d, (a+c)/(b+d))
    '''

    #Saving of model and weights
    json_model = model.to_json()
    model_file_name = "fb15k237-lstm-model-"+str(TOPK)+"-"+type_prediction+".json"
    with open(RESULT_PATH + model_file_name, 'w') as fout:
        fout.write(json_model)

    model_weights_file_name = "fb15k237-lstm-weights-"+str(TOPK)+"-"+type_prediction+".h5"
    model.save_weights(RESULT_PATH + model_weights_file_name)
    print("Saved model to disk")
elif mode == "test":
    print("Loading test data...", end = " ")
    with open(input_test_file, "r") as fin:
        data = json.loads(fin.read())
    print("DONE")
    SAMPLE_SIZE = min(TEST_SAMPLE_SIZE, len(data['x_' + type_prediction]))
    print("Sample size for the considered triples = ", SAMPLE_SIZE)
    print("Sample size of all test triples = ", len(data['x_'+ type_prediction]))
    all_data_x = data['x_' + type_prediction][:SAMPLE_SIZE]
    all_data_y = data['y_' + type_prediction][:SAMPLE_SIZE]
    N_TEST = len(all_data_x)

    model_file_name = "fb15k237-lstm-model-"+str(TOPK)+"-"+type_prediction+".json"
    with open(RESULT_PATH + model_file_name, 'r') as fin:
        file_model = fin.read()

    loaded_model = model_from_json(file_model)
    model_weights_file_name = "fb15k237-lstm-weights-"+str(TOPK)+"-"+type_prediction+".h5"
    loaded_model.load_weights(RESULT_PATH + model_weights_file_name)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    x_test = np.array(all_data_x)
    y_test = np.array(all_data_y, dtype=np.int32)
    N_FEATURES = len(x_test[0])
    x_test = np.reshape(x_test, (N_TEST//TOPK, TOPK, N_FEATURES))
    y_test = np.reshape(y_test, (N_TEST//TOPK, TOPK, 1))
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
    print("# of 1s expected: ", list(fpy).count(1))
    b = (np.array(fpy) == 1).sum()
    c = (fp_predicted[np.array(fpy) == 0] == 0).sum()
    d = (np.array(fpy) == 0).sum()
    print("Predicting 1s   : ", a, b, a/b)
    print("Predicting 0s   : ", c, d, c/d)
    print("Predicting both : ", a+c, b+d, (a+c)/(b+d))
else:
    print("Options are \"train\" and \"test\"")
