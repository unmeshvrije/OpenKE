import numpy as np
from keras.models import model_from_json
from answer_classifier import AnswerClassifier
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm

class MLPClassifierSupervised(AnswerClassifier):
    def __init__(self, type_prediction, db, topk, queries_file_path, emb_model, model_file_path, model_weights_path, training_file):
        super(MLPClassifierSupervised, self).__init__(type_prediction, queries_file_path, db, emb_model, topk)
        self.model_file_path = model_file_path
        self.model_weights_path = model_weights_path
        self.training_file = training_file

    def set_logfile(self, logfile):
        self.logfile = logfile

    def print_answer_entities(self):
        if self.logfile == None:
            return
        log = open(self.logfile, "w")
        for index, x in enumerate(self.test_queries_answers["fil"]):
            e = int(x[0])
            r = int(x[1])
            a = int(x[2])
            head = e
            tail = a
            if self.type_prediction == "head":
                head = a
                tail = e
            if self.y_test_fil[index] == 1 and self.y_predicted_fil[index] == 0:
                print("$$Expected (1) Predicted (0): $", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail], "$$$", file=log)
            if self.y_predicted_fil[index] == 1 and self.y_test_fil[index] == 0:
                print("**Expected (0) Predicted (1): * ", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail] , " ***", file=log)
            if self.y_predicted_fil[index] == 1 and self.y_test_fil[index] == 1:
                print("##Expected (1) Predicted (1): # ", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail] , " ###", file=log)
            if self.y_predicted_fil[index] == 0 and self.y_test_fil[index] == 0:
                print("##Expected (0) Predicted (0): # ", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail] , " ###", file=log)
            if (index+1) % self.topk == 0:
                print("*" * 80, file = log)

        log.close()

    def load_training_data(self):
        print("Loading training data...", self.training_file, end = " ")
        with open(self.training_file, "rb") as fin:
            training_data = pickle.load(fin)
        print("DONE")

        SAMPLE_SIZE = len(training_data['x_' + self.type_prediction])
        print("Sample size for the total data = ", SAMPLE_SIZE)
        all_data_x = training_data['x_' + self.type_prediction][:SAMPLE_SIZE]
        all_data_y = training_data['y_' + self.type_prediction][:SAMPLE_SIZE]
        N_TOTAL = len(all_data_x)
        N_VALID = 10000
        N_TRAIN = int(N_TOTAL - N_VALID)

        print("N_TOTAL = ", N_TOTAL)
        print("N_TRAIN = ", N_TRAIN)
        print("N_TEST  =", N_VALID)
        print("#"*80)

        x_train = np.array(all_data_x[:N_TRAIN])
        y_train = np.array(all_data_y[:N_TRAIN], dtype=np.int32)

        N_FEATURES = len(x_train[0])

        print("N_FEATURES = ", N_FEATURES)
        x_train = np.reshape(x_train, (N_TRAIN//self.topk, self.topk, N_FEATURES))
        y_train = np.reshape(y_train, (N_TRAIN//self.topk, self.topk, 1))
        return x_train, y_train

    def predict(self):
        with open(self.model_file_path, 'r') as fin:
            file_model = fin.read()

        loaded_model = model_from_json(file_model)
        loaded_model.load_weights(self.model_weights_path)
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

        # Raw
        x_test_raw = np.reshape(self.x_test_raw, (self.cnt_test_triples//self.topk, self.topk, self.emb_dim))
        predicted_raw = loaded_model.predict_classes(x_test_raw)
        self.y_predicted_raw = predicted_raw.flatten().astype(np.int32)

        # Filtered
        x_test_fil = np.reshape(self.x_test_fil, (self.cnt_test_triples//self.topk, self.topk, self.emb_dim))
        #if self.threshold != 0.5:
        #TODO: find probabilities of model.predict(x_train)
        # Train the LR on that.

        x_train, y_train = self.load_training_data()
        x_train_probabilities = loaded_model.predict(x_train)

        #lr_model = LogisticRegression(solver = 'lbfgs')
        lr_model = svm.SVC() #LogisticRegression(solver = 'lbfgs')
        cur_shape = x_train_probabilities.shape
        x_train_probabilities = np.reshape(x_train_probabilities, (cur_shape[0]*cur_shape[1], 1))
        y_train = np.reshape(y_train, (cur_shape[0] * cur_shape[1], 1))
        lr_model.fit(x_train_probabilities, y_train.ravel())

        x_test_probabilities = loaded_model.predict(x_test_fil)
        cur_shape_test = x_test_probabilities.shape
        x_test_probabilities = np.reshape(x_test_probabilities, (cur_shape_test[0]*cur_shape_test[1], 1))
        self.y_predicted_fil = lr_model.predict(x_test_probabilities)
        self.y_predicted_fil_abs = self.y_predicted_fil

