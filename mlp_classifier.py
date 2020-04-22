import numpy as np
from keras.models import model_from_json
from answer_classifier import AnswerClassifier

class MLPClassifier(AnswerClassifier):
    def __init__(self, type_prediction, db, topk, queries_file_path, emb_model, model_file_path, model_weights_path, threshold = 0.5):
        super(MLPClassifier, self).__init__(type_prediction, queries_file_path, db, emb_model, topk)
        self.model_file_path = model_file_path
        self.model_weights_path = model_weights_path
        self.threshold = threshold

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

    def predict(self):
        with open(self.model_file_path, 'r') as fin:
            file_model = fin.read()

        loaded_model = model_from_json(file_model)
        loaded_model.load_weights(self.model_weights_path)
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

        if self.use_generator:
            # evaluate_generator() works but it gives confusion matrix as the output
            # we want to have true labels (y_test_raw) that are already read by the generator
            # and then predict_generator()
            probabilities_raw    = loaded_model.predict_generator(self.test_generator["raw"])
            predicted_raw        = (probabilities_raw > self.threshold).astype(int)
            self.y_predicted_raw = predicted_raw.flatten().astype(np.int32)
            self.y_test_raw      = self.test_labels["raw"]

            probabilities_fil    = loaded_model.predict_generator(self.test_generator["fil"])
            predicted_fil        = (probabilities_fil > self.threshold).astype(int)
            self.y_predicted_fil = predicted_fil.flatten().astype(np.int32)
            self.y_test_fil      = self.test_labels["fil"]
        else:
            # Raw
            x_test_raw = np.reshape(self.x_test_raw, (self.cnt_test_triples//self.topk, self.topk, self.emb_dim))
            predicted_raw = loaded_model.predict_classes(x_test_raw)
            self.y_predicted_raw = predicted_raw.flatten().astype(np.int32)

            # Filtered
            x_test_fil = np.reshape(self.x_test_fil, (self.cnt_test_triples//self.topk, self.topk, self.emb_dim))
            if self.threshold != 0.5:
                probabilities = loaded_model.predict(x_test_fil)
                predicted_fil = (probabilities > self.threshold).astype(int)
                self.y_predicted_fil = predicted_fil.flatten().astype(np.int32)
            else:
                predicted_fil = loaded_model.predict_classes(x_test_fil)
                self.y_predicted_fil = predicted_fil.flatten().astype(np.int32)
