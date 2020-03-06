import numpy as np
from keras.models import model_from_json
from answer_classifier import AnswerClassifier

class MLPClassifier(AnswerClassifier):
    def __init__(self, type_prediction, topk, triples_file_path, model_file_path, model_weights_path):
        super(MLPClassifier, self).__init__(type_prediction, triples_file_path)
        self.model_file_path = model_file_path
        self.model_weights_path = model_weights_path
        self.topk = topk

    def predict(self):
        with open(self.model_file_path, 'r') as fin:
            file_model = fin.read()

        loaded_model = model_from_json(file_model)
        loaded_model.load_weights(self.model_weights_path)
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        x_test = np.reshape(self.x_test, (self.cnt_test_triples//self.topk, self.topk, self.emb_dim))
        predicted = loaded_model.predict_classes(x_test)
        self.y_predicted = predicted.flatten().astype(np.int32)
