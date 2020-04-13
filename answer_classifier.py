from abc import ABC, abstractmethod
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class AnswerClassifier(ABC):

    def __init__(self, type_prediction, test_triples_file):
        self.x_test_raw = None
        self.y_test_raw = None
        self.x_test_fil = None
        self.y_test_fil = None
        self.entity_dict = None
        self.relation_dict = None
        self.logfile = None

        self.y_predicted_raw = []
        self.y_predicted_fil = []
        self.type_prediction = type_prediction
        self.init_test_triples(test_triples_file)


    def init_test_triples(self, test_triples_file):
        # Read the test file
        # A test triples file has 3 + 2 +  200*3 features where first three features are
        # h <SPACE> r <SPACE> ans1 + rank  + score + followed by h_bar + r_bar + ans1_bar
        with open(test_triples_file, "rb") as fin:
            data = pickle.load(fin)
        self.x_test_raw = np.array(data['x_' + self.type_prediction + "_raw"])
        self.y_test_raw = np.array(data['y_' + self.type_prediction + "_raw"], dtype = np.int32)
        self.x_test_fil = np.array(data['x_' + self.type_prediction + "_fil"])
        self.y_test_fil = np.array(data['y_' + self.type_prediction + "_fil"], dtype = np.int32)
        self.cnt_test_triples = len(self.x_test_raw)
        print("Size of all test triples = ", self.cnt_test_triples)
        self.emb_dim = len(self.x_test_raw[0])

    def init_entity_dict(self, entity_dict_file, rel_dict_file):
        with open(entity_dict_file, 'rb') as fin:
            self.entity_dict = pickle.load(fin)

        with open(rel_dict_file, 'rb') as fin:
            self.relation_dict = pickle.load(fin)


    @abstractmethod
    def set_logfile(self, logfile):
        pass

    @abstractmethod
    def print_answer_entities(self):
        '''
            print stringized entities from ids in the log
        '''
        pass

    @abstractmethod
    def predict(self):
        '''
        For a single test triple (h, r, ans1):

        1. SubgraphClassifier would work as follows
        Use x_test , compute h * r (where * is the operator used by the embedding model)
        Find topK subgraphs, and check if they contain the ans1

        2. LSTMClassifier would work as follows
        Use pre-trained model using training triples

        Both fill y_predicted_raw and y_predicted_fil in the end
        '''
        pass

    def results(self):
        print("#" * 20 + "   RAW   " + "#" * 20)
        raw_cnt_tuple = np.unique(self.y_predicted_raw, return_counts = True)
        print("# of predicted : ", raw_cnt_tuple)
        raw_conf_mat = confusion_matrix(self.y_test_raw, self.y_predicted_raw)
        print(raw_conf_mat)
        raw_result = classification_report(self.y_test_raw, self.y_predicted_raw, output_dict = True)
        raw_result['predicted_cnt'] = {}
        raw_result['predicted_cnt']['0'] = raw_cnt_tuple[1][0]
        raw_result['predicted_cnt']['1'] = raw_cnt_tuple[1][1]
        raw_result['TP'] = raw_conf_mat[0][0]
        raw_result['FP'] = raw_conf_mat[0][1]
        raw_result['FN'] = raw_conf_mat[1][0]
        raw_result['TN'] = raw_conf_mat[1][1]
        print(classification_report(self.y_test_raw, self.y_predicted_raw))

        print("#" * 20 + "   FILTERED   " + "#" * 20)
        fil_cnt_tuple = np.unique(self.y_predicted_fil, return_counts = True)
        print("# of predicted : ", fil_cnt_tuple)
        fil_conf_mat = confusion_matrix(self.y_test_fil, self.y_predicted_fil)
        print(fil_conf_mat)
        filtered_result = classification_report(self.y_test_fil, self.y_predicted_fil, output_dict = True)
        filtered_result['predicted_cnt'] = {}
        filtered_result['predicted_cnt']['0'] = fil_cnt_tuple[1][0]
        filtered_result['predicted_cnt']['1'] = fil_cnt_tuple[1][1]
        filtered_result['TP'] = fil_conf_mat[0][0]
        filtered_result['FP'] = fil_conf_mat[0][1]
        filtered_result['FN'] = fil_conf_mat[1][0]
        filtered_result['TN'] = fil_conf_mat[1][1]
        print(classification_report(self.y_test_fil, self.y_predicted_fil))

        print("*" * 80)
        if self.entity_dict is not None and self.relation_dict is not None:
            self.print_answer_entities()
        return raw_result, filtered_result
