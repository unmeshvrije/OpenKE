from abc import ABC, abstractmethod
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class AnswerClassifier(ABC):

    def __init__(self, type_prediction, test_triples_file):
        self.x_test = None
        self.y_test = None
        self.y_test_filtered = None
        self.y_predicted = None
        self.type_prediction = type_prediction
        self.init_test_triples(test_triples_file)

    # Use for subgraphs only => move to subgraphs classifier
    def init_embeddings(embeddings_pickle_file):
        # Process pickle file and initialize
        #self.E
        #self.R
        pass

    def init_test_triples(self, test_triples_file):
        # Read the test file
        # A test triples file has 3 + 200*3 features where first three features are
        # h <SPACE> r <SPACE> ans1 followed by h_bar + r_bar + ans1_bar
        with open(test_triples_file, "rb") as fin:
            data = pickle.load(fin)
        SAMPLE_SIZE = len(data['x_' + self.type_prediction])
        print("Size of all test triples = ", len(data['x_'+ self.type_prediction]))
        self.x_test = np.array(data['x_' + self.type_prediction])
        self.y_test = np.array(data['y_' + self.type_prediction], dtype = np.int32)
        self.y_test_filtered = np.array(data['y_' + self.type_prediction + '_filtered'], dtype = np.int32)
        self.cnt_test_triples = len(self.x_test)
        self.emb_dim = len(self.x_test[0])

    @abstractmethod
    def predict(self):
        '''
        For a single test triple (h, r, ans1):

        1. SubgraphClassifier would work as follows
        Use x_test , compute h * r (where * is the operator used by the embedding model)
        Find topK subgraphs, and check if they contain the ans1

        2. LSTMClassifier would work as follows
        Use pre-trained model using training triples

        Both fill y_predicted in the end
        '''
        pass

    def results(self):
        print(confusion_matrix(self.y_test, self.y_predicted))
        print(classification_report(self.y_test, self.y_predicted))
        print("*" * 80)
        print(confusion_matrix(self.y_test_filtered, self.y_predicted))
        print(classification_report(self.y_test_filtered, self.y_predicted))
