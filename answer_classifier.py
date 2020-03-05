from abc import ABC, abstractmethod

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class AnswerClassifier(ABC):

    def __init__(self):
        self.y_test = None

    def init_embeddings(embeddings_pickle_file):
        # Process pickle file and initialize
        #self.E
        #self.R
        pass

    def init_test_triples(test_triples_text_file):
        # Read the test file
        # A test triples text file looks like below
        # h <SPACE> r <SPACE> ans1
        # h <SPACE> r <SPACE> ans2
        # h <SPACE> r <SPACE> ...
        # h <SPACE> r <SPACE> ansK
        #self.x_test
        #self.y_test
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

        Both generate y_predicted in the end
        '''
        self.y_predicted
        pass

    def results(self):
        print(confusion_matrix(self.y_test, self.y_predicted))
        print(classification_report(self.y_test, self.y_predicted))
