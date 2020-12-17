from classifier import Classifier
from abc import abstractmethod
from support.embedding_model import Embedding_Model

class Supervised_Classifier(Classifier):

    def __init__(self, dataset, type_prediction : {'head', 'tail'}, results_dir, embedding_model : Embedding_Model):
        super(Supervised_Classifier, self).__init__(dataset, type_prediction, results_dir)
        self.embedding_model = embedding_model

    @abstractmethod
    def create_training_data(self, queries_with_answers):
        """
        :param answers: These are a set of queries with related answers. They are produced by create_answers.py
        :return: For each query and answers, this methods creates a set of features and label that will be used
        to train the model
        """
        pass

    @abstractmethod
    def train(self, training_data, model_path):
        """
        This method trains a model to perform some predictions. It receives some training data and then computes
        a model saved in 'model_path'
        :return: Nothing, the model is saved in 'model_path'
        """
        pass