from classifier import Classifier
from abc import abstractmethod
from support.embedding_model import Embedding_Model
import torch
import numpy as np
from tqdm import tqdm

class Supervised_Classifier(Classifier):

    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 results_dir,
                 embedding_model : Embedding_Model,
                 hyper_params,
                 path_model = None):
        super(Supervised_Classifier, self).__init__(dataset, type_prediction, results_dir)
        self.embedding_model = embedding_model
        self.hyper_params = hyper_params
        print("Hyper-parameters:", hyper_params)
        if path_model is None:
            print("Creating a new model ...")
            self.init_model(embedding_model, hyper_params)
        else:
            print("Loading existing model {} ...".format(path_model))
            self.set_model(torch.load(path_model))

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def start_predict(self):
        self.get_model().eval()

    def save_model(self, model_path, epoch):
        print("Saving model after epoch {}".format(epoch))
        torch.save(self.get_model(), model_path)

    def validate(self, val_data):
        self.get_model().eval()
        # switch to evaluate mode
        true_positives = true_negatives = false_positives = false_negatives = 0
        with torch.no_grad():
            for data_point in tqdm(val_data):
                XS = torch.Tensor(data_point['X'])
                XS = XS.view(1, XS.shape[0], XS.shape[1])
                annotated_answers = self.get_model()(XS)
                annotated_answers = (annotated_answers.numpy() > 0.5).astype(int)
                true_answers = data_point['Y']
                true_positives += np.sum(np.logical_and(annotated_answers,true_answers))
                true_negatives += np.sum(np.logical_and(np.logical_not(annotated_answers), np.logical_not(true_answers)))
                false_positives += np.sum(np.logical_and(annotated_answers, np.logical_not(true_answers)))
                false_negatives += np.sum(np.logical_and(np.logical_not(annotated_answers), true_answers))

            # Measure the F1
        rec = true_positives / (true_positives + false_negatives)
        prec = true_positives / (true_positives + false_positives)
        f1 = 2 * (prec * rec) / (prec + rec)
        print("F1 on the validation dataset was {}".format(f1))
        self.get_model().train()
        return f1

    @abstractmethod
    def init_model(self, embedding_model, hyper_params):
        pass

    @abstractmethod
    def create_training_data(self, queries_with_answers):
        """
        :param answers: These are a set of queries with related answers. They are produced by create_answers.py
        :return: For each query and answers, this methods creates a set of features and label that will be used
        to train the model
        """
        pass

    @abstractmethod
    def train(self, training_data, valid_data, model_path):
        """
        This method trains a model to perform some predictions. It receives some training and valid data and then computes
        a model saved in 'model_path'
        :return: Nothing, the model is saved in 'model_path'
        """
        pass