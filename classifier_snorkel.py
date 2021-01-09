import classifier_majmin
from snorkel.labeling.model.label_model import LabelModel
import numpy as np

class Classifier_Snorkel(classifier_majmin.Classifier_MajMin):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 topk,
                 results_dir,
                 classifiers,
                 model_name,
                 model_path = None):
        self.classifiers = classifiers
        self.topk = topk
        self.dataset_name = dataset.get_name()
        self.classifiers_annotations = None
        self.model_name = model_name
        self.type_prediction = type_prediction
        super(Classifier_Snorkel, self).__init__(dataset, type_prediction, results_dir, None, None, model_path)

    def init_model(self, embedding_model, hyper_params):
        # TODO: Load the model
        pass

    def create_training_data(self, queries_with_annotated_answers):
        # TODO: Create the training data to train the snorkel model
        pass

    def get_name(self):
        return "Snorkel"

    def train(self, training_data, model_path):
        label_model = LabelModel(verbose=False)
        label_model.fit(training_data, n_epochs=500, optimizer="adam")
        label_model.save(model_path)

    def predict(self, query_with_answers):
        # TODO: Do the prediction
        pass