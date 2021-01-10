import supervised_classifier
from snorkel.labeling.model.label_model import LabelModel
from tqdm import tqdm
import numpy as np
from support.utils import *

FALSE=0
TRUE=1
ABSTAIN=-1

class Classifier_Snorkel(supervised_classifier.Supervised_Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 topk,
                 results_dir,
                 classifiers,
                 embedding_model_name,
                 model_path = None):
        self.classifiers = classifiers
        self.topk = topk
        self.dataset_name = dataset.get_name()
        self.classifiers_annotations = None
        self.embedding_model_name = embedding_model_name
        self.type_prediction = type_prediction
        self.snorkel_model = None
        self.result_dir = results_dir
        self.test_annotations = None
        if model_path is not None:
            print("Loading existing model {} ...".format(model_path))
            label_model = LabelModel(verbose=True)
            label_model.load(model_path)
            self.set_model(label_model)

    def init_model(self, embedding_model, hyper_params):
        pass

    def _annotate_labels(self, labels):
        l = []
        for a in labels:
            if a == False:
                l.append(FALSE)
            else:
                l.append(TRUE)
        return l

    def create_training_data(self, queries_with_answers):
        classifiers_annotations = load_classifier_annotations(self.classifiers,
                                                                   self.result_dir,
                                                                   self.dataset_name,
                                                                   self.embedding_model_name,
                                                                   "train",
                                                                   self.topk,
                                                                   self.type_prediction)
        training_data = []
        print("  Creating training data ...")
        for keys, annotations in tqdm(classifiers_annotations.items()):
            for answer, annotation in annotations.items():
                l = self._annotate_labels(annotation)
                training_data.append(l)
        training_data = np.asarray(training_data)
        return training_data

    def get_name(self):
        return "Snorkel"

    def train(self, training_data, valid_data, model_path):
        label_model = LabelModel(verbose=True)
        label_model.fit(training_data, n_epochs=500, optimizer="adam")
        label_model.save(model_path)

    def predict(self, query_with_answers):
        if self.test_annotations is None:
            self.test_annotations = load_classifier_annotations(self.classifiers,
                                                                       self.result_dir,
                                                                       self.dataset_name,
                                                                       self.embedding_model_name,
                                                                       "test",
                                                                       self.topk,
                                                                       self.type_prediction)

        # for q in query_with_answers:
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 0 or self.type_prediction == 'tail')
        assert (typ == 1 or self.type_prediction == 'head')
        assert ((ent, rel) in self.test_annotations)

        # Check that the output matches the filtered answers
        filtered_answers = set()
        for a in query_with_answers['answers_fil']:
            filtered_answers.add(a)

        annotated_answers = []
        for entity_id, labels in self.test_annotations[(ent, rel)].items():
            assert (entity_id in filtered_answers)
            assert (len(labels) == len(self.classifiers))
            l = np.asarray(self._annotate_labels(labels))
            l = l.reshape(1, -1)
            out = self.get_model().predict(l)
            checked = out[0] == TRUE
            annotated_answers.append({'entity_id': entity_id, 'checked': checked, 'score': 1})
        return annotated_answers