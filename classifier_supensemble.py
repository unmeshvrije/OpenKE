import supervised_classifier
from support.utils import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

FALSE=0
TRUE=1

class Classifier_SuperEnsemble(supervised_classifier.Supervised_Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 topk,
                 results_dir,
                 classifiers,
                 embedding_model_name,
                 model_path=None):
        self.classifiers = classifiers
        self.topk = topk
        self.dataset_name = dataset.get_name()
        self.embedding_model_name = embedding_model_name
        self.type_prediction = type_prediction
        self.result_dir = results_dir
        self.test_annotations = None
        if model_path is not None:
            print("Loading existing model {} ...".format(model_path))
            model = joblib.load(model_path)
            self.set_model(model)

    def get_name(self):
        return "SupEnsemble"

    def init_model(self, embedding_model, hyper_params):
        self.model = RandomForestClassifier(max_depth=2, random_state=0)

    def create_training_data(self, queries_with_answers):
        classifiers_annotations = load_classifier_annotations(self.classifiers,
                                                                    self.result_dir,
                                                                    self.dataset_name,
                                                                    self.embedding_model_name,
                                                                    "train",
                                                                    self.topk,
                                                                    self.type_prediction,
                                                                    return_scores=False)
        training_data = []
        print("  Creating training data ...")
        for keys, annotations in tqdm(classifiers_annotations.items()):
            for answer, annotation in annotations.items():
                training_data.append(annotation)
        training_data = np.asarray(training_data)
        count_true = np.zeros(len(self.classifiers), dtype=int)
        count_false = np.zeros(len(self.classifiers), dtype=int)
        for t in training_data:
            for i, a in enumerate(t):
                if a == TRUE:
                    count_true[i] += 1
                else:
                    count_false[i] += 1
        for i, classifier in enumerate(self.classifiers):
            print("Classifier {} TRUE {} FALSE {}".format(classifier, count_true[i], count_false[i]))
        return training_data

    def train(self, training_data, valid_data, model_path):
        X, y = make_classification(n_samples=1000,
                                   n_features=4,
                                   n_informative = 2,
                                   n_redundant = 0,
                                   random_state = 0,
                                   shuffle = False)
        self.model.fit(X, y)
        if model_path is not None:
            self.model.save(model_path)
            joblib.dump(self.model, model_path)

    def predict(self, query_with_answers, provenance_test="test"):
        if self.test_annotations is None:
            self.test_annotations = load_classifier_annotations(self.classifiers,
                                                                self.result_dir,
                                                                self.dataset_name,
                                                                self.embedding_model_name,
                                                                provenance_test,
                                                                self.topk,
                                                                self.type_prediction,
                                                                return_scores=True)
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
            filtered_answers.add(a['entity_id'])

        annotated_answers = []
        for entity_id, labels in self.test_annotations[(ent, rel)].items():
            assert (entity_id in filtered_answers)
            assert (len(labels) == len(self.classifiers))
            l = labels.reshape(1, -1)
            out = self.get_model().predict(l)
            checked = out[0] == TRUE
            annotated_answers.append({'entity_id': entity_id, 'checked': checked, 'score': 1})
        return annotated_answers