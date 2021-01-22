import classifier_snorkel
from flyingsquid.label_model import LabelModel
import numpy as np
from support.utils import *
import json

FALSE=-1
TRUE=1
ABSTAIN=0

class Classifier_Squid(classifier_snorkel.Classifier_Snorkel):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 topk,
                 results_dir,
                 classifiers,
                 embedding_model_name,
                 model_path = None,
                 abstain_scores = None):
        self.classifiers = classifiers
        self.topk = topk
        self.dataset_name = dataset.get_name()
        self.embedding_model_name = embedding_model_name
        self.type_prediction = type_prediction
        self.result_dir = results_dir
        self.test_annotations = None
        self.abstain_scores = abstain_scores
        if self.abstain_scores is not None:
            assert(len(self.abstain_scores) == len(classifiers))
        super(Classifier_Squid, self).__init__(dataset, type_prediction, topk, results_dir, classifiers, embedding_model_name, model_path=None, abstain_scores=abstain_scores)
        if model_path is not None:
            print("Loading existing model {} ...".format(model_path))
            with open(model_path + '.meta', 'rt') as fin:
                a = json.load(fin)
                self.classifiers = a['classifiers']
            with open(model_path, 'rb') as fin:
                cls, attrs = pickle.load(fin)
                label_model = LabelModel.load(cls, attrs)
                self.set_model(label_model)

    def get_name(self):
        return "SQUID"

    def train(self, training_data, valid_data, model_path):
        data = training_data['data']
        self.classifiers = training_data['classifiers']
        self.model = LabelModel(len(self.classifiers))
        self.model.fit(data)
        if model_path is not None:
            pickledVersion = self.model.save()
            with open(model_path, 'wb') as fout:
                pickle.dump(pickledVersion, fout)
            with open(model_path + '.meta', 'wt') as fout:
                json.dump({'classifiers' : self.classifiers, 'retained_columns' : training_data['retained_columns'] }, fout)
                fout.close()

    def _annotate_labels(self, labels):
        l = []
        if self.abstain_scores is None:
            for a in labels:
                if a == False:
                    l.append(FALSE)
                else:
                    l.append(TRUE)
        else:
            for i, a in enumerate(labels):
                low_score, hi_score = self.abstain_scores[i]
                if a < low_score:
                    l.append(FALSE)
                elif a >= hi_score:
                    l.append(TRUE)
                else:
                    l.append(ABSTAIN)
        return l

    def create_training_data(self, queries_with_answers):
        classifiers_annotations = load_classifier_annotations(self.classifiers,
                                                                    self.result_dir,
                                                                    self.dataset_name,
                                                                    self.embedding_model_name,
                                                                    "train",
                                                                    self.topk,
                                                                    self.type_prediction,
                                                                    return_scores=True)
        training_data = []
        print("  Creating training data ...")
        for keys, annotations in tqdm(classifiers_annotations.items()):
            for answer, annotation in annotations.items():
                l = self._annotate_labels(annotation)
                training_data.append(l)
        training_data = np.asarray(training_data)
        count_true = np.zeros(len(self.classifiers), dtype=int)
        count_false = np.zeros(len(self.classifiers), dtype=int)
        count_abstain = np.zeros(len(self.classifiers), dtype=int)
        for t in training_data:
            for i, a in enumerate(t):
                if a == TRUE:
                    count_true[i] += 1
                elif a == FALSE:
                    count_false[i] += 1
                else:
                    count_abstain[i] += 1
        selected_classifiers = []
        keep_columns = []
        for i, classifier in enumerate(self.classifiers):
            if count_true[i] != 0 or (3 - len(selected_classifiers)) == (len(self.classifiers) - i):
                selected_classifiers.append(classifier)
                keep_columns.append(i)
            else:
                print("Dropping classifier {}".format(classifier))
            print("Classifier {} TRUE {} FALSE {} ABSTAIN {}".format(classifier, count_true[i], count_false[i], count_abstain[i]))
        if len(selected_classifiers) < len(self.classifiers):
            new_training_data = np.zeros(shape=(len(training_data), len(selected_classifiers)), dtype=np.int)
            for i, row in enumerate(training_data):
                for j, idx in enumerate(keep_columns):
                    new_training_data[i][j] = training_data[i][idx]
            training_data = new_training_data
        return {'classifiers' : selected_classifiers, 'retained_columns' : keep_columns, 'data' : training_data}


    def predict(self, query_with_answers, type_answers, provenance_test = "test"):
        if self.test_annotations is None:
            self.test_annotations = load_classifier_annotations(self.classifiers,
                                                                        self.result_dir,
                                                                        self.dataset_name,
                                                                        self.embedding_model_name,
                                                                        provenance_test,
                                                                        self.topk,
                                                                        self.type_prediction,
                                                                        return_scores=True)

        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 0 or self.type_prediction == 'tail')
        assert (typ == 1 or self.type_prediction == 'head')
        assert ((ent, rel) in self.test_annotations)

        # Check that the output matches the filtered answers
        filtered_answers = set()
        for a in query_with_answers[type_answers]:
            filtered_answers.add(a['entity_id'])

        annotated_answers = []
        for entity_id, labels in self.test_annotations[(ent, rel)].items():
            assert (entity_id in filtered_answers)
            assert (len(labels) == len(self.classifiers))
            l = np.asarray(self._annotate_labels(labels))
            l = l.reshape(1, -1)
            out = self.get_model().predict(l).reshape(1)
            checked = out[0] == TRUE
            annotated_answers.append({'entity_id': entity_id, 'checked': checked, 'score': 1})
        return annotated_answers