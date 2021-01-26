import supervised_classifier
from snorkel.labeling.model.label_model import LabelModel
import numpy as np
from support.utils import *
import json

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
            print("Abstain scores: ", str(self.abstain_scores))
        if model_path is not None:
            print("Loading existing model {} ...".format(model_path))
            label_model = LabelModel(verbose=True)
            label_model.load(model_path)
            self.set_model(label_model)
            with open(model_path + '.meta', 'rt') as fin:
                a = json.load(fin)
                self.classifiers = a['classifiers']
                if self.abstain_scores is not None:
                    new_scores = []
                    for colId in a['retained_columns']:
                        new_scores.append(self.abstain_scores[colId])
                    self.abstain_scores = new_scores

    def init_model(self, embedding_model, hyper_params):
        pass

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

    def create_training_data(self, queries_with_answers, valid_dataset=None):
        classifiers_annotations = load_classifier_annotations(self.classifiers,
                                                                    self.result_dir,
                                                                    self.dataset_name,
                                                                    self.embedding_model_name,
                                                                    "train",
                                                                    self.topk,
                                                                    self.type_prediction,
                                                                    return_scores=True)
        # load the test dataset to retrieve the annotation on the valid dataset
        selected_classifiers = []
        keep_columns = []
        if valid_dataset is not None:
            if self.type_prediction == 'head':
                accepted_type = 0
            else:
                accepted_type = 1
            valid_queries = [(q['query']['ent'], q['query']['rel']) for k, q in valid_dataset.items() if
                             q['query']['type'] == accepted_type]
            valid_queries = set(valid_queries)
            test_classifiers_annotations = load_classifier_annotations(self.classifiers,
                                                                       self.result_dir,
                                                                       self.dataset_name,
                                                                       self.embedding_model_name,
                                                                       "test",
                                                                       self.topk,
                                                                       self.type_prediction,
                                                                       return_scores=True)
            count_true = np.zeros(len(self.classifiers), dtype=int)
            for key, annotations in tqdm(test_classifiers_annotations.items()):
                if key in valid_queries:
                    for answer, annotation in annotations.items():
                        l = self._annotate_labels(annotation)
                        for j, label in enumerate(l):
                            if label == True:
                                count_true[j] += 1
            for i, classifier in enumerate(self.classifiers):
                if count_true[i] != 0 or (3 - len(selected_classifiers)) == (len(self.classifiers) - i):
                    selected_classifiers.append(classifier)
                    keep_columns.append(i)
                else:
                    print("Dropping classifier {}".format(classifier))
        else:
            for i, classifier in enumerate(self.classifiers):
                selected_classifiers.append(classifier)
                keep_columns.append(i)

        print("  Creating training data ...")
        training_data = []
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

        if len(selected_classifiers) < len(self.classifiers):
            new_training_data = np.zeros(shape=(len(training_data), len(selected_classifiers)), dtype=np.int)
            for i, row in enumerate(training_data):
                for j, idx in enumerate(keep_columns):
                    new_training_data[i][j] = training_data[i][idx]
            training_data = new_training_data
        return {'classifiers': selected_classifiers, 'retained_columns': keep_columns, 'data': training_data}

    def get_name(self):
        return "Snorkel"

    def train(self, training_data, valid_data, model_path):
        self.model = LabelModel(verbose=True)
        data = training_data['data']
        if self.classifiers != training_data['classifiers']:
            new_scores = []
            for colId in training_data['retained_columns']:
                new_scores.append(self.abstain_scores[colId])
            self.abstain_scores = new_scores
            self.classifiers = training_data['classifiers']

        self.model.fit(data, n_epochs=500, optimizer="adam")
        if model_path is not None:
            self.model.save(model_path)
            with open(model_path + '.meta', 'wt') as fout:
                json.dump({'classifiers' : self.classifiers, 'retained_columns' : training_data['retained_columns'] }, fout)
                fout.close()

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

        # for q in query_with_answers:
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
            out = self.get_model().predict(l)
            checked = out[0] == TRUE
            annotated_answers.append({'entity_id': entity_id, 'checked': checked, 'score': 1})
        return annotated_answers