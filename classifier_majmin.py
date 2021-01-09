import supervised_classifier
import json
import pickle
from support.utils import *

from support.embedding_model import Embedding_Model

class Classifier_MajMin(supervised_classifier.Supervised_Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 topk,
                 results_dir,
                 model_name,
                 classifiers,
                 is_min_voting = False,
                 hyper_params = None,
                 model_path = None):
        self.classifiers = classifiers
        self.topk = topk
        self.dataset_name = dataset.get_name()
        self.is_min_voting = is_min_voting
        self.classifiers_annotations = None
        self.model_name = model_name
        self.type_prediction = type_prediction
        super(Classifier_MajMin, self).__init__(dataset, type_prediction, results_dir, None, hyper_params, model_path)

    def init_model(self, embedding_model, hyper_params):
        pass

    def create_training_data(self, queries_with_annotated_answers):
        pass

    def get_name(self):
        if self.is_min_voting == True:
            return "Min"
        else:
            return "Maj"

    def train(self, training_data, model_path):
        pass

    def predict(self, query_with_answers):
        # Load the answers provided by all classifiers
        if self.classifiers_annotations is None:
            self.classifiers_annotations = {}
            for idx, classifier in enumerate(self.classifiers):
                suf = '-' + classifier
                file_name = get_filename_answer_annotations(self.dataset_name, self.model_name, "test", self.topk, self.type_prediction, suf)
                file_path = self.result_dir + '/' + self.dataset_name + '/annotations/' + file_name
                classifier_annotations = pickle.load(open(file_path, 'rb'))
                for _, a in enumerate(classifier_annotations):
                    ent = a['query']['ent']
                    rel = a['query']['rel']
                    typ = a['query']['type']
                    assert (typ == 0 or self.type_prediction == 'tail')
                    assert (typ == 1 or self.type_prediction == 'head')
                    if (ent, rel) not in self.classifiers_annotations:
                        assert (idx == 0)
                        self.classifiers_annotations[(ent, rel)] = {}
                    else:
                        assert(idx != 0)
                    answer_set = self.classifiers_annotations[(ent, rel)]
                    for answer in a['annotated_answers']:
                        if answer['entity_id'] not in answer_set:
                            assert (idx == 0)
                            answer_set[answer['entity_id']] = [answer['checked']]
                        else:
                            assert (idx != 0)
                            answer_set[answer['entity_id']].append(answer['checked'])
            for key, ans in self.classifiers_annotations.items():
                assert (len(ans) == self.topk)

        #for q in query_with_answers:
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 0 or self.type_prediction == 'tail')
        assert (typ == 1 or self.type_prediction == 'head')
        assert((ent, rel) in self.classifiers_annotations)

        # Check that the output matches the filtered answers
        filtered_answers = set()
        for a in  query_with_answers['answers_fil']:
            filtered_answers.add(a)

        annotated_answers = []
        threshold_for_true = 0 # If min voting is active, one true label is enough
        if not self.is_min_voting:
            threshold_for_true = int(len(self.classifiers) / 2 + 1)
        for entity_id, labels in self.classifiers_annotations[(ent, rel)].items():
            assert(entity_id in filtered_answers)
            assert(len(labels) == len(self.classifiers))
            count = 0
            for l in labels:
                if l == True:
                    count += 1
            checked = False
            if count >= threshold_for_true:
                checked = True
            annotated_answers.append({'entity_id': entity_id, 'checked': checked, 'score': 1})
        return annotated_answers