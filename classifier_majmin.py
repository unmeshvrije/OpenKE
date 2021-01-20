import supervised_classifier
import pickle
from support.utils import *

class Classifier_MajMin(supervised_classifier.Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 topk,
                 results_dir,
                 embedding_model_name,
                 classifiers,
                 is_min_voting = False):
        self.classifiers = classifiers
        self.topk = topk
        self.dataset_name = dataset.get_name()
        self.is_min_voting = is_min_voting
        self.embedding_model_name = embedding_model_name
        self.type_prediction = type_prediction
        self.classifiers_annotations = None
        super(Classifier_MajMin, self).__init__(dataset, type_prediction, results_dir)

    def get_name(self):
        if self.is_min_voting == True:
            return "Min"
        else:
            return "Maj"

    def predict(self, query_with_answers, type_answers):
        # Load the answers provided by all classifiers
        if self.classifiers_annotations is None:
            self.classifiers_annotations = load_classifier_annotations(self.classifiers,
                                                                       self.result_dir,
                                                                       self.dataset_name,
                                                                       self.embedding_model_name,
                                                                       "test",
                                                                       self.topk,
                                                                       self.type_prediction)

        #for q in query_with_answers:
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 0 or self.type_prediction == 'tail')
        assert (typ == 1 or self.type_prediction == 'head')
        assert((ent, rel) in self.classifiers_annotations)

        # Check that the output matches the filtered answers
        filtered_answers = set()
        for a in  query_with_answers[type_answers]:
            filtered_answers.add(a['entity_id'])

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