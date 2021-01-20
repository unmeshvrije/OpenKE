import classifier
import random


class Classifier_Threshold(classifier.Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 results_dir,
                 threshold):
        super(Classifier_Threshold, self).__init__(dataset, type_prediction, results_dir)
        self.threshold = threshold

    def get_name(self):
        return "Threshold"

    def predict(self, query_with_answers, type_answers):
        typ = query_with_answers['type']
        assert (typ == 1 or self.type_prediction == 'head')
        assert (typ == 0 or self.type_prediction == 'tail')

        annotated_answers = []
        for i, answer in enumerate(query_with_answers[type_answers]):
            checked = 1
            if i >= self.threshold:
                checked = 0
            annotated_answers.append({'entity_id' : answer['entity_id'], 'checked' : checked, 'score': checked})
        return annotated_answers