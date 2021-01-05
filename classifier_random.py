import classifier
import random


class Classifier_Random(classifier.Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 results_dir):
        super(Classifier_Random, self).__init__(dataset, type_prediction, results_dir)

    def get_name(self):
        return "Random"

    def predict(self, query_with_answers):
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 1 or self.type_prediction == 'head')
        assert (typ == 0 or self.type_prediction == 'tail')

        annotated_answers = []
        for answer in query_with_answers['answers_fil']:
            checked = random.randint(0, 2)
            annotated_answers.append({'entity_id' : answer, 'checked' : checked, 'score': checked})
        return annotated_answers