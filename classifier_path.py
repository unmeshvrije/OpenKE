import classifier
import random

class Classifier_Path(classifier.Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 results_dir):
        super(Classifier_Path, self).__init__(dataset, type_prediction, results_dir)

    def get_name(self):
        return "Path"

    def predict(self, query_with_answers):
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 1 or self.type_prediction == 'head')
        assert (typ == 0 or self.type_prediction == 'tail')

        annotated_answers = []
        neighbour_entity = self.dataset.get_neighbours(ent)
        if typ == 1:
            known_answers = self.dataset.get_known_answers_for_hr(ent, rel)
        else:
            known_answers = self.dataset.get_known_answers_for_tr(ent, rel)
        known_answers = set(known_answers)
        for answer in query_with_answers['answers_fil']:
            neighbour_answer = self.dataset.get_neighbours(answer)
            common_neighbours = neighbour_entity.intersection(neighbour_answer)
            found = False
            for common_neighbour in common_neighbours:
                neighbour2 = self.dataset.get_neighbours(common_neighbour)
                intersection = known_answers.intersection(neighbour2)
                if len(intersection) > 0:
                    found = True
                    break
            annotated_answers.append({'entity_id': answer, 'checked': found, 'score': 1})
        return annotated_answers