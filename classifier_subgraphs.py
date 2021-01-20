import classifier
from support.embedding_model import *

class Classifier_Subgraphs(classifier.Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 embedding_model : Embedding_Model,
                 results_dir,
                 k):
        self.embedding_model = embedding_model
        self.embedding_model.make_subgraph_embeddings(dataset)
        self.k = k
        super(Classifier_Subgraphs, self).__init__(dataset, type_prediction, results_dir)

    def get_name(self):
        return "Sub"

    def predict(self, query_with_answers, type_answers):
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 1 or self.type_prediction == 'head')
        assert (typ == 0 or self.type_prediction == 'tail')

        if typ == 0:
            similar_subgraphs = self.embedding_model.get_most_similar_subgraphs(SubgraphType.POS, ent, rel, k=self.k)
        else:
            similar_subgraphs = self.embedding_model.get_most_similar_subgraphs(SubgraphType.SPO, ent, rel, k=self.k)

        # Collect all acceptable entities
        acceptable_answers = set()
        for subgraph in similar_subgraphs:
            if subgraph.type == SubgraphType.SPO:
                subgraph_members = self.dataset.get_known_answers_for_hr(subgraph.ent, subgraph.rel)
            else:
                subgraph_members = self.dataset.get_known_answers_for_tr(subgraph.ent, subgraph.rel)
            for s in subgraph_members:
                acceptable_answers.add(s)

        annotated_answers = []
        for a in query_with_answers[type_answers]:
            answer = a['entity_id']
            checked = answer in acceptable_answers
            score = 0
            if checked:
                score = 1
            annotated_answers.append({'entity_id': answer, 'checked': checked, 'score': score})
        return annotated_answers