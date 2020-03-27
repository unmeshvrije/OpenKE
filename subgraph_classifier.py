import numpy as np
import json
import pickle
from tqdm import tqdm
from answer_classifier import AnswerClassifier
from subgraphs import Subgraph
from subgraphs import SUBTYPE
from numpy import linalg as LA
from subgraphs import read_triples

class SubgraphClassifier(AnswerClassifier):

    def __init__(self, type_prediction, topk_answers_per_query, queries_file_path, embeddings_file_path, subgraphs_file_path, sub_emb_file_path, model_str, training_file_path, subgraph_threshold_percentage = 0.2):
        super(SubgraphClassifier, self).__init__(type_prediction, queries_file_path)
        self.topk_answers_per_query = topk_answers_per_query
        self.emb_file_path = embeddings_file_path
        self.sub_file_path = subgraphs_file_path
        self.sub_emb_file_path = sub_emb_file_path
        self.training_file_path = training_file_path
        self.subgraph_threshold_percentage = subgraph_threshold_percentage
        self.init_embeddings()
        self.init_subgraphs()
        self.init_sub_embeddings()
        self.init_training_triples()
        self.init_model_score_function(model_str)

    def init_training_triples(self):
        triples = read_triples(self.training_file_path)
        # triples are in the form (h,r,t)
        # For type_prediction : head, we sort by tail
        if self.type_prediction == "head":
            self.training_triples = sorted(triples, key = lambda l : (l[2], l[1]))
        elif self.type_prediction == "tail":
            self.training_triples = sorted(triples, key = lambda l : (l[2], l[0]))

    def init_model_score_function(self, model_str):
        if model_str == "transe":
            self.model_score = self.transe_score

    def init_embeddings(self):
        with open (self.emb_file_path, 'r') as fin:
            params = json.loads(fin.read())
        self.E = params['ent_embeddings.weight']
        self.R = params['rel_embeddings.weight']

    def init_subgraphs(self):
        with open(self.sub_file_path, 'rb') as fin:
            self.subgraphs = pickle.load(fin)

    def init_sub_embeddings(self):
        with open(self.sub_emb_file_path, 'rb') as fin:
            self.S = pickle.load(fin)

    def transe_score(self, sub_emb, ent_emb, rel_emb, pred_type):
        if pred_type == "tail":
            score = (ent_emb + rel_emb) - sub_emb
        else:
            score = sub_emb + (rel_emb - ent_emb)

        return LA.norm(score, 2)

    def get_subgraph_scores(self, sub_emb, ent_emb, rel_emb, pred_type, score_callback):
        return score_callback(np.array(sub_emb), np.array(ent_emb), np.array(rel_emb), pred_type)

    def predict(self):
        self.predict_internal(self.x_test_raw, self.y_predicted_raw)
        self.predict_internal(self.x_test_fil, self.y_predicted_fil)

    def get_dynamic_topk(self, ent, rel, sub_indexes):
        '''
            1. Search ent, rel in training triples
            2. If answer is found, look for the answer in sorted subgraphs
        '''
        print("ent {}, rel {} ". format(ent, rel))
        answers = []
        for index, triple in enumerate(self.training_triples):
            if triple[2] != rel:
                continue

            if triple[2] > rel:
                break

            if self.type_prediction == "head":
                if triple[1] == ent:
                    answers.append(triple[0])
            elif self.type_prediction == "tail":
                if triple[0] == ent:
                    answers.append(triple[1])

        if len(answers) == 0:
            return int(0.1 * len(sub_indexes))

        found_index = []
        for answer in answers:
            for j, sub_index in enumerate(sub_indexes):
                if answer in self.subgraphs[sub_index].data['entities']:
                    found_index.append(j)
                    break
        if len(found_index) == 0:
            return int(0.1 * len(sub_indexes))

        print("found topk : ", found_index)
        return max(found_index)


    def predict_internal(self, x_test, y_predicted):
        # Go over all test queries
        for index in tqdm(range(0, len(x_test), self.topk_answers_per_query)):
            #print(index , " : ")
            features = np.array(x_test[index: index + self.topk_answers_per_query])
            ent = int(features[0][0])
            rel = int(features[0][1])
            topk_ans_entities = features[:, 2].astype(int)

            subgraph_scores = []
            for index, se in enumerate(self.S):
                if self.subgraphs[index].data['ent'] == ent and self.subgraphs[index].data['rel'] == rel:
                    subgraph_scores.append(np.inf)
                else:
                    subgraph_scores.append(self.get_subgraph_scores(se, self.E[ent], self.R[rel],  self.type_prediction, self.model_score))
            sub_indexes = np.argsort(subgraph_scores)

            # TODO: Computer dynamic topk for query (ent, rel, ?)
            # and only check the answer in those subgraphs (sub_indexes)
            topk_subgraphs = self.get_dynamic_topk(ent, rel, sub_indexes)

            # Check topk_subgraphs and if it is > 10
            threshold_subgraphs = int(self.subgraph_threshold_percentage * topk_subgraphs)

            for i, answer in enumerate(topk_ans_entities):
                cnt_presence_in_sub = 0;
                # Check in only topK subgraphs
                for j, sub_index in enumerate(sub_indexes[:topk_subgraphs]):
                    if answer in self.subgraphs[sub_index].data['entities']:
                        cnt_presence_in_sub += 1
                        #print("{} FOUND in subgraph # {}". format(answer, j))
                #if cnt_presence_in_sub != 0:
                if cnt_presence_in_sub > threshold_subgraphs: #topk_subgraphs/2:
                    y_predicted.append(1)
                else:
                    y_predicted.append(0)

