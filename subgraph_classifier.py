import numpy as np
import json
import pickle
from tqdm import tqdm
from answer_classifier import AnswerClassifier
from subgraphs import Subgraph
from subgraphs import SUBTYPE
from numpy import linalg as LA

class SubgraphClassifier(AnswerClassifier):

    def __init__(self, type_prediction, topk, triples_file_path, embeddings_file_path, subgraphs_file_path, sub_emb_file_path):
        super(SubgraphClassifier, self).__init__(type_prediction, triples_file_path)
        self.topk = topk
        self.emb_file_path = embeddings_file_path
        self.sub_file_path = subgraphs_file_path
        self.sub_emb_file_path = sub_emb_file_path
        self.init_embeddings()
        self.init_subgraphs()
        self.init_sub_embeddings()

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

        return LA.norm(score, 2) #torch.norm(score, 1, -1).flatten()

    def get_subgraph_scores(self, sub_emb, ent_emb, rel_emb, pred_type, score_callback):
        return score_callback(np.array(sub_emb), np.array(ent_emb), np.array(rel_emb), pred_type)

    def predict(self):
        # Go over all test triples
        print(type(self.x_test))
        for index in range(len(self.x_test)):
            print(index , " : ")
            features = np.array(self.x_test[index: index + self.topk])
            ent = int(features[0][0])
            rel = int(features[0][1])
            topk_ans_entities = features[:, 2].astype(int)

            subgraph_scores = []
            for se in self.S:
                subgraph_scores.append(self.get_subgraph_scores(se, self.E[ent], self.R[rel],  self.type_prediction, self.transe_score))
            sub_indexes = np.argsort(subgraph_scores)

            # Computer dynamic topk for query (ent, rel, ?)
            # and only check the answer in those subgraphs (sub_indexes)
            topk_subgraphs = int(0.1 * len(sub_indexes))
            for i, answer in enumerate(topk_ans_entities):
                cnt_presence_in_sub = 0;
                # Check in only topK subgraphs
                for j, sub_index in enumerate(sub_indexes[:topk_subgraphs]):
                    if answer in self.subgraphs[sub_index].data['entities']:
                        cnt_presence_in_sub += 1
                        #print("{} FOUND in subgraph # {}". format(answer, j))
                if cnt_presence_in_sub != 0:
                    self.y_predicted.append(1)
                else:
                    self.y_predicted.append(0)

            index += self.topk

