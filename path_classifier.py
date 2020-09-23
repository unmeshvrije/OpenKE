import numpy as np
import json
import pickle
from tqdm import tqdm
from answer_classifier import AnswerClassifier
from subgraphs import Subgraph
from subgraphs import SUBTYPE
from numpy import linalg as LA
from subgraphs import read_triples
from openke.utils import DeepDict
#import copy
import torch
import kge.model

class PathClassifier(AnswerClassifier):

    def __init__(self, type_prediction, db, topk_answers_per_query, queries_file_path, embeddings_file_path, emb_model, training_file_path, score_threshold_percentage = 0.1):
        super(PathClassifier, self).__init__(type_prediction, queries_file_path, db, emb_model, topk_answers_per_query)
        self.topk_answers_per_query = topk_answers_per_query
        #self.emb_file_path = embeddings_file_path
        self.training_file_path = training_file_path
        self.score_threshold_percentage = score_threshold_percentage
        #self.init_embeddings(emb_model)
        self.init_graph()
        self.cnt_subgraphs_dict = {}
        # This is the list of Counts of subgraphs / % Threshold
        # Count of subgraphs in which the answer was found.
        # % Threshold for this query (dynamically computed, hence different for every query)
        self.cnt_subgraphs_dict["raw"] = []
        self.cnt_subgraphs_dict["fil"] = []

    def set_logfile(self, logfile):
        self.logfile = logfile

    def print_path(self, path):
        for p in path:
            ent = p[0]
            rel = p[1]
            if rel != -1:
                print("({}){},({}){}) ;".format(ent, self.entity_dict[ent], rel, self.relation_dict[rel]))
            else:
                print("(({}){},-1) ;".format(ent, self.entity_dict[ent]))

        #log.close()

    def print_answer_entities(self):
        if self.logfile == None:
            return
        log = open(self.logfile, "w")
        for index, x in enumerate(self.x_test_fil):
            e = int(x[0])
            r = int(x[1])
            a = int(x[2])
            head = e
            tail = a
            if self.type_prediction == "head":
                head = a
                tail = e
            sub = "" #"{" + self.cnt_subgraphs_dict["fil"][index] + "}"
            if self.y_test_fil[index] == 1 and self.y_predicted_fil[index] == 0:
                print("$$Expected (1) Predicted (0): $", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail], sub," $$$", file=log)
            if self.y_predicted_fil[index] == 1 and self.y_test_fil[index] == 0:
                print("**Expected (0) Predicted (1): * ", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail] , sub," ***", file=log)
            if self.y_predicted_fil[index] == 1 and self.y_test_fil[index] == 1:
                print("##Expected (1) Predicted (1): # ", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail] , sub," ###", file=log)
            if self.y_predicted_fil[index] == 0 and self.y_test_fil[index] == 0:
                print("##Expected (0) Predicted (0): # ", self.entity_dict[head] , " , ", self.relation_dict[r] , " => ", self.entity_dict[tail] , sub, " ###", file=log)
            if (index+1) % self.topk_answers_per_query == 0:
                print("*" * 80, file = log)

        log.close()

    #def read_complex_embeddings(filename):
    #    model = kge.model.KgeModel.load_from_checkpoint(filename)
    #    E = model._entity_embedder._embeddings_all()
    #    R = model._relation_embedder._embeddings_all()
    #    return E.tolist(), R.tolist()

    def init_embeddings(self, emb_model):
        if emb_model == "complex":
            model = kge.model.KgeModel.load_from_checkpoint(self.emb_file_path)
            E = model._entity_embedder._embeddings_all()
            R = model._relation_embedder._embeddings_all()
            self.E = E.tolist()
            self.R = R.tolist()
        else:
            with open (self.emb_file_path, 'r') as fin:
                params = json.loads(fin.read())
            self.E = params['ent_embeddings.weight']
            self.R = params['rel_embeddings.weight']

        self.N = len(self.E)
        self.M = len(self.R)

    def init_graph(self):
        self.kg = DeepDict()
        self.kg_rel = DeepDict()
        triples = read_triples(self.training_file_path)
        for triple in triples:
            h = triple[0]
            t = triple[1]
            r = triple[2]
            if h not in self.kg:
                self.kg[h] = DeepDict()
            if t not in self.kg:
                self.kg[t] = DeepDict()
            if r not in self.kg[h]:
                self.kg[h][r] = []
            if r not in self.kg[t]:
                self.kg[t][r] = []

            if r not in self.kg_rel:
                self.kg_rel[r] = DeepDict()
            if h not in self.kg_rel[r]:
                self.kg_rel[r][h] = []
            if t not in self.kg_rel[r]:
                self.kg_rel[r][t] = []

            self.kg[h][r].append(t)
            self.kg[t][r].append(h)

            self.kg_rel[r][h].append(t)
            self.kg_rel[r][t].append(h)

    def transe_score(self, sub_emb, ent_emb, rel_emb, pred_type):
        if pred_type == "tail":
            score = (ent_emb + rel_emb) - sub_emb
        else:
            score = sub_emb + (rel_emb - ent_emb)

        return LA.norm(score, 2)

    def get_all_paths_util(self, u, r, d, visited, path, paths):
        # Mark the current node as visited and store in path
        visited[u]= True
        #print("(u,r) => {}, {}".format(u, r))
        path.append((u,r))
        # If current vertex is same as destination, then add the path to paths
        if u == d:
            #print("######## FOUND")
            #self.print_path(path)
            #print("########")
            paths.append(path)
        else:
            if len(path) > 10: # this prevents infinite recursion
                return
            if u in self.kg:
                for r in self.kg[u]:
                    for i in self.kg[u][r]:
                        if visited[i] == False:
                            self.get_all_paths_util(i, r, d, visited, path, paths)
                            path.pop()
                            visited[u]= False

    def get_all_paths(self, s, d):
        # Mark all the vertices as not visited
        visited =[False]*(self.N)
        # Create an array to store paths
        paths = []
        path = []
        # Call the recursive helper function to print all paths
        self.get_all_paths_util(s, -1, d, visited, path, paths)
        return paths

    def get_random_walks(self, src, rel, dst):
        if src not in self.kg or dst not in self.kg:
            return []

        paths = []
        src_arcs = [r for r in self.kg[src] if r != rel]
        dst_arcs = [r for r in self.kg[dst] if r != rel]

        for r1 in src_arcs:
            for r2 in dst_arcs:
                if r1 == r2:
                    continue
                common = [x for x in self.kg_rel[r1][src] if x in self.kg_rel[r2][dst]]
                if len(common) != 0:
                    for i,c in enumerate(common):
                        path = [(src,r1), (c,r2),(dst,-1)]
                        paths.append(path)
                        #self.print_path(path)

        return paths

    def predict(self):
        self.predict_internal(self.x_test_raw, self.y_predicted_raw, "raw")
        self.predict_internal(self.x_test_fil, self.y_predicted_fil, "fil")
        # replace all 0s with -1
        for x in self.y_predicted_fil:
            if x == 1:
                self.y_predicted_fil_abs.append(x)
            elif x == 0:
                self.y_predicted_fil_abs.append(-1)
        #self.predict_internal(self.x_test_fil, self.y_predicted_fil_abs, "abs")

    def predict_internal(self, x_test, y_predicted, setting):
        # Go over all test queries
        cnt_subgraphs_index = 0
        for index in tqdm(range(0, len(x_test), self.topk_answers_per_query)):
            #print(index , " : ")
            features = np.array(x_test[index: index + self.topk_answers_per_query])
            ent = int(features[0][0])
            rel = int(features[0][1])
            topk_ans_entities = features[:, 2].astype(int)

            '''
                Find all true answers for (head, rel, ) from training set (from self.kg)

                find paths between <head, ... b_i)>
            '''
            query_results = dict.fromkeys(topk_ans_entities)
            for k in query_results.keys():
                query_results[k] = False
            #print("(ent,rel) : {}, {}".format(ent, rel))
            # TODO: Could be a key exception if no such answers exist
            if ent not in self.kg or rel not in self.kg[ent]:
                known_answers = []
            else:
                # consider only topk known answers
                known_answers = self.kg[ent][rel][:self.topk_answers_per_query]

            for ka in known_answers:
                #paths = self.generate_all_paths(ent, rel, ka, topk_ans_entities) # avoid rel and all top_ans_entities
                #print("(ent,rel, known_ans) => ({}, {}, {})".format(ent, rel, ka))
                #paths = self.get_all_paths(ent, ka) # avoid rel and all top_ans_entities
                paths = self.get_random_walks(ent, rel, ka)
                # path is [(s, src_arcs), (x1,r2), (x2,r3), ..., , (d, -1)] where s and d are source and destinations
                for path in paths:
                    #print(path)
                    if len(path) > 2:
                        for ae in topk_ans_entities:
                            if query_results[ae] == False:
                                if ae in self.kg[path[-2][0]][path[-2][1]]:
                                    # TODO: Check threshold of path[-1][0] and ae
                                    if setting == "fil":
                                        print("({}, {}, {})".format(self.entity_dict[ent], self.relation_dict[rel],self.entity_dict[ae]))
                                        print("MATCHED due to path : ")
                                        self.print_path(path)
                                        print("#" * 80)
                                    query_results[ae] = True

            for i, answer in enumerate(topk_ans_entities):
                if query_results[answer]: #topk_subgraphs/2:
                    y_predicted.append(1)
                else:
                    if setting == "abs":
                        # abstain
                        y_predicted.append(-1)
                    else:
                        y_predicted.append(0)