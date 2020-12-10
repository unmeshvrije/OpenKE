import numpy as np
import json
import pickle
from tqdm import tqdm
from files_unmesh.answer_classifier import AnswerClassifier
from openke.module.model import TransE, RotatE, ComplEx
from numpy import linalg as LA
from files_unmesh.subgraphs import read_triples
from openke.data import TrainDataLoader
import torch
import kge.model

class SubgraphClassifier(AnswerClassifier):

    def __init__(self, type_prediction, db, topk_answers_per_query, queries_file_path, embeddings_file_path, subgraphs_file_path, sub_emb_file_path, emb_model, training_file_path, db_path, subgraph_threshold_percentage = 0.1):
        super(SubgraphClassifier, self).__init__(type_prediction, queries_file_path, db, emb_model, topk_answers_per_query)
        self.topk_answers_per_query = topk_answers_per_query
        self.emb_file_path = embeddings_file_path
        self.sub_file_path = subgraphs_file_path
        self.sub_emb_file_path = sub_emb_file_path
        self.training_file_path = training_file_path
        self.subgraph_threshold_percentage = subgraph_threshold_percentage
        self.init_embeddings(emb_model)
        self.init_subgraphs()
        self.init_sub_embeddings()
        self.init_training_triples()

        self.init_train_dataloader(db_path)
        self.model_name = emb_model
        self.init_model_score_function(emb_model)
        self.cnt_subgraphs_dict = {}
        # This is the list of Counts of subgraphs / % Threshold
        # Count of subgraphs in which the answer was found.
        # % Threshold for this query (dynamically computed, hence different for every query)
        self.cnt_subgraphs_dict["raw"] = []
        self.cnt_subgraphs_dict["fil"] = []
        self.cnt_subgraphs_dict["abs"] = []

    def set_logfile(self, logfile):
        self.logfile = logfile

    def init_train_dataloader(self, db_path):
        self.train_dataloader = TrainDataLoader(
            in_path = db_path,
            nbatches = 100,
            threads = 8,
            sampling_mode = "normal",
            bern_flag = 1,
            filter_flag = 1,
            neg_ent = 25,
            neg_rel = 0
            )
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
            sub = "{" + self.cnt_subgraphs_dict["fil"][index] + "}"
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

    def init_training_triples(self):
        triples = read_triples(self.training_file_path)
        # triples are in the form (h,r,t)
        # For type_prediction : head, we sort by tail
        if self.type_prediction == "head":
            self.training_triples = sorted(triples, key = lambda l : (l[2], l[1]))
        elif self.type_prediction == "tail":
            self.training_triples = sorted(triples, key = lambda l : (l[2], l[0]))

    def init_model_score_function(self, emb_model):
        if emb_model == "transe":
            N_DIM = 200
            #self.model_score = self.transe_score
            self.model = TransE(
                    ent_tot = self.train_dataloader.get_ent_tot(),
                    rel_tot = self.train_dataloader.get_rel_tot(),
                    dim = N_DIM,
                    p_norm = 1,
                    norm_flag = True
                    )
        elif emb_model == "rotate":
            N_DIM = 200
            self.model = RotatE(
                            ent_tot  = self.train_dataloader.get_ent_tot(),
                            rel_tot = self.train_dataloader.get_rel_tot(),
                            dim = N_DIM,
                            margin = 6.0,
                            epsilon = 2.0)
        elif emb_model == "complex":
            N_DIM = 256
            self.model = ComplEx(
                    ent_tot = self.train_dataloader.get_ent_tot(),
                    rel_tot = self.train_dataloader.get_rel_tot(),
                    dim = N_DIM
                    );

    def init_embeddings(self, emb_model):
        if emb_model == "complex":
            model = kge.model.KgeModel.load_from_checkpoint(self.emb_file_path)
            E_temp = model._entity_embedder._embeddings_all()
            R_temp = model._relation_embedder._embeddings_all()
            self.E = E_temp.tolist()
            self.R = R_temp.tolist()
        else:
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

    def complex_score(self, sub_emb, ent_emb, rel_emb, pred_type):
        # separate real and imag embeddings
        mid = len(sub_emb)/2
        sub_re = sub_emb[:mid]
        sub_im = sub_emb[mid:]
        ent_re = ent_emb[:mid]
        ent_im = ent_emb[mid:]
        rel_re = rel_emb[:mid]
        rel_im = rel_emb[mid:]

        if pred_type == "tail":
            score = (ent_emb + rel_emb) - sub_emb
        else:
            score = sub_emb + (rel_emb - ent_emb)

        return LA.norm(score, 2)

    def rotate_score(self, sub_emb, ent_emb, rel_emb, pred_type):
        if pred_type == "tail":
            score = (ent_emb + rel_emb) - sub_emb
        else:
            score = sub_emb + (rel_emb - ent_emb)

        return LA.norm(score, 2)

    def transe_score(self, sub_emb, ent_emb, rel_emb, pred_type):
        if pred_type == "tail":
            score = (ent_emb + rel_emb) - sub_emb
        else:
            score = sub_emb + (rel_emb - ent_emb)

        return LA.norm(score, 2)

    #def get_subgraph_scores(self, sub_emb, ent_emb, rel_emb, pred_type, score_callback):
    #    return score_callback(np.array(sub_emb), np.array(ent_emb), np.array(rel_emb), pred_type)
    #def get_subgraph_scores(self, sub_emb, ent_emb, rel_emb, pred_type):


    def predict(self):
        #self.predict_internal(self.x_test_raw, self.y_predicted_raw, "raw")
        self.predict_internal(self.x_test_fil, self.y_predicted_fil, "fil")
        self.y_predicted_raw = self.y_predicted_fil
        # replace all 0s with -1
        for x in self.y_predicted_fil:
            if x == 1:
                self.y_predicted_fil_abs.append(x)
            elif x == 0:
                self.y_predicted_fil_abs.append(-1)
        #self.predict_internal(self.x_test_fil, self.y_predicted_fil_abs, "abs")

    def get_dynamic_topk(self, ent, rel, sub_indexes):
        '''
            1. Search ent, rel in training triples
            2. If answer is found, look for the answer in sorted subgraphs
        '''
        #print("ent {}, rel {} ". format(ent, rel))
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
        for j, sub_index in enumerate(sub_indexes):
            if j > len(sub_indexes)/2:
                break
            for answer in answers:
                if answer in self.subgraphs[sub_index].data['entities']:
                    found_index.append(j)
                    break
        if len(found_index) == 0:
            return int(0.1 * len(sub_indexes))

        #print("found topk : ", found_index)
        return max(found_index)


    def predict_internal(self, x_test, y_predicted, setting):
        # Go over all test queries
        cnt_subgraphs_index = 0
        for index in tqdm(range(0, len(x_test), self.topk_answers_per_query)):
            print(index , " : ")
            features = np.array(x_test[index: index + self.topk_answers_per_query])
            ent = int(features[0][0])
            rel = int(features[0][1])
            topk_ans_entities = features[:, 2].astype(int)

            # call get_subgraph_scores only once and get all scores
            new_E = torch.Tensor(self.E[ent])[np.newaxis, :]
            new_R = torch.Tensor(self.R[rel])[np.newaxis, :]
            new_S = torch.Tensor(self.S)
            if self.model_name == "complex":
                s_re, s_im = torch.chunk(new_S, 2, dim = -1)
                e_re, e_im = torch.chunk(new_E, 2, dim = -1)
                r_re, r_im = torch.chunk(new_R, 2, dim = -1)
                subgraph_scores = self.model._calc(s_re, s_im, e_re, e_im, r_re, r_im)
            else:
                subgraph_scores = self.model._calc(torch.Tensor(self.S), new_E, new_R, self.type_prediction+'_batch')
            '''
            subgraph_scores = []
            for index, se in enumerate(self.S):
                if self.subgraphs[index].data['ent'] == ent and self.subgraphs[index].data['rel'] == rel:
                    subgraph_scores.append(np.inf)
                else:
                    subgraph_scores.append(self.get_subgraph_scores(se, self.E[ent], self.R[rel],  self.type_prediction, self.model_score))
            '''
            sub_indexes = np.argsort(subgraph_scores)

            topk_subgraphs = self.get_dynamic_topk(ent, rel, sub_indexes)

            # Check topk_subgraphs and if it is > 10
            threshold_subgraphs = int(self.subgraph_threshold_percentage * topk_subgraphs)

            threshold_subgraphs = min(len(sub_indexes)*0.1, threshold_subgraphs)
            # working
            '''
            for i, answer in enumerate(topk_ans_entities):
                cnt_presence_in_sub = 0;
                # Check in only topK subgraphs
                for j, sub_index in enumerate(sub_indexes[:topk_subgraphs]):
                    if answer in self.subgraphs[sub_index].data['entities']:
                        cnt_presence_in_sub += 1
                        #print("{} FOUND in subgraph # {}". format(answer, j))
                #if cnt_presence_in_sub != 0:
                self.cnt_subgraphs_dict[setting].append(str(cnt_presence_in_sub) + " / " + str(threshold_subgraphs))
                if cnt_presence_in_sub > threshold_subgraphs: #topk_subgraphs/2:
                    y_predicted.append(1)
                else:
                    y_predicted.append(0)
            '''
            cntMap = dict.fromkeys(topk_ans_entities)
            for sub_index in sub_indexes[:topk_subgraphs]:
                for i, answer in enumerate(topk_ans_entities):
                    if answer in self.subgraphs[sub_index].data['entities']:
                        if cntMap[answer] == None:
                            cntMap[answer] = 1
                        else:
                            cntMap[answer] += 1

            for key in topk_ans_entities:
                self.cnt_subgraphs_dict[setting].append(str(cntMap[key]) + " / " + str(threshold_subgraphs))
                #if cntMap[key] and cntMap[key] > threshold_subgraphs:
                if cntMap[key] and cntMap[key] > 0:#threshold_subgraphs:
                    y_predicted.append(1)
                else:
                    if setting == "abs":
                        y_predicted.append(-1)
                    else:
                        y_predicted.append(0)

