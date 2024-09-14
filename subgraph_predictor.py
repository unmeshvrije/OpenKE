import numpy as np
import json
import pickle5 as pickle
from tqdm import tqdm
from collections import defaultdict
from openke.module.model import TransE, RotatE, ComplEx, DistMult, HolE
from subgraphs import Subgraph
from subgraphs import SUBTYPE
from numpy import linalg as LA
from subgraphs import read_triples, make_adjacency_dict, update_adjacency_dict
from openke.data import TrainDataLoader
import torch
import time
import timeit
import kge.model
import torch.nn.functional as F
import nanopq
import os
import random

from util import timer

class SubgraphPredictor():

    def __init__(self, db, subgraph_type, topk_subgraphs, embeddings_file_path, subgraphs_file_path, sub_emb_dir_path, emb_model, training_file_path, db_path, subgraph_threshold_percentage = 0.1, score_func = "avg"):

        self.db = db
        self.subgraph_type = subgraph_type
        self.topk_subgraphs = topk_subgraphs
        self.dynamic_topk = False
        self.dynamic_threshold = False
        if topk_subgraphs == -1:
            self.dynamic_topk = True
        elif topk_subgraphs == -2:
            self.dynamic_threshold = True

        self.emb_file_path = embeddings_file_path
        self.sub_file_path = subgraphs_file_path

        # fb15k237-rotate-avgemb-tau-10.pkl
        if not sub_emb_dir_path.endswith("/"):
            sub_emb_dir_path += "/"
        self.sub_avgemb_file_path = sub_emb_dir_path + self.db + "-" + emb_model + "-" + self.subgraph_type + "-avgemb-tau-10.pkl"
        self.sub_varemb_file_path = sub_emb_dir_path + self.db + "-" + emb_model + "-" + self.subgraph_type + "-varemb-tau-10.pkl"

        self.training_file_path = training_file_path
        self.training_triples = read_triples(training_file_path)
        self.adj_list_out, self.adj_list_in = make_adjacency_dict(self.training_triples)
        self.subgraph_threshold_percentage = subgraph_threshold_percentage
        self.score_func = score_func

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

    def set_test_triples(self, queries_file_path, num_test_queries):
        self.test_triples = read_triples(queries_file_path)[:num_test_queries]
        self.adj_list_out, self.adj_list_in = update_adjacency_dict(self.adj_list_out, self.adj_list_in, self.test_triples)

    def set_logfile(self, logfile):
        self.logfile = logfile

    def init_entity_dict(self, entity_dict_file, rel_dict_file):
        with open(entity_dict_file, 'rb') as fin:
            self.entity_dict = pickle.load(fin)

        with open(rel_dict_file, 'rb') as fin:
            self.relation_dict = pickle.load(fin)

    @timer
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
            if (index+1) % self.topk_subgraphs == 0:
                print("*" * 80, file = log)

        log.close()

    @timer
    def init_training_triples(self):
        self.triples = read_triples(self.training_file_path)
        # triples are in the form (h,t,r)
        # For type_prediction : head, we sort by tail
        self.training_triples_head_predictions = sorted(self.triples, key = lambda l : (l[2], l[1]))
        self.training_triples_tail_predictions = sorted(self.triples, key = lambda l : (l[2], l[0]))

        self.spo_dict = defaultdict(list) # key (h,r) -> value (list of t1, t2, ...)
        self.pos_dict = defaultdict(list) # key (t,r) -> value (list of h1, h2, ...)

        for head, tail, relation in self.triples:
            self.spo_dict[(head, relation)].append(tail)
            self.pos_dict[(tail, relation)].append(head)

        '''
        self.training_triples_head_predictions = {}
        self.training_triples_tail_predictions = {}
        print("HERE " *50, flush=True)
        for i in tqdm(range(0, len(triples))):
            print("{}, {}, {}".format(triples[i][0], triples[i][1], triples[i][2]), flush=True)
            h = triples[i][0]
            r = triples[i][1]
            t = triples[i][2]
            heads = self.training_triples_head_predictions.get((r,t), [])
            heads.append(h)
            tails = self.training_triples_tail_predictions.get((r,h), [])
            tails.append(t)
        '''

    @timer
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
                    )
        elif emb_model == "distmult":
            N_DIM = 200
            self.model = DistMult(
                    ent_tot = self.train_dataloader.get_ent_tot(),
                    rel_tot = self.train_dataloader.get_rel_tot(),
                    dim = N_DIM
                    #margin = 6.0,
                    #epsilon = 2.0
                    )
        elif emb_model == "hole":
            N_DIM = 200
            self.model = HolE(
                    ent_tot = self.train_dataloader.get_ent_tot(),
                    rel_tot = self.train_dataloader.get_rel_tot(),
                    dim = N_DIM
                    #margin = 6.0,
                    #epsilon = 2.0
                    )
        else:
            print(f"Unsupported model: {emb_model}")
            sys.exit()
        # This is crucial
        self.entity_total = self.train_dataloader.get_ent_tot()
        self.relation_total = self.train_dataloader.get_rel_tot()
        self.model.cuda()

    @timer
    def init_embeddings(self, emb_model):
        with open (self.emb_file_path, 'r') as fin:
            parameters = json.loads(fin.read())
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i]).to('cuda')
        if emb_model == "complex":
            self.E = parameters['ent_re_embeddings.weight'] + parameters['ent_im_embeddings.weight']
            self.R = parameters['rel_re_embeddings.weight'] + parameters['rel_im_embeddings.weight']
        else:
            self.E = parameters['ent_embeddings.weight']
            self.R = parameters['rel_embeddings.weight']

    @timer
    def init_subgraphs(self):
        with open(self.sub_file_path, 'rb') as fin:
            self.subgraphs = pickle.load(fin)

    @timer
    def init_sub_embeddings(self):
        with open(self.sub_avgemb_file_path, 'rb') as fin:
            self.SA = torch.Tensor(pickle.load(fin)).to('cuda')
        with open(self.sub_varemb_file_path, 'rb') as fin:
            self.SV = torch.Tensor(pickle.load(fin)).to('cuda')

    #def get_subgraph_scores(self, sub_emb, ent_emb, rel_emb, pred_type, score_callback):
    #    return score_callback(np.array(sub_emb), np.array(ent_emb), np.array(rel_emb), pred_type)
    #def get_subgraph_scores(self, sub_emb, ent_emb, rel_emb, pred_type):

    def get_dynamic_threshold(self, ent, rel, ent_emb, rel_emb, type_pred, model_name):
        '''
            1. Search ent, rel in training triples
            2. If answer is found, look for the scores of these answers
            3. Get the minimum of these scores
        '''
        #print("ent {}, rel {} ". format(ent, rel))
        if type_pred == "head":
            training_triples = self.training_triples_head_predictions
        else:
            training_triples = self.training_triples_tail_predictions

        answers = []
        for index, triple in enumerate(training_triples):
            if triple[2] != rel:
                continue

            if triple[2] > rel:
                break

            if type_pred == "head":
                if triple[1] == ent:
                    answers.append(triple[0])
            elif type_pred == "tail":
                if triple[0] == ent:
                    answers.append(triple[1])

        if len(answers) == 0:
            return 0.0

        all_answer_emb = self.E[np.array(answers)]
        if model_name == "complex":
            a_re, a_im = torch.chunk(all_answer_emb, 2, dim = -1)
            e_re, e_im = torch.chunk(ent_emb, 2, dim = -1)
            r_re, r_im = torch.chunk(rel_emb, 2, dim = -1)
        if type_pred == "head":
            if model_name == "complex":
                all_answer_scores = self.model._calc(a_re, a_im, e_re, e_im, r_re, r_im)
            else:
                all_answer_scores = self.model._calc(all_answer_emb, ent_emb, rel_emb, 'head_batch')
        else:
            if model_name == "complex":
                all_answer_scores = self.model._calc(e_re, e_im, a_re, a_im, r_re, r_im)
            else:
                all_answer_scores = self.model._calc(ent_emb, all_answer_emb, rel_emb, 'tail_batch')

        #return torch.mean(all_answer_scores).cpu().numpy()
        return torch.min(all_answer_scores).cpu().numpy()

    def get_dynamic_topk(self, ent, rel, sub_indexes, type_pred):
        '''
            1. Search ent, rel in training triples
            2. If answer is found, look for the answer in sorted subgraphs
        '''
        if type_pred == "head":
            training_triples = self.training_triples_head_predictions
        else:
            training_triples = self.training_triples_tail_predictions

        answers = []
        for index, triple in enumerate(training_triples):
            if triple[2] != rel:
                continue

            if triple[2] > rel:
                break

            if type_pred == "head":
                if triple[1] == ent:
                    answers.append(triple[0])
            elif type_pred == "tail":
                if triple[0] == ent:
                    answers.append(triple[1])

        if len(answers) == 0:
            return int(0.1 * len(sub_indexes))

        '''
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

        return max(found_index)
        '''
        found_index = 0
        j = len(sub_indexes)-1
        while j > 0:
            for answer in answers:
                if answer in self.subgraphs[sub_indexes[j]].data['entities']:
                    found_index = j
                    break
            j //= 2

        return found_index if found_index > 0 else int(0.1 * len(sub_indexes))

    '''
    def get_matching_entities(self, sub_type, e, r):
        entities = []
        for triple in self.triples:
            if sub_type == SUBTYPE.SPO and triple[0] == e and triple[2] == r:
                entities.append(triple[1])
                if len(entities) == 10:
                    return entities
            elif triple[1] == e and triple[2] == r:
                entities.append(triple[0])
                if len(entities) == 10:
                    return entities
        return entities
    '''

    def get_matching_entities(self, sub_type, e, r):
        entities = []
        # TODO: return subgraphs with (e,r) or make dictionaries with 'r' as key and list of e's that are present in training set
        if sub_type == SUBTYPE.SPO:
            entities = self.spo_dict.get((e, r), [])[:10] # [triple[1] for triple in self.triples if triple[0] == e and triple[2] == r][:10]
        else:
            entities = self.pos_dict.get((e, r), [])[:10] # [triple[0] for triple in self.triples if triple[1] == e and triple[2] == r][:10]

        return entities


    def get_kl_divergence_scores(self, ent, rel, sub_type, db, model, sub_type_str):
        '''
        Get the entities with this ent and rel from db.
        sample some entities for trueAvg and trueVar embeddings
        now find KL divergence with these trueAvg and trueVar embeddings
        with all other subgraphs
        '''
        dim = self.E.size()[1]
        me = self.get_matching_entities(sub_type, ent, rel)
        count = len(me)
        n_subgraphs = len(self.subgraphs)
        if count == 0:
            return [0.0] * n_subgraphs
        summation = torch.sum(self.E[me])
        # TODO: check if we should pass mean to the kl_div() function
        mean = summation / count if count > 0 else summation

        # Calculate kl scores with all subgraphs
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
        kl_scores = [F.kl_div(self.SA[i], summation, reduction='batchmean') for i in range(n_subgraphs)]

        #with open(scores_file, 'wb') as fout:
        #    all_kl_scores[ent][rel] = kl_scores
        #    pickle.dump(all_kl_scores, fout, protocol = pickle.HIGHEST_PROTOCOL)

        return kl_scores

    def precalculate_kl_divergence_scores(self):
        print("Precomputing kl divergence scores...")
        kl_scores = dict()
        kl_scores['head'] = dict()
        kl_scores['tail'] = dict()
        for index in tqdm(range(0, len(self.test_triples))):
            head = int(self.test_triples[index][0])
            tail = int(self.test_triples[index][1])
            rel  = int(self.test_triples[index][2])

            if tail not in kl_scores['head']:
                kl_scores['head'][tail] = dict()
            if head not in kl_scores['tail']:
                kl_scores['tail'][head] = dict()

            #time_start = timeit.default_timer()
            new_H = self.E[head]
            new_R = self.R[rel]
            new_T = self.E[tail]
            new_S = self.SA
            subgraph_scores_head_prediction = torch.Tensor(self.get_kl_divergence_scores(tail, rel, SUBTYPE.POS, self.db, self.model_name, self.subgraph_type))
            subgraph_scores_tail_prediction = torch.Tensor(self.get_kl_divergence_scores(head, rel, SUBTYPE.SPO, self.db, self.model_name, self.subgraph_type))
            kl_scores['head'][tail][rel] = subgraph_scores_head_prediction
            kl_scores['tail'][head][rel] = subgraph_scores_tail_prediction
        return kl_scores
    
    def subgraph_of_right_type(self, index, expected_type):
        return (expected_type == "star" and self.subgraphs[index].data['subType'] in [SUBTYPE.SPO, SUBTYPE.POS]) or (expected_type == "diamond" and self.subgraphs[index].data['subType'] not in [SUBTYPE.SPO, SUBTYPE.POS])


    def predict(self, kl_scores_dir):
        self.hitsHead = 0
        self.hitsTail = 0
        precision_sum_head = 0
        precision_sum_tail = 0
        precision_value_count_head = 0
        precision_value_count_tail = 0
        hits_head_scann = 0
        hits_tail_scann = 0
        self.head_subgraph_comparisons = 0
        self.tail_subgraph_comparisons = 0
        max_subset_size_head = 0
        max_subset_size_tail = 0
        dim = self.E.size()[1]
        all_tail_answer_embeddings = torch.empty(0, dim).to('cuda')
        all_head_answer_embeddings = torch.empty(0, dim).to('cuda')

        dataset = self.E.cpu().numpy()
        normalized_dataset = dataset / np.linalg.norm(dataset, axis = 1)[:, np.newaxis]

        product_quantizator = nanopq.PQ(M = 8, Ks=128, verbose=True) #Instantiate quantizator with 10 subspaces
        X_code = []

        subgraph_center_dict = dict()
        if self.subgraph_type in ["star", "all"]:
            for index, se in enumerate(self.SA):
                if (self.subgraphs[index].data['subType'] in [SUBTYPE.SPO, SUBTYPE.POS]):
                    ent = self.subgraphs[index].data['ent']
                    rel = self.subgraphs[index].data['rel']
                    if ent not in subgraph_center_dict:
                        subgraph_center_dict[ent] = dict()
                    if rel not in subgraph_center_dict[ent]:
                        subgraph_center_dict[ent][rel] = []
                    subgraph_center_dict[ent][rel].append(index)
        if self.subgraph_type in ["diamond", "all"]:
            for index, se in enumerate(self.SA):
                if (self.subgraphs[index].data['subType'] not in [SUBTYPE.SPO, SUBTYPE.POS]):
                    ent1 = self.subgraphs[index].data['ent1']
                    ent2 = self.subgraphs[index].data['ent2']
                    rel1 = self.subgraphs[index].data['rel1']
                    rel2 = self.subgraphs[index].data['rel2']
                    if ent1 not in subgraph_center_dict:
                        subgraph_center_dict[ent1] = dict()
                    if rel1 not in subgraph_center_dict[ent1]:
                        subgraph_center_dict[ent1][rel1] = []
                    if ent2 not in subgraph_center_dict:
                        subgraph_center_dict[ent2] = dict()
                    if rel2 not in subgraph_center_dict[ent2]:
                        subgraph_center_dict[ent2][rel2] = []
                    subgraph_center_dict[ent1][rel1].append(index)
                    subgraph_center_dict[ent2][rel2].append(index)
        
        if self.test_triples is None:
            print("ERROR: set_test_triples() is not called.")
            return
        
        if self.score_func == "kl":
            scores_file = kl_scores_dir + self.db + '-' + self.model_name + '-' + self.subgraph_type + '-' + str(len(self.test_triples)) + '-kl-scores.pkl'
            if os.path.isfile(scores_file) and os.stat(scores_file).st_size != 0:
                with open(scores_file, 'rb') as fin:
                    kl_scores = pickle.load(fin)
            else:
                with open(scores_file, 'wb') as fout:
                    kl_scores = self.precalculate_kl_divergence_scores()
                    pickle.dump(kl_scores, fout, protocol = pickle.HIGHEST_PROTOCOL)


        if self.score_func == "nn":
            max_training_vector_count = 5000
            dim = 200
            if self.model_name == "rotate":
                dim = 400
            training_vectors_new = []
            training_vector_size = min(len(self.triples), max_training_vector_count)
            for index in range(0, training_vector_size):
                head_id = self.triples[index][0]
                tail_id = self.triples[index][1]
                training_vectors_new.append(self.E[head_id])
                training_vectors_new.append(self.E[tail_id])
            training_vectors_new = torch.cat(training_vectors_new)
            training_vectors_new = torch.reshape(training_vectors_new, [2 * training_vector_size, dim])
            product_quantizator.fit(training_vectors_new.cpu().numpy())
            X_code = product_quantizator.encode(self.SA.detach().cpu().numpy())

        #searcher = scann.ScannBuilder(normalized_dataset, 7000, "dot_product").tree(3000, 300, training_sample_size = 14541).score_ah(2, anisotropic_quantization_threshold = 0.2).reorder(4000).create_pybind()


        for index in tqdm(range(0, len(self.test_triples))):
            head = int(self.test_triples[index][0])
            tail = int(self.test_triples[index][1])
            rel  = int(self.test_triples[index][2])

            head_answers = self.adj_list_in[tail][rel]
            tail_answers = self.adj_list_out[head][rel]

            #time_start = timeit.default_timer()
            new_H = self.E[head]
            new_R = self.R[rel]
            new_T = self.E[tail]
            new_S = self.SA
            if self.score_func == "kl":
                # Compute KL divergence scores
                subgraph_scores_head_prediction = kl_scores['head'][tail][rel]
                subgraph_scores_tail_prediction = kl_scores['tail'][head][rel]
                new_R.unsqueeze_(0)
            elif self.score_func == "nn":
                query_head_prediction = self.model._vector_op(new_T, new_R, 'head_pred')
                query_tail_prediction = self.model._vector_op(new_H, new_R, 'tail_pred')
                subgraph_scores_head_prediction = torch.Tensor(product_quantizator.dtable(query_head_prediction.detach().cpu().numpy()).adist(X_code))
                subgraph_scores_tail_prediction = torch.Tensor(product_quantizator.dtable(query_tail_prediction.detach().cpu().numpy()).adist(X_code))
                new_R.unsqueeze_(0)
            else:
                if self.model_name == "complex":
                    s_re, s_im = torch.chunk(new_S, 2, dim = -1)
                    h_re, h_im = torch.chunk(new_H, 2, dim = -1)
                    t_re, t_im = torch.chunk(new_T, 2, dim = -1)
                    r_re, r_im = torch.chunk(new_R, 2, dim = -1)
                    subgraph_scores_head_prediction = self.model._calc(s_re, s_im, t_re, t_im, r_re, r_im)
                    subgraph_scores_tail_prediction = self.model._calc(s_re, s_im, h_re, h_im, r_re, r_im)
                else:# self.model_name == "rotate":
                    new_H.unsqueeze_(0)
                    new_T.unsqueeze_(0)
                    new_R.unsqueeze_(0)
                    subgraph_scores_head_prediction = self.model._calc(new_S, new_T, new_R, 'head_batch')
                    subgraph_scores_tail_prediction = self.model._calc(new_H, new_S, new_R, 'tail_batch')



            if head in subgraph_center_dict and rel in subgraph_center_dict[head]:
                for index in subgraph_center_dict[head][rel]:
                    subgraph_scores_tail_prediction[index] = np.inf
            if tail in subgraph_center_dict and rel in subgraph_center_dict[tail]:
                for index in subgraph_center_dict[tail][rel]:
                    subgraph_scores_head_prediction[index] = np.inf


            sub_indexes_head_prediction = torch.argsort(subgraph_scores_head_prediction)
            sub_indexes_tail_prediction = torch.argsort(subgraph_scores_tail_prediction)
            #time_end = timeit.default_timer()
            #print("time taken to sort scores = {}s".format((time_end - time_start)*1000))

            if self.dynamic_topk:
                topk_subgraphs_head = self.get_dynamic_topk(tail, rel, sub_indexes_head_prediction, "head")
                topk_subgraphs_tail = self.get_dynamic_topk(head, rel, sub_indexes_tail_prediction, "tail")
                # Check topk_subgraphs and if it is >= 10
                topk_subgraphs_head = max(10, topk_subgraphs_head)
                topk_subgraphs_tail = max(10, topk_subgraphs_tail)
                #print("Looking for answers in {}/{} subgraphs".format(topk_subgraphs_head, len(self.subgraphs)))
            elif self.dynamic_threshold:
                thresh_subgraphs_head = self.get_dynamic_threshold(tail, rel, new_T, new_R, "head", self.model_name)
                thresh_subgraphs_tail = self.get_dynamic_threshold(head, rel, new_H, new_R, "tail", self.model_name)

                sub_indexes_head_prediction = np.where(subgraph_scores_head_prediction.cpu().numpy() > thresh_subgraphs_head)[0]
                sub_indexes_tail_prediction = np.where(subgraph_scores_tail_prediction.cpu().numpy() > thresh_subgraphs_tail)[0]
                topk_subgraphs_head = -1
                topk_subgraphs_tail = -1 # consider all of these indexes
                #print("Looking for answers in {}/{} subgraphs".format(len(sub_indexes_head_prediction), len(self.subgraphs)))
            else:
                #print("topk subgraphs : ", self.topk_subgraphs)
                topk_subgraphs_head = self.topk_subgraphs
                topk_subgraphs_tail = self.topk_subgraphs

            time_start = timeit.default_timer()
            subset_head_predictions = set()

            def update_predictions_head(subgraph_type):
                if self.subgraph_type == "all":
                    k_to_consider = topk_subgraphs_head
                    for sub_index in sub_indexes_head_prediction:
                        if self.subgraph_of_right_type(sub_index, subgraph_type):
                            k_to_consider -= 1
                            subset_head_predictions.update(self.subgraphs[sub_index].data['entities'])
                            if k_to_consider == 0:
                                break
                else:
                    for sub_index in sub_indexes_head_prediction[:topk_subgraphs_head]:
                        subset_head_predictions.update(self.subgraphs[sub_index].data['entities'])

            def check_hit_head():
                if head in subset_head_predictions:
                    self.hitsHead += 1
                    self.head_subgraph_comparisons += len(subset_head_predictions)
                    return True
                return False

            relevant_subgraph_type = self.subgraph_type
            if self.subgraph_type == "all":
                relevant_subgraph_type = "star"
            update_predictions_head(relevant_subgraph_type)
            hit_found = check_hit_head()
            if hit_found == False and self.subgraph_type == "all":
                subset_head_predictions = set()
                update_predictions_head("diamond")
                check_hit_head()

            true_positives_head = 0
            for prediction in subset_head_predictions:
                if prediction in head_answers:
                    true_positives_head += 1
            if true_positives_head != 0:
                precision_sum_head += float(true_positives_head)/float(len(head_answers))
                precision_value_count_head += 1
            #print("head total sub comparisons {} ({})".format(len(subset_head_predictions), head_subgraph_comparisons))
            #max_subset_size_head = max(len(subset_head_predictions), max_subset_size_head)

            #print("Length of subset = ", len(subset_head_predictions))
            #topk_subgraphs_scann = min(100000, len(subset_head_predictions))
            #if topk_subgraphs_scann == 1000:
            #    hitsHead -= 1

            #print("Searching in {}".format(topk_subgraphs_scann))
            #head_neighbours, head_distances = searcher.search_batched(answer_embedding_head.cpu().numpy(), final_num_neighbors = topk_subgraphs_scann)
            #print("Actual neighbours returned :  ",len(np.squeeze(head_neighbours)))
            #print("Actual neighbours returned :  ",(np.squeeze(head_neighbours)))
            #if head in np.squeeze(head_neighbours):
            #    print("ScaNN HEAD FOUND")
            #    hits_head_scann += 1

            subset_tail_predictions = set()
            def update_predictions_tail(subgraph_type):
                if self.subgraph_type == "all":
                    k_to_consider = topk_subgraphs_head
                    for sub_index in sub_indexes_tail_prediction:
                        if self.subgraph_of_right_type(sub_index, subgraph_type):
                            k_to_consider -= 1
                            subset_tail_predictions.update(self.subgraphs[sub_index].data['entities'])
                            if k_to_consider == 0:
                                break
                else:
                    for sub_index in sub_indexes_tail_prediction[:topk_subgraphs_tail]:
                        subset_tail_predictions.update(self.subgraphs[sub_index].data['entities'])

            def check_hit_tail():
                if tail in subset_tail_predictions:
                    self.hitsTail += 1
                    self.tail_subgraph_comparisons += len(subset_tail_predictions)
                    return True
                return False

            relevant_subgraph_type = self.subgraph_type
            if self.subgraph_type == "all":
                relevant_subgraph_type = "star"
            update_predictions_tail(relevant_subgraph_type)
            hit_found = check_hit_tail()
            if hit_found == False and self.subgraph_type == "all":
                subset_head_predictions = set()
                update_predictions_tail("diamond")
                check_hit_tail()

            true_positives_tail = 0
            for prediction in subset_tail_predictions:
                if prediction in tail_answers:
                    true_positives_tail += 1
            if true_positives_tail != 0:
                precision_sum_tail += float(true_positives_tail)/float(len(tail_answers))
                precision_value_count_tail += 1
            #max_subset_size_tail = max(len(subset_tail_predictions), max_subset_size_tail)
            #topk_subgraphs_scann = min(100000, len(subset_tail_predictions))
            #if topk_subgraphs_scann == 1000:
            #    hitsTail -= 1

            #tail_neighbours, tail_distances = searcher.search_batched(answer_embedding_tail.cpu().numpy(), final_num_neighbors = topk_subgraphs_scann)
            #if tail in np.squeeze(tail_neighbours):
            #    print("ScaNN TAIL FOUND")
            #    hits_tail_scann += 1
            time_end = timeit.default_timer()

        # calculate recall
        print()
        print("Recall (H) :", float(self.hitsHead)/float((len(self.test_triples))))
        print("Recall (T) :", float(self.hitsTail)/float((len(self.test_triples))))
        if precision_value_count_head != 0:
            print("Precision (H) :", float(precision_sum_head)/float(precision_value_count_head))
        if precision_value_count_tail != 0:
            print("Precision (T) :", float(precision_sum_tail)/float(precision_value_count_tail))
        head_normal_comparisons = self.entity_total * self.hitsHead
        if head_normal_comparisons != 0:
            print("%Red (H)    :", float(head_normal_comparisons - self.head_subgraph_comparisons)/
            float(head_normal_comparisons)*100)
        tail_normal_comparisons = self.entity_total * self.hitsTail
        if tail_normal_comparisons != 0:
            print("%Red (T)    :", float(tail_normal_comparisons - self.tail_subgraph_comparisons)/
            float(tail_normal_comparisons)*100)
