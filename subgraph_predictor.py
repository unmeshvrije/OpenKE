import numpy as np
import json
import pickle
from tqdm import tqdm
from openke.module.model import TransE, RotatE, ComplEx
from subgraphs import Subgraph
from subgraphs import SUBTYPE
from numpy import linalg as LA
from subgraphs import read_triples
from openke.data import TrainDataLoader
import torch
import time
import scann
import timeit

class SubgraphPredictor():

    def __init__(self, type_prediction, db, topk_subgraphs, embeddings_file_path, subgraphs_file_path, sub_emb_file_path, emb_model, training_file_path, db_path, subgraph_threshold_percentage = 0.1):

        self.type_prediction = type_prediction
        self.topk_subgraphs = topk_subgraphs
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

    def set_test_triples(self, queries_file_path, num_test_queries):
        self.test_triples = read_triples(queries_file_path)[:num_test_queries]

    def set_logfile(self, logfile):
        self.logfile = logfile

    def init_entity_dict(self, entity_dict_file, rel_dict_file):
        with open(entity_dict_file, 'rb') as fin:
            self.entity_dict = pickle.load(fin)

        with open(rel_dict_file, 'rb') as fin:
            self.relation_dict = pickle.load(fin)

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

    def init_training_triples(self):
        triples = read_triples(self.training_file_path)
        # triples are in the form (h,r,t)
        # For type_prediction : head, we sort by tail
        self.training_triples_head_predictions = sorted(triples, key = lambda l : (l[2], l[1]))
        self.training_triples_tail_predictions = sorted(triples, key = lambda l : (l[2], l[0]))

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
        # This is crucial
        self.entity_total = self.train_dataloader.get_ent_tot()
        self.relation_total = self.train_dataloader.get_rel_tot()
        self.model.cuda()

    def init_embeddings(self, emb_model):
        #if emb_model == "complex":
        #    model = kge.model.KgeModel.load_from_checkpoint(self.emb_file_path)
        #    E_temp = model._entity_embedder._embeddings_all()
        #    R_temp = model._relation_embedder._embeddings_all()
        #    self.E = E_temp.tolist()
        #    self.R = R_temp.tolist()
        #else:
        with open (self.emb_file_path, 'r') as fin:
            parameters = json.loads(fin.read())
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i]).to('cuda')

        self.E = parameters['ent_embeddings.weight']
        self.R = parameters['rel_embeddings.weight']

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


    def get_dynamic_topk(self, ent, rel, sub_indexes, type_pred):
        '''
            1. Search ent, rel in training triples
            2. If answer is found, look for the answer in sorted subgraphs
        '''
        #print("ent {}, rel {} ". format(ent, rel))
        if type_pred == "head":
            training_triples = training_triples_head_predictions
        else:
            training_triples = training_triples_tail_predictions

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


    def predict(self, dynamic_topk = False):
        hitsHead = 0
        hitsTail = 0
        head_subgraph_comparisons = 0
        tail_subgraph_comparisons = 0
        max_subset_size_head = 0
        max_subset_size_tail = 0
        dim = self.E.size()[1]
        #print("dim = ", dim)
        all_tail_answer_embeddings = torch.empty(0, dim).to('cuda')
        all_head_answer_embeddings = torch.empty(0, dim).to('cuda')

        if self.test_triples is None:
            print("ERROR: set_test_triples() is not called.")
            return

        for index in tqdm(range(0, len(self.test_triples))):
            head = int(self.test_triples[index][0])
            tail = int(self.test_triples[index][1])
            rel  = int(self.test_triples[index][2])

            #time_start = timeit.default_timer()
            new_H = self.E[head]
            new_R = self.R[rel]
            new_T = self.E[tail]
            new_S = torch.Tensor(self.S).to('cuda')
            if self.model_name == "complex":
                s_re, s_im = torch.chunk(new_S, 2, dim = -1).to('cuda')
                h_re, h_im = torch.chunk(new_H, 2, dim = -1).to('cuda')
                t_re, t_im = torch.chunk(new_T, 2, dim = -1).to('cuda')
                r_re, r_im = torch.chunk(new_R, 2, dim = -1).to('cuda')
                subgraph_scores_head_prediction = self.model._calc(s_re, s_im, t_re, t_im, r_re, r_im)
                subgraph_scores_tail_prediction = self.model._calc(s_re, s_im, h_re, h_im, r_re, r_im)
            elif self.model_name == "rotate":
                new_H.unsqueeze_(0)
                new_T.unsqueeze_(0)
                new_R.unsqueeze_(0)
                subgraph_scores_head_prediction = self.model._calc(new_S, new_T, new_R, 'head_batch')
                subgraph_scores_tail_prediction = self.model._calc(new_H, new_S, new_R, 'tail_batch')

                answer_embedding_head = self.model._calc_embedding(new_H, new_T, new_R, 'head_batch')
                answer_embedding_tail = self.model._calc_embedding(new_H, new_T, new_R, 'tail_batch')

                temp = answer_embedding_tail.unbind()
                answer_embedding_tail = torch.cat(temp, dim = -1).squeeze(0)
                all_tail_answer_embeddings = torch.cat((all_tail_answer_embeddings, answer_embedding_tail), dim = 0)

                temp = answer_embedding_head.unbind()
                answer_embedding_head = torch.cat(temp, dim = -1).squeeze(0)
                all_head_answer_embeddings = torch.cat((all_head_answer_embeddings, answer_embedding_head), dim = 0)
            #time_end = timeit.default_timer()

            # Set scores of known subgraph(s) to infinity.
            #time_start = timeit.default_timer()
            for index, se in enumerate(self.S):
                if self.subgraphs[index].data['ent'] == head and self.subgraphs[index].data['rel'] == rel:
                    subgraph_scores_tail_prediction[index] = np.inf
                if self.subgraphs[index].data['ent'] == tail and self.subgraphs[index].data['rel'] == rel:
                    subgraph_scores_head_prediction[index] = np.inf

            sub_indexes_head_prediction = torch.argsort(subgraph_scores_head_prediction)
            sub_indexes_tail_prediction = torch.argsort(subgraph_scores_tail_prediction)
            #time_end = timeit.default_timer()
            #print("time taken to sort scores = {}s".format((time_end - time_start)*1000))

            if dynamic_topk:
                topk_subgraphs_head = self.get_dynamic_topk(tail, rel, sub_indexes, "head")
                topk_subgraphs_tail = self.get_dynamic_topk(head, rel, sub_indexes, "tail")
                # Check topk_subgraphs and if it is >= 10
                topk_subgraphs_head = max(10, topk_subgraphs_head)
                topk_subgraphs_tail = max(10, topk_subgraphs_tail)
            else:
                topk_subgraphs_head = self.topk_subgraphs
                topk_subgraphs_tail = self.topk_subgraphs

            time_start = timeit.default_timer()
            subset_head_predictions = set()
            for sub_index in sub_indexes_head_prediction[:topk_subgraphs_head]:
                subset_head_predictions.update(self.subgraphs[sub_index].data['entities'])
            if head in subset_head_predictions:
                hitsHead += 1
            head_subgraph_comparisons += len(subset_head_predictions)
            max_subset_size_head = max(len(subset_head_predictions), max_subset_size_head)

            subset_tail_predictions = set()
            for sub_index in sub_indexes_tail_prediction[:topk_subgraphs_tail]:
                subset_tail_predictions.update(self.subgraphs[sub_index].data['entities'])
            if tail in subset_tail_predictions:
                hitsTail += 1
            tail_subgraph_comparisons += len(subset_tail_predictions)
            max_subset_size_tail = max(len(subset_tail_predictions), max_subset_size_tail)
            time_end = timeit.default_timer()

        # calculate recall
        print("Recall (H) :", float(hitsHead)/float((len(self.test_triples))))
        print("Recall (T) :", float(hitsTail)/float((len(self.test_triples))))
        head_normal_comparisons = self.entity_total * hitsHead
        print("%Red (H)    :", float(head_normal_comparisons - head_subgraph_comparisons)/
        float(head_normal_comparisons)*100)
        tail_normal_comparisons = self.entity_total * hitsTail
        print("%Red (H)    :", float(tail_normal_comparisons - tail_subgraph_comparisons)/
        float(tail_normal_comparisons)*100)

        #ScaNN based scores
        dataset = self.E.cpu().numpy()
        queries_head = all_head_answer_embeddings.cpu().numpy()
        queries_tail = all_tail_answer_embeddings.cpu().numpy()

        normalized_dataset = dataset / np.linalg.norm(dataset, axis = 1)[:, np.newaxis]
        # Create ScaNN searcher
        topk_scann = max(max_subset_size_head, max_subset_size_tail)
        print(max_subset_size_head)
        print(max_subset_size_tail)
        print("topk scann", topk_scann)
        #searcher = scann.ScannBuilder(normalized_dataset, topk_scann, "dot_product").score_brute_force().create_pybind()
        searcher = scann.ScannBuilder(normalized_dataset, 2000, "dot_product").tree(1000, 100).score_ah(2, anisotropic_quantization_threshold = 0.2).create_pybind()
        print(type(searcher))

        start = time.time()
        head_neighbours, head_distances = searcher.search_batched(queries_head)
        tail_neighbours, tail_distances = searcher.search_batched(queries_tail)
        end = time.time()

        print("head neighbours : ")
        print(head_neighbours.shape)
        print(head_neighbours[:5])
        print(tail_neighbours.shape)
        def compute_scann_recall(neighbours, true_neighbours):
            #total = 0
            #for gt_row, row in zip(true_neighbours, neighbours):
            #    total += np.intersect1d(gt_row, row).shape[0]
            #return total / true_neighbours.size
            hits_scann = 0
            for i, x in enumerate(neighbours):
                #print (true_neighbours[i] , " : ", x)
                if true_neighbours[i] in x:
                    hits_scann += 1
            return float(hits_scann) / float(len(true_neighbours)) * 100

        true_head_neighbours = np.array(self.test_triples)[:, 0]
        true_tail_neighbours = np.array(self.test_triples)[:, 1]


        print("Recall (H) ScaNN : ", compute_scann_recall(head_neighbours, true_head_neighbours))
        print("Recall (T) ScaNN : ", compute_scann_recall(tail_neighbours, true_tail_neighbours))
        print("Time: ", end - start)

    #def predict_internal(self, ent, rel, ans, tester):
    #    # call get_subgraph_scores only once and get all scores
    #    new_E = torch.Tensor(self.E[ent])[np.newaxis, :]
    #    new_R = torch.Tensor(self.R[rel])[np.newaxis, :]
    #    new_S = torch.Tensor(self.S)
    #    if self.model_name == "complex":
    #        s_re, s_im = torch.chunk(new_S, 2, dim = -1)
    #        e_re, e_im = torch.chunk(new_E, 2, dim = -1)
    #        r_re, r_im = torch.chunk(new_R, 2, dim = -1)
    #        subgraph_scores = self.model._calc(s_re, s_im, e_re, e_im, r_re, r_im)
    #    else:
    #        #subgraph_scores = self.model._calc(torch.Tensor(self.S), new_E, new_R, self.type_prediction+'_batch')
    #        # this won't work:
    #        #TODO: here we need to pass only indices of Entities and relations
    #        # def forward() from the models will then find embeddings based on them
    #        subgraph_scores = self.model.predict({
    #        'batch_h': tester.to_var(np.array(self.E[ent]), tester.use_gpu),
    #        'batch_t': tester.to_var(np.array(self.S), tester.use_gpu),
    #        'batch_r': tester.to_var(np.array(self.R[rel]), tester.use_gpu),
    #        'mode': "tail_batch" # or head_batch
    #        })

    #    print("sub scores     = ", len(subgraph_scores))
    #    print("sub embeddings = ", len(self.S))

    #    # Set scores of known subgraph(s) to infinity.
    #    for index, se in enumerate(self.S):
    #        if self.subgraphs[index].data['ent'] == ent and self.subgraphs[index].data['rel'] == rel:
    #            subgraph_scores[index] = np.inf

    #    sub_indexes = np.argsort(subgraph_scores)
    #    topk_subgraphs = 5#self.get_dynamic_topk(ent, rel, sub_indexes)

    #    # Check topk_subgraphs and if it is > 10
    #    #threshold_subgraphs = int(self.subgraph_threshold_percentage * topk_subgraphs)

    #    #threshold_subgraphs = min(len(sub_indexes)*0.1, threshold_subgraphs)
    #    # working
    #    '''
    #    for i, answer in enumerate(topk_ans_entities):
    #        cnt_presence_in_sub = 0;
    #        # Check in only topK subgraphs
    #        for j, sub_index in enumerate(sub_indexes[:topk_subgraphs]):
    #            if answer in self.subgraphs[sub_index].data['entities']:
    #                cnt_presence_in_sub += 1
    #                #print("{} FOUND in subgraph # {}". format(answer, j))
    #        #if cnt_presence_in_sub != 0:
    #        self.cnt_subgraphs_dict[setting].append(str(cnt_presence_in_sub) + " / " + str(threshold_subgraphs))
    #        if cnt_presence_in_sub > threshold_subgraphs: #topk_subgraphs/2:
    #            y_predicted.append(1)
    #        else:
    #            y_predicted.append(0)
    #    '''
    #    found_answer = False
    #    for sub_index in sub_indexes[:topk_subgraphs]:
    #        if ans in self.subgraphs[sub_index].data['entities']:
    #            print("Found in sub: ")#, self.subgraphs[sub_index].data)
    #            return True

