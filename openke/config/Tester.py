# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import random
from random import shuffle
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

from numpy import linalg as LA

from openke.utils import DeepDict
from numpy.ctypeslib import ndpointer

class Tester(object):

    def __init__(self, db, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.ansHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ansTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        #self.lib.ansHead.restype = ndpointer(dtype=ctypes.c_int64, shape=(1000,))
        #self.lib.ansTail.restype = ndpointer(dtype=ctypes.c_int64, shape=(1000,))
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.db = db
        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def get_subgraph_scores(self, S, e, r, pred_type):
        S = np.array(S, dtype=float)
        e = np.array(e, dtype=float)
        r = np.array(r, dtype=float)
        if pred_type == "tail":
            score = (e + r) - S
        else:
            score = S + (r - e)

        return LA.norm(score, 2) #torch.norm(score, 1, -1).flatten()

    def run_sub_prediction(self, E, R, topk, outfile_name, spo_subgraphs, pos_subgraphs, spo_avg_embeddings, pos_avg_embeddings, filtered = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        training_range = tqdm(self.data_loader)
        test_data = []
        len_training = len(training_range)
        for index, [data_head, data_tail] in enumerate(training_range):
            print(index, " / ", len_training)
            # Head answers
            #print("tail : ", data_head['batch_t'][0])
            #print("head : ", data_tail['batch_h'][0])
            #print("data rel  : ", data_tail['batch_r'][0])
            #print(type(data_tail['batch_h'][0]))
            record = DeepDict()
            record['head'] = int(data_tail['batch_h'][0])
            record['tail'] = int(data_head['batch_t'][0])
            record['rel']  = int(data_head['batch_r'][0])

            score = self.test_one_step(data_head)
            indexes = np.argsort(score)
            sorted_scores = np.sort(score)
            #print("Scores :" , score)
            #print("indexes : ", indexes)
            #print(len(score))
            truths = np.zeros(topk, dtype=int)

            if filtered:
                self.lib.ansHeadInTest(indexes.__array_interface__["data"][0], index, topk, truths.__array_interface__["data"][0])
            else:
                self.lib.ansHead(indexes.__array_interface__["data"][0], index, topk, truths.__array_interface__["data"][0])

            #''' New code
            t = record['tail']
            r = record['rel']
            subgraph_scores = []
            print("t, r => ", t, r)
            for pae in pos_avg_embeddings:
                subgraph_scores.append(self.get_subgraph_scores(pae, E.weight.data[t], R.weight.data[r], "head"))
            #print("# of scores ", len(subgraph_scores))
            sub_indexes = np.argsort(subgraph_scores)
            topk_entities = indexes[:topk].astype(int).tolist()
            correctness   = truths[:topk].astype(int).tolist()
            print(sub_indexes[:topk])
            for i, answer in enumerate(topk_entities):
                for j, sub_index in enumerate(sub_indexes):
                    if answer in pos_subgraphs[sub_index].data['entities']:
                        print("{} FOUND in subgraph # {}". format(answer, j))
            #''' New code ends

            record['head_predictions'] = DeepDict()
            record['head_predictions']['entity'] = indexes[:topk].astype(int).tolist()
            record['head_predictions']['score' ] = sorted_scores[:topk].astype(float).tolist()
            record['head_predictions']['correctness'] = truths[:topk].astype(int).tolist()
            #print("just printing the list")
            #print(truths.tolist())


            # Tail answers
            score_tail = self.test_one_step(data_tail)
            indexes_tail = np.argsort(score_tail)
            sorted_score_tail = np.sort(score_tail)
            truths_tail = np.zeros(topk, dtype=int)

            if filtered:
                self.lib.ansTailInTest(indexes_tail.__array_interface__["data"][0], index, topk, truths_tail.__array_interface__["data"][0])
            else:
                self.lib.ansTail(indexes_tail.__array_interface__["data"][0], index, topk, truths_tail.__array_interface__["data"][0])
            record['tail_predictions'] = DeepDict()
            record['tail_predictions']['entity'] = indexes_tail[:topk].astype(int).tolist()
            record['tail_predictions']['score' ] = sorted_score_tail[:topk].astype(float).tolist()
            record['tail_predictions']['correctness'] = truths_tail[:topk].astype(int).tolist()
            test_data.append(record)
        # Write all the records to the scores file
        with open(outfile_name, "w") as fout:
            fout.write(json.dumps(test_data))

    def run_ans_prediction(self, ent_embeddings, topk, outfile_name, filtered = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        training_range = tqdm(self.data_loader)
        test_data = []
        len_training = len(training_range)
        for index, [data_head, data_tail] in enumerate(training_range):
            print(index, " / ", len_training)
            # Head answers
            #print("tail : ", data_head['batch_t'][0])
            #print("head : ", data_tail['batch_h'][0])
            #print("data rel  : ", data_tail['batch_r'][0])
            #print(type(data_tail['batch_h'][0]))
            record = DeepDict()
            record['head'] = int(data_tail['batch_h'][0])
            record['tail'] = int(data_head['batch_t'][0])
            record['rel']  = int(data_head['batch_r'][0])

            score = self.test_one_step(data_head)
            indexes = np.argsort(score)
            sorted_scores = np.sort(score)
            #print("Scores :" , score)
            #print("indexes : ", indexes)
            #print(len(score))
            truths = np.zeros(topk, dtype=int)
            if filtered:
                self.lib.ansHeadInTest(indexes.__array_interface__["data"][0], index, topk, truths.__array_interface__["data"][0])
            else:
                self.lib.ansHead(indexes.__array_interface__["data"][0], index, topk, truths.__array_interface__["data"][0])

            record['head_predictions'] = DeepDict()
            record['head_predictions']['entity'] = indexes[:topk].astype(int).tolist()
            record['head_predictions']['score' ] = sorted_scores[:topk].astype(float).tolist()
            record['head_predictions']['correctness'] = truths[:topk].astype(int).tolist()
            test_unm_head = truths[:topk].astype(int).tolist()
            #print(test_unm_head)
            #print("just printing the list")
            #print(truths.tolist())


            # Tail answers
            score_tail = self.test_one_step(data_tail)
            indexes_tail = np.argsort(score_tail)
            sorted_score_tail = np.sort(score_tail)
            truths_tail = np.zeros(topk, dtype=int)

            if filtered:
                self.lib.ansTailInTest(indexes_tail.__array_interface__["data"][0], index, topk, truths_tail.__array_interface__["data"][0])
            else:
                self.lib.ansTail(indexes_tail.__array_interface__["data"][0], index, topk, truths_tail.__array_interface__["data"][0])
            record['tail_predictions'] = DeepDict()
            record['tail_predictions']['entity'] = indexes_tail[:topk].astype(int).tolist()
            record['tail_predictions']['score' ] = sorted_score_tail[:topk].astype(float).tolist()
            record['tail_predictions']['correctness'] = truths_tail[:topk].astype(int).tolist()
            test_data.append(record)
        # Write all the records to the scores file
        random.seed(42)
        shuffle(test_data)
        with open(outfile_name, "w") as fout:
            fout.write(json.dumps(test_data))

    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            #print("unm : len : " , len(score))
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            #score = self.test_one_step(data_tail)
            #self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)
        return mrr, mr, hit10, hit3, hit1

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod
