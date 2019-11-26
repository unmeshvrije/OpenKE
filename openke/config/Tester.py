# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

from openke.utils import DeepDict
from numpy.ctypeslib import ndpointer

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.ansHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ansTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ansHead.restype = ndpointer(dtype=ctypes.c_int64, shape=(1000,))
        self.lib.ansTail.restype = ndpointer(dtype=ctypes.c_int64, shape=(1000,))
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

    def run_ans_prediction(self, ent_embeddings, topk):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        training_range = tqdm(self.data_loader)
        test_data = []
        for index, [data_head, data_tail] in enumerate(training_range):
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
            #print("Scores :" , score)
            #print("indexes : ", indexes)
            #print(len(score))
            truths = np.zeros(topk)
            truths = self.lib.ansHead(indexes.__array_interface__["data"][0], index, topk, truths.__array_interface__["data"][0])

            record['head_predictions'] = DeepDict()
            record['head_predictions']['entity'] = indexes[:topk].astype(int).tolist()
            record['head_predictions']['score' ] = score[:topk].astype(float).tolist()
            record['head_predictions']['correctness'] = truths[:topk].astype(int).tolist()


            # Tail answers
            score = self.test_one_step(data_tail)
            indexes = np.argsort(score)
            truths = np.zeros(topk)
            truths = self.lib.ansTail(indexes.__array_interface__["data"][0], index, topk, truths.__array_interface__["data"][0])
            record['tail_predictions'] = DeepDict()
            record['tail_predictions']['entity'] = indexes[:topk].astype(int).tolist()
            record['tail_predictions']['score' ] = score[:topk].astype(float).tolist()
            record['tail_predictions']['correctness'] = truths[:topk].astype(int).tolist()

                #print(self.model.ent_embeddings(emb_index.cuda()))
                #if truths[i] == 1:
                    #print("{}) {} () -> {}".format(i, entity,  truths[i]))
            # append the record to test_data
            test_data.append(record)
            with open("./results-scores.json", "w") as fout:
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
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
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
