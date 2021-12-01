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

from numpy import linalg as LA

from openke.utils import DeepDict
from numpy.ctypeslib import ndpointer

def numpy_reverse(arr):
    ln = arr.shape[0]
    lidx, ridx = 0, ln - 1

    while lidx < ridx:
        rtmp = arr[ridx]
        arr[ridx] = arr[lidx]
        arr[lidx] = rtmp
        lidx += 1
        ridx -= 1
    return arr

class Tester(object):

    def __init__(self, db, model = None, model_name = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.ansHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ansTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ansHeadInTest.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.ansTailInTest.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
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
        self.model_name = model_name
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


    def run_ans_prediction(self, topk, outfile_name, dyntop, mode):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        training_range = tqdm(self.data_loader)
        test_data = []
        len_training = len(training_range)
        for index, [data_head, data_tail] in enumerate(training_range):
            #print(index, " / ", len_training)
            record = DeepDict()
            record['head'] = int(data_tail['batch_h'][0])
            record['tail'] = int(data_head['batch_t'][0])
            record['rel']  = int(data_head['batch_r'][0])
            suffix = ""
            if mode == "test":
                suffix = "_raw"
            '''
            # Head Predictions
            '''
            scores_head = self.test_one_step(data_head)

            if topk == 9999:
                topk_head = dyntop.get_dyn_topk(record['tail'], record['rel'], "head")
            else:
                topk_head = topk

            # Head answers raw
            #if self.model_name == "complex":
            #    answers_head = np.argsort(scores_head)[::-1]
            #    sorted_scores_head = np.sort(scores_head)[::-1]
            #else:
            answers_head = np.argsort(scores_head)
            sorted_scores_head = np.sort(scores_head)
            if self.model_name == "complex":
                answers_head = numpy_reverse(answers_head)
                sorted_scores_head = numpy_reverse(sorted_scores_head)

            truths_head = np.zeros(topk_head, dtype=int)
            self.lib.ansHead(answers_head.__array_interface__["data"][0], index, topk_head, truths_head.__array_interface__["data"][0])

            record['head_predictions' + suffix] = DeepDict()
            record['head_predictions' + suffix]['entities'] = answers_head[:topk_head].astype(int).tolist()
            record['head_predictions' + suffix]['scores' ] = sorted_scores_head[:topk_head].astype(float).tolist()
            record['head_predictions' + suffix]['correctness'] = truths_head[:topk_head].astype(int).tolist()

            # Head answers filtered
            if mode == "test":
                answers_head_fil = np.full(topk_head, -1, dtype = int)
                truths_head_fil = np.zeros(topk_head, dtype = int)
                # Filter answers from raw answers
                self.lib.ansHeadInTest(answers_head.__array_interface__["data"][0], index, topk_head, truths_head_fil.__array_interface__["data"][0], answers_head_fil.__array_interface__["data"][0])
                # Slice the corresponding scores of the filtered answers from the scores_head
                # Because answers_head was argsort'ed on scores_head
                scores_head_fil = scores_head[answers_head_fil]

                assert(len(answers_head_fil) == len(scores_head_fil) == len(truths_head_fil) == topk_head)
                record['head_predictions_fil'] = DeepDict()
                record['head_predictions_fil']['entities'] = answers_head_fil.astype(int).tolist()
                record['head_predictions_fil']['scores' ] = scores_head_fil.astype(float).tolist()
                record['head_predictions_fil']['correctness'] = truths_head_fil.astype(int).tolist()


            '''
            Tail answers
            '''
            scores_tail = self.test_one_step(data_tail)

            if topk == 9999:
                topk_tail = dyntop.get_dyn_topk(record['head'], record['rel'], "tail")
            else:
                topk_tail = topk

            # Tail answers raw
            answers_tail = np.argsort(scores_tail)
            sorted_scores_tail = np.sort(scores_tail)
            if self.model_name == "complex":
                answers_tail = numpy_reverse(answers_tail)
                sorted_scores_tail = numpy_reverse(sorted_scores_tail)

            truths_tail = np.zeros(topk_tail, dtype=int)
            self.lib.ansTail(answers_tail.__array_interface__["data"][0], index, topk_tail, truths_tail.__array_interface__["data"][0])
            record['tail_predictions' + suffix] = DeepDict()
            record['tail_predictions' + suffix]['entities'] = answers_tail[:topk_tail].astype(int).tolist()
            record['tail_predictions' + suffix]['scores' ] = sorted_scores_tail[:topk_tail].astype(float).tolist()
            record['tail_predictions' + suffix]['correctness'] = truths_tail[:topk_tail].astype(int).tolist()

            # Tail answers filtered
            if mode == "test":
                answers_tail_fil = np.full(topk_tail, -1, dtype = int)
                truths_tail_fil = np.zeros(topk_tail, dtype = int)
                # Filter answers from raw answers
                #print("UNM raw answers : ", answers_tail[:topk_tail].astype(int).tolist())
                self.lib.ansTailInTest(answers_tail.__array_interface__["data"][0], index, topk_tail, truths_tail_fil.__array_interface__["data"][0], answers_tail_fil.__array_interface__["data"][0])

                if record['head'] in answers_tail_fil:
                    print("#$%@"*30)
                    print(record['head'], ",",  record['tail'], ",", record['rel'])
                    print(answers_tail_fil)
                # Slice the corresponding scores of the filtered answers from the scores_tail
                # Because answers_tail was argsort'ed on scores_tail
                #print("UNM fil answers : ", answers_tail_fil)
                scores_tail_fil = scores_tail[answers_tail_fil]
                assert(len(answers_tail_fil) == len(scores_tail_fil) == len(truths_tail_fil) == topk_tail)
                record['tail_predictions_fil'] = DeepDict()
                record['tail_predictions_fil']['entities'] = answers_tail_fil.astype(int).tolist()
                record['tail_predictions_fil']['scores'] = scores_tail_fil.astype(float).tolist()
                record['tail_predictions_fil']['correctness'] = truths_tail_fil.astype(int).tolist()

            test_data.append(record)

        # Write all the records to the scores file
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
