import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, ComplEx
from openke.module.loss import MarginLoss, SigmoidLoss, SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader, TrainingAsTestDataLoader
import os
import sys
import json
import argparse
import pickle
import torch
import torch.nn as nn

from subgraphs import Subgraph
from subgraphs import SUBTYPE
from dynamic_topk import DynamicTopk

def parse_args():
    parser = argparse.ArgumentParser(description = 'Train embeddings of the KG with a given model')
    parser.add_argument('--gpu', dest ='gpu', help = 'Whether to use gpu or not', action = 'store_true')
    parser.add_argument('-result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--mode', dest = 'mode', type = str, choices = ['train', 'test', 'trainAsTest', 'subtest'], \
    help = 'Choice of the mode: train and test are intuitive. trainAsTest uses training data as test', default = None)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--model', dest = 'model', type = str, default = 'transe')
    parser.add_argument('--dyntopk-spo', dest = 'dyntopk_spo', type = str, default = "/var/scratch2/uji300/OpenKE-results/fb15k237/fb15k237-dynamic-topk-tail.pkl")
    parser.add_argument('--dyntopk-pos', dest = 'dyntopk_pos', type = str, default = "/var/scratch2/uji300/OpenKE-results/fb15k237/fb15k237-dynamic-topk-head.pkl")
    parser.add_argument('--topk', dest = 'topk', type = int, default = 10, help = "-1 means dynamic topk")
    return parser.parse_args()

args = parse_args()

# Global constants
N_DIM = 200 # Number of dimensions for embeddings

# Paths
db_path = "./benchmarks/" + args.db + "/"
result_dir = args.result_dir + args.db + "/"
os.makedirs(result_dir, exist_ok = True)
checkpoint_path = result_dir + args.db + "-" + args.model + ".ckpt"
result_path     = result_dir + args.db + "-" + args.model + ".json"

train_dataloader = TrainDataLoader(
    in_path = db_path,
    nbatches = 100,
    threads = 8,
    sampling_mode = "normal",
    bern_flag = 1,
    filter_flag = 1,
    neg_ent = 25,
    neg_rel = 0
    )

def choose_model():
    model = None
    model_with_loss = None
    if args.model == "transe":
        model = TransE(
                ent_tot = train_dataloader.get_ent_tot(),
                rel_tot = train_dataloader.get_rel_tot(),
                dim = N_DIM,
                p_norm = 1,
                norm_flag = True
                )
        # define the loss function
        model_with_loss = NegativeSampling(
            model = model,
            loss = MarginLoss(margin = 1.0),
            batch_size = train_dataloader.get_batch_size()
            )
    elif args.model == "complex":
        model = ComplEx(
                ent_tot = train_dataloader.get_ent_tot(),
                rel_tot = train_dataloader.get_rel_tot(),
                dim = N_DIM
                );
        # define the loss function
        model_with_loss = NegativeSampling(
            model = model,
            loss = MarginLoss(margin = 5.0),
            batch_size = train_dataloader.get_batch_size()
            )

    return model, model_with_loss

def load_pickle(filename):
    with open(filename, 'rb') as fin:
        data = pickle.load(fin)
    return data

if args.mode == "train":

    model, model_with_loss = choose_model()
    trainer = Trainer(model = model_with_loss, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = args.gpu)
    trainer.run()

    model.save_checkpoint(checkpoint_path)
    model.save_parameters(result_path)
elif args.mode == "test":
    test_dataloader = TestDataLoader(db_path, "link")
    model, model_with_loss = choose_model()
    model.load_checkpoint(checkpoint_path)
    model.load_parameters(result_path)
    tester = Tester(args.db, model = model, data_loader = test_dataloader, use_gpu = args.gpu)
    with open (result_path, 'r') as fin:
        params = json.loads(fin.read())
    outfile_name = result_dir + args.db + "-"+ args.model +"-results-scores-"+args.mode+"-topk-"+str(args.topk)+".json"

    dyntopk = None
    if args.topk == 9999:
        dyntopk = DynamicTopk()
        dyntopk.load(args.dyntopk_pos, args.dyntopk_spo)
    tester.run_ans_prediction(params['ent_embeddings.weight'], args.topk, outfile_name, dyntopk, args.mode)
elif args.mode == "trainAsTest":
    new_train_dataloader = TrainingAsTestDataLoader(db_path, "link")
    model, model_with_loss = choose_model()
    model.load_checkpoint(checkpoint_path)
    model.load_parameters(result_path)
    tester = Tester(args.db, model = model, data_loader = new_train_dataloader, use_gpu = args.gpu)
    with open (result_path, 'r') as fin:
        params = json.loads(fin.read())
    outfile_name = result_dir + args.db + "-" + args.model + "-results-scores-"+args.mode+"-topk-"+str(args.topk)+".json"
    dyntopk = None
    if args.topk == 9999:
        dyntopk = DynamicTopk()
        dyntopk.load(args.dyntopk_pos, args.dyntopk_spo)
    tester.run_ans_prediction(params['ent_embeddings.weight'], args.topk, outfile_name, dyntopk, args.mode)
