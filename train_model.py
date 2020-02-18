import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, HolE, ComplEx
from openke.module.loss import MarginLoss, SigmoidLoss, SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader, TrainingAsTestDataLoader
import os
import sys
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'Train embeddings of the KG with a given model')
    parser.add_argument('--gpu', dest ='gpu', help = 'Whether to use gpu or not', action = 'store_true')
    parser.add_argument('--filtered', dest ='filtered', help = 'Whether to use filtered setting or not', action = 'store_true')
    parser.add_argument('--mode', dest = 'mode', type = str, choices = ['train', 'test', 'trainAsTest'], \
    help = 'Choice of the mode: train and test are intuitive. trainAsTest uses training data as test', default = None)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--model', dest = 'model', type = str, default = 'transe')
    parser.add_argument('--topk', dest = 'topk', type = int, default = 10)
    return parser.parse_args()

args = parse_args()

# Global constants
N_DIM = 200 # Number of dimensions for embeddings

# Paths
db_path = "./benchmarks/" + args.db + "/"
result_dir      = "/var/scratch2/uji300/OpenKE-results/" + args.db + "/"
os.makedirs(result_dir, exist_ok = True)
os.makedirs(result_dir + "./checkpoint", exist_ok = True)
os.makedirs(result_dir + "./result", exist_ok = True)
checkpoint_path = result_dir + "./checkpoint/" + args.db + "-" + args.model + ".ckpt"
result_path     = result_dir + ".result/"+ args.db + "-" + args.model + ".json"

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
            loss = MarginLoss(margin = 5.0),
            batch_size = train_dataloader.get_batch_size()
            )
    elif args.model == "hole":
        model = HolE(
                ent_tot = train_dataloader.get_ent_tot(),
                rel_tot = train_dataloader.get_rel_tot(),
                dim = N_DIM
                );
        # define the loss function
        model_with_loss = NegativeSampling(
            model = model,
            loss = SigmoidLoss(),
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

if args.mode == "train":
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

    model, model_with_loss = choose_model()
    trainer = Trainer(model = model_with_loss, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = args.gpu)
    trainer.run()

    model.save_checkpoint(checkpoint_path)
    model.save_parameters(result_path)
elif args.mode == "test":
    test_dataloader = TestDataLoader(db_path, "link")
    model = choose_model()
    model.load_checkpoint(checkpoint_path)
    model.load_parameters(result_path)
    tester = Tester(args.db, model = model, data_loader = test_dataloader, use_gpu = args.gpu)
    with open (result_path, 'r') as fin:
        params = json.loads(fin.read())
    if args.filtered:
        fil = "filtered"
    else:
        fil = "unfiltered"
    outfile_name = result_dir + args.db + "-results-scores-"+args.mode+"-"+str(topk)+"-"+fil+".json"
    tester.run_ans_prediction(params['ent_embeddings.weight'], topk, outfile_name, filtered = args.filtered)
elif args.mode == "trainAsTest":
    new_train_dataloader = TrainingAsTestDataLoader(db_path, "link")
    model = choose_model()
    model.load_checkpoint(checkpoint_path)
    model.load_parameters(result_path)
    tester = Tester(args.db, model = model, data_loader = new_train_dataloader, use_gpu = args.gpu)
    with open (result_path, 'r') as fin:
        params = json.loads(fin.read())
    outfile_name = result_dir + args.db + "-results-scores-"+args.mode+"-"+str(topk)+"-"+fil+".json"
    tester.run_ans_prediction(params['ent_embeddings.weight'], topk, outfile_name)
