import torch
from support.dataset_fb15k237 import Dataset_FB15k237
from support.dataset_dbpedia50 import Dataset_dbpedia50
import json
import os
import argparse
import pickle
import numpy as np
from support.utils import *
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', type = int, default = 10)
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--mode', dest = 'mode', type = str, default = "test", choices = ['train', 'valid', 'test'])
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    return parser.parse_args()

args = parse_args()

# Load the dataset
dataset = None
annotations_dir = args.result_dir + '/' + args.db + '/annotations/'
if args.db == 'fb15k237':
    dataset = Dataset_FB15k237()
elif args.db == 'dbpedia50':
    dataset = Dataset_dbpedia50()

# Load the embedding model
embedding_model_typ = args.model
from support.embedding_model import Embedding_Model
embedding_model = Embedding_Model(args.result_dir, embedding_model_typ, dataset)

queries_full_path = args.result_dir + '/' + args.db + '/queries/' + get_filename_queries(args.db, args.mode, args.type_prediction)
ent_queries = []
rel_queries = []
with open(queries_full_path, "rt") as fin:
    records = json.loads(fin.read())
    for r in records:
        ent_queries.append(r['ent'])
        rel_queries.append(r['rel'])
        assert(args.type_prediction == 'tail' or r['type'] == 0)
        assert (args.type_prediction == 'head' or r['type'] == 1)

topk = args.topk
e = torch.Tensor(ent_queries).long()
r = torch.Tensor(rel_queries).long()
if args.type_prediction == 'tail':
    scores = embedding_model.score_sp(e, r)
else:
    scores = embedding_model.score_po(r, e)
o = torch.argsort(scores, dim=-1, descending = True)

out = []
assert(len(ent_queries) == len(records))
for index in tqdm(range(0, len(ent_queries))):
    ent = e[index].item()
    rel  = r[index].item()
    raw_answers = []
    filtered_answers = []
    for oi in o[index]:
        if len(raw_answers) < topk:
            raw_answers.append({'entity_id' : oi.item(), 'score' : scores[index][oi.item()].item()})
        if args.type_prediction == 'head':
            exists = dataset.exists_htr(oi.item(), ent, rel)
        else:
            exists = dataset.exists_htr(ent, oi.item(), rel)
        if not exists:
            filtered_answers.append({'entity_id' : oi.item(), 'score' : scores[index][oi.item()].item()})
            if len(filtered_answers) == topk:
                break
    assert(len(filtered_answers) == topk)
    q = records[index]
    q['answers_fil'] = filtered_answers
    q['answers_raw'] = raw_answers
    if args.mode == 'test':
        # Add also a list of all true answers that we have in our test set (useful to compute MRR)
        if args.type_prediction == 'head':
            answers = dataset.get_test_answers_for_tr(ent, rel)
        elif args.type_prediction == 'tail':
            answers = dataset.get_test_answers_for_hr(ent, rel)
        else:
            raise Exception("Not supported")
        q['answers_test_file'] = answers
    out.append(q)

answers_dir =  args.result_dir + '/' + args.db + "/answers/"
os.makedirs(answers_dir, exist_ok = True)
answers_filename = get_filename_answers(args.db, args.model, args.mode, args.topk, args.type_prediction)
with open(answers_dir + '/' + answers_filename, 'wb') as fout:
    pickle.dump(out, fout)