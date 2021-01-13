import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import json
import os
import argparse
import pickle
from support.utils import *
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--known_answers_train_file', default=None, dest='known_answers_train_file', type=str)
    parser.add_argument('--known_answers_valid_file', default=None, dest='known_answers_valid_file', type=str)
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', type = int, default = 10)
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--mode', dest = 'mode', type = str, default = "test", choices = ['train', 'valid', 'test'])
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    return parser.parse_args()

args = parse_args()

model_path = args.result_dir + '/' + args.db + "/embeddings/" + get_filename_model(args.db, args.model)
queries_full_path = args.result_dir + '/' + args.db + '/queries/' + get_filename_queries(args.db, args.mode, args.type_prediction)
checkpoint = load_checkpoint(model_path)
model = KgeModel.create_from(checkpoint)

ent_queries = []
rel_queries = []
with open(queries_full_path, "rt") as fin:
    records = json.loads(fin.read())
    for r in records:
        ent_queries.append(r['ent'])
        rel_queries.append(r['rel'])
        assert(args.type_prediction == 'tail' or r['type'] == 0)
        assert (args.type_prediction == 'head' or r['type'] == 1)

known_answers = set()
if args.mode == 'test':
    with open(args.known_answers_train_file, 'rt') as fin:
        nfacts = int(fin.readline())
        for l in fin:
            tkns = l.split(' ')
            h = int(tkns[0])
            t = int(tkns[1])
            r = int(tkns[2])
            if args.type_prediction == 'head':
                known_answers.add((t,r,h))
            elif args.type_prediction == 'tail':
                known_answers.add((h,r,t))
    with open(args.known_answers_valid_file, 'rt') as fin:
        nfacts = int(fin.readline())
        for l in fin:
            tkns = l.split(' ')
            h = int(tkns[0])
            t = int(tkns[1])
            r = int(tkns[2])
            if args.type_prediction == 'head':
                known_answers.add((t,r,h))
            elif args.type_prediction == 'tail':
                known_answers.add((h,r,t))

topk = args.topk
e = torch.Tensor(ent_queries).long()
r = torch.Tensor(rel_queries).long()
if args.type_prediction == 'tail':
    scores = model.score_sp(e, r)
else:
    scores = model.score_po(r, e)
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
        if (ent, rel, oi.item()) not in known_answers:
            filtered_answers.append({'entity_id' : oi.item(), 'score' : scores[index][oi.item()].item()})
            if len(filtered_answers) == topk:
                break
    assert(len(filtered_answers) == topk)
    q = records[index]
    q['answers_fil'] = filtered_answers
    q['answers_raw'] = raw_answers
    out.append(q)

answers_dir =  args.result_dir + '/' + args.db + "/answers/"
os.makedirs(answers_dir, exist_ok = True)
answers_filename = get_filename_answers(args.db, args.model, args.mode, args.topk, args.type_prediction)
with open(answers_dir + '/' + answers_filename, 'wb') as fout:
    pickle.dump(out, fout)