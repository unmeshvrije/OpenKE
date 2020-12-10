import sys
import pickle as pl
import argparse
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--embfile', dest ='emb_file', type = str, help = 'File containing entity embeddings.')
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = 'OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = 'OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--model', dest ='model',type = str, default = "transe", help = 'Embedding model name.')
    parser.add_argument('--pred', dest ='pred', type = str, required = True, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    parser.add_argument('--true-out', dest='true_out_file', type=str, help='File containing the true /expected answers.', default='OpenKE-results/fb15k237/out/fb15k237-transe-annotated-topk-10-tail.out')
    parser.add_argument('--extended_answers', dest='ext_out_file', type=str, help='File where the extended test set should be stored (json)', required=True)
    return parser.parse_args()
args = parse_args()

def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pl.load(fin)
    return pkl

entity_dict     = load_pickle(args.ent_dict)
relation_dict   = load_pickle(args.rel_dict)
testfile        = load_pickle(args.test_file)

x_label = 'x_' + args.pred + '_fil'
y_label = 'y_' + args.pred + '_fil'
x = testfile[x_label]
y = testfile[y_label]

true_y = np.empty(len(x), dtype = np.int)
true_y.fill(-1);
with open(args.true_out_file) as fin:
    lines = fin.readlines()
    for i, label in enumerate(lines):
        if label == "\n" or int(label) == -1:
            continue
        true_y[i] = int(label)

cnt_answers = 0
queries = []
current_query = None
for i in range(len(true_y)):
    if true_y[i] != -1:
        cnt_answers += 1
        if current_query is None:
            assert(i % 10 == 0)
            entity = int(x[i][0])
            rel = int(x[i][1])
            current_query = {}
            current_query['e'] = entity
            current_query['r'] = rel
            current_query['str_e'] = entity_dict[entity]
            current_query['str_r'] = relation_dict[rel]
            current_query['idx'] = i
            answers_per_query = []
            current_query['answers'] = answers_per_query
            queries.append(current_query)
        answer = int(x[i][2])
        str_answer = entity_dict[answer]
        answers_per_query.append({'answer' : answer, 'str_answer': str_answer, 'cor': int(true_y[i])})
        if len(answers_per_query) == args.topk:
            current_query = None
    else:
        current_query = None

print('N. queries: ', len(queries))
print('Annotated answers: ', cnt_answers)
json.dump(queries, open(args.ext_out_file, 'wt'))