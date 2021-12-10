import argparse
import json
from support.utils import *
from sklearn.model_selection import train_test_split
import os

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest ='db', type = str, default = "fb15k237", choices=['fb15k237', 'dbpedia50'])
    return parser.parse_args()

args = parse_args()
topk = 10

# Load the gold standard
gold_dir = args.result_dir + '/' + args.db + '/annotations/'
gold_filename = get_filename_gold(args.db, topk)
with open(gold_dir + gold_filename, 'rt') as fin:
    gold_annotations = json.load(fin)

gold_valid_filename = get_filename_gold(args.db, topk, '-valid')
gold_test_filename = get_filename_gold(args.db, topk, '-test')

if os.path.exists(gold_dir + '/' + gold_valid_filename):
    # Copy all the new annotations as test annotations
    with open(gold_dir + '/' + gold_test_filename, 'rt') as fin:
        gold_test_annotations = json.load(fin)
    with open(gold_dir + '/' + gold_valid_filename, 'rt') as fin:
        gold_valid_annotations = json.load(fin)
    os.rename(gold_dir + '/' + gold_test_filename, gold_dir + '/' + gold_test_filename + '-backup')
    assert(len(gold_test_annotations) < len(gold_annotations))
    new_test_annotations = gold_test_annotations
    for key, a in gold_annotations.items():
        if key not in gold_valid_annotations and key not in gold_test_annotations:
            new_test_annotations[key] = a
    with open(gold_dir + '/' + gold_test_filename, 'wt') as fout:
        json.dump(new_test_annotations, fout)
else:
    # split
    array_gold_annotations = [ a for key, a in gold_annotations.items() if a['valid_annotations'] == True ]
    array_gold_test_annotations, array_gold_valid_annotations = train_test_split(array_gold_annotations, test_size=50)
    gold_test_annotations = {}
    for a in array_gold_test_annotations:
        gold_test_annotations[a['query']['id']] = a
    gold_valid_annotations = {}
    for a in array_gold_valid_annotations:
        gold_valid_annotations[a['query']['id']] = a
    with open(gold_dir + '/' + gold_test_filename, 'wt') as fout:
        json.dump(gold_test_annotations, fout)
    with open(gold_dir + '/' + gold_valid_filename, 'wt') as fout:
        json.dump(gold_valid_annotations, fout)