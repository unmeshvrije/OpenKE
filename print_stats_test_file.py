import argparse
import json
from support.utils import *
from support.dataset_fb15k237 import Dataset_FB15k237

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237'])
    return parser.parse_args()

args = parse_args()

gold_dir = args.result_dir + '/' + args.db + '/annotations/'
gold_filename = get_filename_gold(args.db, 10, '-test')
with open(gold_dir + gold_filename, 'rt') as fin:
    gold_annotations = json.load(fin)

# Load dataset
dataset = Dataset_FB15k237()

rels = set()
for _, annotation in gold_annotations.items():
    rel = annotation['query']['rel']
    rels.add(rel)

for rel in rels:
    txt_rel = dataset.get_relation_text(rel)
    print(txt_rel)