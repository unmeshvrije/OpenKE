import os
import sys
import json
from random import shuffle
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read answer embeddings based on topk and prepare triples(training/test) for LSTM/other RNN models.')
    parser.add_argument('--embfile', dest ='embfile', type = str, help = 'File containing embeddings.')
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = '/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = '/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--combemb', dest ='combine_emb', help = 'Whether to combine embeddings of ent and rel in input', action = 'store_true')
    parser.add_argument('--topk', dest = 'topk', type = int, default = 10)
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--mode', required = True, dest = 'mode', type = str, default = None, help = "train or test")
    parser.add_argument('--ansfile', dest ='ansfile', type = str, help = 'File containing answers as predicted by the model.')
    return parser.parse_args()

args = parse_args()

def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

entity_dict     = load_pickle(args.ent_dict)
relation_dict   = load_pickle(args.rel_dict)

topk = args.topk
db = args.db
result_dir =  args.result_dir + args.db + "/"
os.makedirs(result_dir, exist_ok = True)

# Read the answers file (generated from the test option of the model)
print("Reading answers file...", end=" ")
with open(args.ansfile, "r") as fin:
    res = json.loads(fin.read())
print("DONE")

triples= {}

if args.mode == "train":
    rf_arr = [""]
else:
    rf_arr = ["_fil"] # "_raw,"

ht = ["tail", "head"]
for index in range(len(ht)):
    log = open("./delme"+ht[index], "w")
    log_queries = open("./queries-"+ht[index]+".log", "w")
    for rf in rf_arr:
        unique_pairs = set()
        dup_count = 0
        for i,r in enumerate(tqdm(res)):
            if (r['rel'], r[ht[(index+1)%2]]) not in unique_pairs:
                unique_pairs.add((r['rel'],r[ht[(index+1)%2]]))
                for rank, (e,s,c) in enumerate(zip(\
                r[ht[index]+'_predictions'+rf]['entities'],\
                r[ht[index]+'_predictions'+rf]['scores'], \
                r[ht[index]+'_predictions'+rf]['correctness'])):
                    # log
                    rel = r['rel']
                    ent = r[ht[(index+1)%2]]
                    print("{}, {}, {} ({})".format(entity_dict[ent], relation_dict[rel], entity_dict[e], s), file = log)
            else:
                dup_count += 1

        print(ht[index] + " : " + rf)
        log.close()
        print("# duplicates : ", dup_count)
        print("DONE")

#ans_file = os.path.basename(args.ansfile)
