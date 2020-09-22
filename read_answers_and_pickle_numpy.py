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
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--combemb', dest ='combine_emb', help = 'Whether to combine embeddings of ent and rel in input', action = 'store_true')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--mode', required = True, dest = 'mode', type = str, default = None, help = "train or test")
    parser.add_argument('--ansfile', dest ='ansfile', type = str, help = 'File containing answers as predicted by the model.')
    return parser.parse_args()

args = parse_args()


topk = args.topk
db = args.db
result_dir =  args.result_dir + args.db + "/"
os.makedirs(result_dir, exist_ok = True)
combine_embeddings = args.combine_emb

# Read embedding file
print("Reading embeddings file...", end=" ")
with open(args.embfile, "r") as fin:
    params = json.loads(fin.read())

embeddings = params['ent_embeddings.weight']
rel_embeddings = params['rel_embeddings.weight']
dim = len(embeddings[0])
if "rotate" in args.embfile:
    dim = int(dim/2)
print("dim = ", dim)
print("DONE")

# Read the answers file (generated from the test option of the model)
print("Reading answers file...", end=" ")
with open(args.ansfile, "r") as fin:
    res = json.loads(fin.read())
print("DONE")

def postprocess(x_data, features_dim):
    newdata = np.zeros(shape=(len(x_data), features_dim), dtype=np.float64)
    for i in range(len(x_data)):
        newdata[i]  = x_data[i][5:]
    return newdata

triples= {}

if args.mode == "train":
    rf_arr = [""]
else:
    rf_arr = ["_raw", "_fil"]

ht = ["head", "tail"]
for index in range(len(ht)):
    for rf in rf_arr:
        x_head = []
        y_head = []
        unique_pairs = set()
        dup_count = 0
        for i,r in enumerate(tqdm(res)):
            if (r['rel'], r[ht[(index+1)%2]]) not in unique_pairs:
                unique_pairs.add((r['rel'],r[ht[(index+1)%2]]))
                for rank, (e,s,c) in enumerate(zip(\
                r[ht[index]+'_predictions'+rf]['entities'],\
                r[ht[index]+'_predictions'+rf]['scores'], \
                r[ht[index]+'_predictions'+rf]['correctness'])):
                    features = []
                    features.append(r[ht[(index+1)%2]])
                    features.append(r['rel'])
                    features.append(e)
                    features.append(s)
                    features.append(rank)

                    features.extend(rel_embeddings[r['rel']])
                    features.extend(embeddings[r[ht[(index+1)%2]]][:dim])
                    features.extend(embeddings[e][:dim])

                    x_head.append(features)
                    y_head.append(c)
            else:
                dup_count += 1
        # add the x_head and y_head to dictionary
        triples['x_' + ht[index] + rf] = postprocess(x_head, dim*3)
        triples['y_' + ht[index] + rf] = y_head

        print(ht[index] + " : " + rf)
        print("# records    : ", len(x_head))
        print("# duplicates : ", dup_count)
        print("DONE")

ans_file = os.path.basename(args.ansfile)

query_features_combined = ""
if combine_embeddings:
    query_features_combined += "-combined"

answers_features_file = result_dir + "data/" + ans_file.split('.')[0] + query_features_combined + ".pkl"
with open(answers_features_file, "wb") as fout:
    pickle.dump(triples, fout, protocol = pickle.HIGHEST_PROTOCOL)
