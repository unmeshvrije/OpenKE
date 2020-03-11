import os
import sys
import json
from random import shuffle
import random
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read answer embeddings based on topk and prepare triples(training/test) for LSTM/other RNN models.')
    parser.add_argument('--embfile', dest ='embfile', type = str, help = 'File containing embeddings.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--ansfile', dest ='ansfile', type = str, help = 'File containing answers as predicted by the model.')
    return parser.parse_args()

args = parse_args()


topk = args.topk
db = args.db
result_dir =  args.result_dir + args.db + "/"
os.makedirs(result_dir, exist_ok = True)

# Read embedding file
print("Reading embeddings file...", end=" ")
with open(args.embfile, "r") as fin:
    params = json.loads(fin.read())
embeddings = params['ent_embeddings.weight']
rel_embeddings = params['rel_embeddings.weight']
print("DONE")

# Read the answers file (generated from the test option of the model)
print("Reading answers file...", end=" ")
with open(args.ansfile, "r") as fin:
    res = json.loads(fin.read())
print("DONE")

triples= {}

ht = ["head", "tail"]
for index in range(len(ht)):
    for rf in ["raw", "fil"]:
        x_head = []
        y_head = []
        unique_pairs = set()
        dup_count = 0
        for i,r in enumerate(res):
            #print(i, " :")
            if (r['rel'], r[ht[(index+1)%2]]) not in unique_pairs:
                unique_pairs.add((r['rel'],r[ht[(index+1)%2]]))
                for rank, (e,s,c) in enumerate(zip(\
                r[ht[index]+'_predictions_'+rf]['entities'],\
                r[ht[index]+'_predictions_'+rf]['scores'], \
                r[ht[index]+'_predictions_'+rf]['correctness'])):
                    features = []
                    features.append(r[ht[(index+1)%2]])
                    features.append(r['rel'])
                    features.append(e)
                    features.append(s)
                    features.append(rank)
                    features.extend(embeddings[r['rel']])
                    features.extend(embeddings[r[ht[(index+1)%2]]])
                    features.extend(embeddings[e])

                    x_head.append(features)
                    y_head.append(c)
            else:
                dup_count += 1
        # add the x_head and y_head to dictionary
        triples['x_' + ht[index] + rf] = x_head
        triples['y_' + ht[index] + rf] = y_head

        print(ht[index] + " : " + rf)
        print("# records    : ", len(x_head))
        print("# duplicates : ", dup_count)
        print("DONE")

ans_file = os.path.basename(args.ansfile)

answers_features_file = result_dir + "ans-features-" + ans_file.split('.')[0] + ".pkl"
with open(answers_features_file, "wb") as fout:
    pickle.dump(triples, fout, protocol = pickle.HIGHEST_PROTOCOL)
