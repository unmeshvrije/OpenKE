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

triples_features = {}
triples_labels_raw = {}
triples_labels_filtered = {}
x_head = []
y_head = []
y_head_filtered = []
unique_pairs = set()
dup_count = 0
for i,r in enumerate(res):
#    print(i, " :")
    if (r['rel'], r['tail']) not in unique_pairs:
        unique_pairs.add((r['rel'],r['tail']))
        for rank, (e,s,c) in enumerate(zip(r['head_predictions']['entity'], r['head_predictions']['score'], r['head_predictions']['correctness'])):
            features = []
            #features.append(s)
            #features.append(rank)
            features.append(r['tail'])
            features.append(r['rel'])
            features.append(e)
            features.extend(embeddings[r['rel']])
            features.extend(embeddings[r['tail']])
            features.extend(embeddings[e])

            x_head.append(features)
            y_head.append(c)
        if 'correctness_filtered' in r['head_predictions']:
            for cf in r['head_predictions']['correctness_filtered']:
                y_head_filtered.append(cf)
    else:
        dup_count += 1

print("# records (head predictions)    : ", len(x_head))
print("# duplicates (head predictions) : ", dup_count)
print("DONE")

triples_features['x_head'] = x_head
triples_labels_raw['y_head'] = y_head
triples_labels_filtered['y_head_filtered'] = y_head_filtered

print("Converting the answers into tail predictions triples...", end = " ")

dup_count = 0
unique_pairs.clear()
x_tail = []
y_tail = []
y_tail_filtered = []
for i,r in enumerate(res):
#    print(i, " :")
    if (r['rel'], r['head']) not in unique_pairs:
        unique_pairs.add((r['rel'],r['head']))
        for rank, (e,s,c) in enumerate(zip(r['tail_predictions']['entity'], r['tail_predictions']['score'], r['tail_predictions']['correctness'])):
            features = []
            #features.append(s)
            #features.append(rank)
            features.append(r['head'])
            features.append(r['rel'])
            features.append(e)
            features.extend(embeddings[r['rel']])
            features.extend(embeddings[r['head']])
            features.extend(embeddings[e])

            x_tail.append(features)
            y_tail.append(c)
        if 'correctness_filtered' in r['tail_predictions']:
            for cf in r['tail_predictions']['correctness_filtered']:
                y_tail_filtered.append(cf)
    else:
        dup_count += 1

triples_features['x_tail'] = x_tail
triples_labels_raw['y_tail'] = y_tail
triples_labels_filtered['y_tail_filtered'] = y_tail_filtered
print("# records (tail predictions)    : ", len(x_tail))
print("# duplicates (tail predictions) : ", dup_count)
print("DONE")

ans_file = os.path.basename(args.ansfile)

answers_features_file = result_dir + "ans-features-" + ans_file.split('.')[0] + ".pkl"
with open(answers_features_file, "wb") as fout:
    pickle.dump(triples_features, fout, protocol = pickle.HIGHEST_PROTOCOL)

answers_labels_raw_file = result_dir + "ans-labels-raw-" + ans_file.split('.')[0] + ".pkl"
with open(answers_labels_raw_file, "wb") as fout:
    pickle.dump(triples_labels_raw, fout, protocol = pickle.HIGHEST_PROTOCOL)

if len(triples_labels_filtered['y_head_filtered']) != 0:
    answers_labels_filtered_file = result_dir + "ans-labels-filtered-" + ans_file.split('.')[0] + ".pkl"
    with open(answers_labels_filtered_file, "wb") as fout:
        pickle.dump(triples_labels_filtered, fout, protocol = pickle.HIGHEST_PROTOCOL)

