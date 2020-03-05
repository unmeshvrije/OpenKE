import os
import sys
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read answer embeddings based on topk and prepare training for LSTM/other RNN models.')
    parser.add_argument('--embfile', dest ='embfile', type = str, help = 'File containing embeddings.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--ansfile', dest ='ansfile', type = str, help = 'File containing answers as predicted by the model.')
    return parser.parse_args()

args = parse_args()


topk = args.topk
db = args.db
result_dir = "/var/scratch2/uji300/OpenKE-results/" + db + "/"

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

print("Converting the answers into head predictions training...")
training = {}
x_train_head = []
y_train_head = []
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
            features.extend(embeddings[r['rel']])
            features.extend(embeddings[r['tail']])
            features.extend(embeddings[e])

            x_train_head.append(features)
            y_train_head.append(c)
    else:
        dup_count += 1

print("# records (head predictions)    : ", len(x_train_head))
print("# duplicates (head predictions) : ", dup_count)
print("DONE")

training['x_head'] = x_train_head
training['y_head'] = y_train_head

print("Converting the answers into tail predictions training...", end = " ")

dup_count = 0
unique_pairs.clear()
x_train_tail = []
y_train_tail = []

for i,r in enumerate(res):
#    print(i, " :")
    if (r['rel'], r['head']) not in unique_pairs:
        unique_pairs.add((r['rel'],r['head']))
        for rank, (e,s,c) in enumerate(zip(r['tail_predictions']['entity'], r['tail_predictions']['score'], r['tail_predictions']['correctness'])):
            features = []
            #features.append(s)
            #features.append(rank)
            features.extend(embeddings[r['rel']])
            features.extend(embeddings[r['head']])
            features.extend(embeddings[e])

            x_train_tail.append(features)
            y_train_tail.append(c)
    else:
        dup_count += 1

training['x_tail'] = x_train_tail
training['y_tail'] = y_train_tail

print("# records (tail predictions)    : ", len(x_train_tail))
print("# duplicates (tail predictions) : ", dup_count)
print("DONE")

ans_file = os.path.basename(args.ansfile)
lstm_training_file = result_dir + "lstm-" + ans_file.split('.')[0] + ".json"
#"/var/scratch2/uji300/OpenKE-results/fb15k237-test-topk-"+str(topk)+"-filtered"+".json"
with open(lstm_training_file, "w") as fout:
    fout.write(json.dumps(training))

