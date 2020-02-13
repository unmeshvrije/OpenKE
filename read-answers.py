import sys
import json
from random import shuffle

print("Reading embeddings file...", end=" ")
# Read embedding file
with open("/home/uji300/OpenKE/result/fb15k237-transe.json", "r") as fin:
    params = json.loads(fin.read())
embeddings = params['ent_embeddings.weight']
rel_embeddings = params['rel_embeddings.weight']

print("DONE")
print("Reading answers file...", end=" ")

topk = sys.argv[1]
db = "FB15K237"
# Read the answers file (generated from the test option of the model)
answer_file = sys.argv[2] #db+"-results-scores-"+str(topk)+".json"
with open(answer_file, "r") as fin:
    res = json.loads(fin.read())

# It is very important to shuffle otherwise all triples with same relations are together
shuffle(res)

print("DONE")
print("Converting the answers into head predictions training...", end = " ")

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
            #features.append(r['head'])
            #features.append(r['rel'])
            #features.append(r['tail'])
            #features.append(s)
            #features.append(rank)
            features.extend(embeddings[r['rel']])
            features.extend(embeddings[r['tail']])
            features.extend(embeddings[e])

            x_train_head.append(features)
            y_train_head.append(c)
    else:
        dup_count += 1

print("For head prediction training data...")
print("duplicate count : ", dup_count)
print("# of records : ", len(x_train_head))

dup_count = 0
unique_pairs.clear()
training['x_head'] = x_train_head
training['y_head'] = y_train_head

print("DONE")
print("Converting the answers into tail predictions training...", end = " ")

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
print("# of records = ", len(x_train_tail))
print("DONE")

print("for tail prediction training: ")
print("duplication count : ", dup_count)

lstm_training_file = "/var/scratch2/uji300/OpenKE-results/fb15k237-test-topk-"+str(topk)+"-filtered"+".json"
with open(lstm_training_file, "w") as fout:
    fout.write(json.dumps(training))

