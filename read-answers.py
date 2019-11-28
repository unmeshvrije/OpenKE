import json

print("Reading embeddings file...", end=" ")
# Read embedding file
with open("/home/uji300/OpenKE/result/fb15k237-transe.json", "r") as fin:
    params = json.loads(fin.read())
embeddings = params['ent_embeddings.weight']
rel_embeddings = params['rel_embeddings.weight']

print("DONE")
print("Reading answers file...", end=" ")

# Read the anwers file
with open("/home/uji300/OpenKE/fb15k237-top10-answers.json", "r") as fin:
    res = json.loads(fin.read())

print("DONE")
print("Converting the answers into head predictions training...", end = " ")

training = {}
x_train_head = []
y_train_head = []
for i,r in enumerate(res):
#    print(i, " :")
    for rank, (e,s,c) in enumerate(zip(r['head_predictions']['entity'], r['head_predictions']['score'], r['head_predictions']['correctness'])):
        features = []
        #features.append(r['head'])
        #features.append(r['rel'])
        #features.append(r['tail'])
        features.append(s)
        features.append(rank)
        features.extend(embeddings[r['rel']])
        features.extend(embeddings[r['tail']])
        features.extend(embeddings[e])

        x_train_head.append(features)
        y_train_head.append(c)

training['x_head'] = x_train_head
training['y_head'] = y_train_head

print("DONE")
print("Converting the answers into tail predictions training...", end = " ")

x_train_tail = []
y_train_tail = []
for i,r in enumerate(res):
#    print(i, " :")
    for rank, (e,s,c) in enumerate(zip(r['tail_predictions']['entity'], r['tail_predictions']['score'], r['tail_predictions']['correctness'])):
        features = []
        features.append(s)
        features.append(rank)
        features.extend(embeddings[r['rel']])
        features.extend(embeddings[r['head']])
        features.extend(embeddings[e])

        x_train_tail.append(features)
        y_train_tail.append(c)

training['x_tail'] = x_train_tail
training['y_tail'] = y_train_tail
print("DONE")

with open("/var/scratch2/uji300/OpenKE-results/fb15k237-trainingv2.json", "w") as fout:
    fout.write(json.dumps(training))

