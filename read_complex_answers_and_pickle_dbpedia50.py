import torch
import kge.model
import numpy as np
import random
import json
import os
import sys
import argparse
import pickle
from tqdm import tqdm
# download link for this checkpoint given under results above

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/dbpedia50/misc/dbpedia50-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/dbpedia50/misc/dbpedia50-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test triples.',
    default = '/home/xxx/OpenKE/benchmarks/dbpedia50/test2id.txt')
    parser.add_argument('--trainfile', dest ='train_file', type = str, help = 'File containing train triples.',
    default = '/home/xxx/OpenKE/benchmarks/dbpedia50/train2id.txt')
    parser.add_argument('--validfile', dest ='valid_file', type = str, help = 'File containing valid triples.',
    default = '/home/xxx/OpenKE/benchmarks/dbpedia50/valid2id.txt')
    parser.add_argument('--inputfile', required = True, dest ='input_file', type = str, help = 'File containing sample queries from TransE (train/test).',
    default = "/var/scratch2/xxx/OpenKE-results/dbpedia50/data/dbpedia50-transe-test-topk-10.json"
)
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/xxx/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', type = int, default = 10)
    parser.add_argument('--abstain', dest = 'abstain', default = False, action = 'store_true')
    parser.add_argument('--logans', dest = 'logans', default = False, action = 'store_true')
    parser.add_argument('--db', dest = 'db', type = str, default = "dbpedia50")
    parser.add_argument('--mode', required = True, dest = 'mode', type = str, default = "test", choices = ['train', 'test'])
    parser.add_argument('--model', dest ='model',type = str, default = "transe", help = 'Embedding model name.')
    #parser.add_argument('--pred', dest ='pred', type = str, required = True, choices = ['head', 'tail'], help = 'Prediction type (head/tail)', default='tail')
    return parser.parse_args()

args = parse_args()

def postprocess(x_data, features_dim):
    newdata = np.zeros(shape=(len(x_data), features_dim), dtype=np.float64)
    for i in range(len(x_data)):
        newdata[i]  = x_data[i][5:]
    return newdata

'''
    1. Setup  out and log directories
'''
result_dir =  args.result_dir + args.db + "/out/"
log_dir =  args.result_dir + args.db + "/logs/"
os.makedirs(result_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
model_path = args.result_dir + args.db + "/embeddings/" + args.db + "-complex.pt"
model = kge.model.KgeModel.load_from_checkpoint(model_path)
'''
    2. Load string dictionaries
'''
ent_dict_file = args.ent_dict
rel_dict_file = args.rel_dict
def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl
ent_dict = load_pickle(ent_dict_file)
rel_dict = load_pickle(rel_dict_file)


rev_ent_dict = {v : k for k,v in ent_dict.items()}
rev_rel_dict = {v : k for k,v in rel_dict.items()}
'''
    3. Build dictionary of answers for train and valid and test datasets
'''
def build_ans_dict_from(file_name):
    tail_ans_dict = {}
    head_ans_dict = {}
    with open(file_name, "r") as fin:
        lines = fin.readlines()
        for line in lines[1:]:
            head = int(line.split()[0])
            tail = int(line.split()[1])
            rel  = int(line.split()[2].rstrip())
            if (head,rel) not in tail_ans_dict:
                tail_ans_dict[(head,rel)] = [tail]
            else:
                tail_ans_dict[(head,rel)].append(tail)

            if (tail,rel) not in head_ans_dict:
                head_ans_dict[(tail,rel)] = [head]
            else:
                head_ans_dict[(tail,rel)].append(head)
    return tail_ans_dict, head_ans_dict

train_file = args.train_file
valid_file = args.valid_file
test_file  = args.test_file
tail_ans_in_training, head_ans_in_training  = build_ans_dict_from(train_file)
tail_ans_in_valid, head_ans_in_valid        = build_ans_dict_from(valid_file)
tail_ans_in_test, head_ans_in_test          = build_ans_dict_from(test_file)

# Combine train and valid answer dictionaries
tail_ans_dict = tail_ans_in_training
for k,v in tail_ans_in_valid.items():
    if k in tail_ans_dict.keys():
        tail_ans_dict[k].extend(v)
    else:
        tail_ans_dict[k] = v

head_ans_dict = head_ans_in_training
for k,v, in head_ans_in_valid.items():
    if k in head_ans_dict.keys():
        head_ans_dict[k].extend(v)
    else:
        head_ans_dict[k] = v

def is_present(tup, ans, ans_dict):
    if (tup[0], tup[1]) in ans_dict.keys():
        return ans in ans_dict[tup]
    return False


'''
    4. Read the input file from TransE
    This contains the triples in the same order as used in TransE
'''
with open(args.input_file, "r") as fin:
    records = json.loads(fin.read())


############## Start building conversion dictionary #################
print("start building")
kge_ent_ids = model.dataset.entity_ids()
kge_n_entities = len(kge_ent_ids)
kge_rel_ids = model.dataset.relation_ids()
kge_n_relations = len(kge_rel_ids)

kge_ent_strings = model.dataset.entity_strings(torch.Tensor(list(range(kge_n_entities))).long())
kge_rel_strings = model.dataset.relation_strings(torch.Tensor(list(range(kge_n_relations))).long())

kge_ent_ids = list(range(kge_n_entities))
kge_rel_ids = list(range(kge_n_relations))
#for i in range(5):
#    print(i , " : id = " , kge_ent_ids[i] , "str = " , kge_ent_strings[i])

kge_rel_dict = {x:y for x,y in zip(kge_rel_strings, kge_rel_ids)}
kge_ent_dict = {x:y for x,y in zip(kge_ent_strings, kge_ent_ids)}

#for i, (k,v) in enumerate(kge_ent_dict.items()):
#    print(i , " : ", k , "=>", v)
#    if i == 5:
#        break
print("end building")
############## End   building conversion dictionary #################


query_string = open("dbpedia-test-queries-string.log", "w")

'''
    5. Build the head,rel and tail,rel arrays for prediction
'''
heads_for_unique_tail_queries = []
rels_for_unique_tail_queries  = []
tails_for_unique_head_queries = []
rels_for_unique_head_queries  = []
unique_pairs_hr = set()
unique_pairs_rt = set()
#kge_to_openke_ent_dict = {}
#kge_to_openke_rel_dict = {}

for r in records:
    head = kge_ent_dict[ent_dict[r['head']]]
    #kge_to_openke_ent_dict[head] = r['head']
    tail = kge_ent_dict[ent_dict[r['tail']]]
    #kge_to_openke_ent_dict[tail] = r['tail']
    rel  = kge_rel_dict[rel_dict[r['rel']]]
    #kge_to_openke_rel_dict[rel] = r['rel']

    if (head, rel) not in unique_pairs_hr:
        unique_pairs_hr.add((head, rel))
        heads_for_unique_tail_queries.append(head)
        rels_for_unique_tail_queries.append(rel)
    if (tail, rel) not in unique_pairs_rt:
        unique_pairs_rt.add((tail,rel))
        tails_for_unique_head_queries.append(tail)
        rels_for_unique_head_queries.append(rel)

    #heads.append(r['head'])
    #tails.append(r['tail'])
    #rels.append(r['rel'])
    h_str = model.dataset.entity_strings(r['head'])
    t_str = model.dataset.entity_strings(r['tail'])
    r_str = model.dataset.relation_strings(r['rel'])
    if h_str is not None and t_str is not None and r_str is not None:
        print(h_str + "," + r_str + " => "+ t_str, file = query_string)
        print(ent_dict[head] + "," + rel_dict[rel] + " -->* " + ent_dict[tail], file = query_string)
        print(kge_ent_dict[ent_dict[head]] , "," , kge_rel_dict[rel_dict[rel]] , " ==# " , kge_ent_dict[ent_dict[tail]], file = query_string)
        print("_ " * 40, file = query_string)
    else:
        print("some IDs not present")

query_string.close()

E = model._entity_embedder._embeddings_all()
R = model._relation_embedder._embeddings_all()
ent_dim = len(E[0].tolist())
rel_dim = len(R[0].tolist())
suf = ""
if args.mode == "test":
    suf = "_fil"

'''
    5. Tail predictions
'''
topk = args.topk
s = torch.Tensor(heads_for_unique_tail_queries).long()              # subject indexes
p = torch.Tensor(rels_for_unique_tail_queries).long()               # relation indexes
scores = model.score_sp(s, p)                                       # scores of all objects for (s,p,?)
o = torch.argsort(scores, dim=-1, descending = True)                # index of highest-scoring objects
scores_sp_sorted, indices = torch.sort(scores, dim = -1, descending = True)  # sorted scores

triples = {}

X_TAIL = []
Y_TAIL = []

if args.logans:
    log = open("complex-answers-tail-"+args.mode+".log", "w")

for index in tqdm(range(0, len(heads_for_unique_tail_queries))):
    head = s[index].item()
    rel  = p[index].item()
    h = model.dataset.entity_strings(s[index])
    r = model.dataset.relation_strings(p[index])
    filtered_answers = []
    if args.mode == "train":
        for oi in o[index][:topk]:
            filtered_answers.append(oi.item())
    else:# if mode == test, then skip answers from train+valid
        if (head, rel) not in tail_ans_dict.keys():
            for oi in o[index][:topk]:
                filtered_answers.append(oi.item())
        else:
            for oi in o[index]:
                if oi.item() not in tail_ans_dict[(head,rel)]:
                    filtered_answers.append(oi.item())
                    if len(filtered_answers) == topk:
                        break

    assert(len(filtered_answers) == topk)

    for i in range(topk):
        ans = filtered_answers[i]
        features = []
        h_openke = rev_ent_dict[str(h)]
        r_openke = rev_rel_dict[str(r)]
        features.append(h_openke)
        features.append(r_openke)
        try:
            ans_openke = rev_ent_dict[str(model.dataset.entity_strings(ans))]
        except KeyError:
            print("Answer out of database found")
            ans_openke = 0
        features.append(ans_openke)
        features.append(scores_sp_sorted[index][ans].item())
        features.append(i+1)
        features.extend(R[rel].tolist()) #cpu().data.numpy())
        features.extend(E[head].tolist())#cpu().data.numpy())
        features.extend(E[ans].tolist())#cpu().data.numpy())
        X_TAIL.append(features)

        if args.mode == "test":
            correctness = is_present((h_openke,r_openke), ans_openke, tail_ans_in_test)
        else:
            correctness = is_present((h_openke,r_openke), ans_openke, tail_ans_dict)

        if correctness:
            print("FOUND!!!")

        Y_TAIL.append(int(correctness))

        if args.logans:
            t = model.dataset.entity_strings(filtered_answers[i])
            print(h, ",", r, ",", t, ";", head, ",", rel, ",", filtered_answers[i], sep='', file = log)
    if args.logans:
        print("_ " * 40, file = log)


triples['x_tail' + suf] = postprocess(X_TAIL, (ent_dim*2) + rel_dim)
triples['y_tail' + suf] = Y_TAIL
if args.mode == "test":
    triples['x_tail' + "_raw"] = postprocess(X_TAIL, (ent_dim*2) + rel_dim)
    triples['y_tail' + "_raw"] = Y_TAIL


if args.logans:
    log.close()

'''
    6. Head predictions
'''
o = torch.Tensor(tails_for_unique_head_queries).long()             # object indexes
p = torch.Tensor(rels_for_unique_head_queries).long()             # relation indexes
scores = model.score_po(p, o)                # scores of all subjects for (?,p,o)
s = torch.argsort(scores, dim=-1, descending = True)             # index of highest-scoring objects
#o = torch.argmax(scores, dim=-1)             # index of highest-scoring objects
scores_po_sorted, indices = torch.sort(scores, dim = -1, descending = True)

X_HEAD = []
Y_HEAD = []

if args.logans:
    log = open("complex-answers-head-"+args.mode+".log", "w")

for index in tqdm(range(0, len(tails_for_unique_head_queries))):
    tail = o[index].item()
    rel  = p[index].item()
    t = model.dataset.entity_strings(o[index])
    r = model.dataset.relation_strings(p[index])

    filtered_answers = []

    if args.mode == "train":
        for si in s[index][:topk]:
            filtered_answers.append(si.item())
    else:
        if (tail, rel) not in head_ans_dict.keys():
            for si in s[index][:topk]:
                filtered_answers.append(si.item())
        else:
            for si in s[index]:
                if si.item() not in head_ans_dict[(tail,rel)]:
                    filtered_answers.append(si.item())
                    if len(filtered_answers) == topk:
                        break

    assert(len(filtered_answers) == topk)

    for i in range(topk):
        ans = filtered_answers[i]
        features = []
        t_openke = rev_ent_dict[str(t)]
        r_openke = rev_rel_dict[str(r)]
        features.append(t_openke)
        features.append(r_openke)
        try:
            ans_openke = rev_ent_dict[str(model.dataset.entity_strings(ans))]
        except:
            print("Answer out of database found!")
            ans_openke = 0
        features.append(ans_openke)
        features.append(scores_po_sorted[index][ans].item())
        features.append(i+1)
        features.extend(R[rel].tolist())#cpu().data.numpy())
        features.extend(E[tail].tolist())#cpu().data.numpy())
        features.extend(E[ans].tolist())#cpu().data.numpy())
        X_HEAD.append(features)

        if args.mode == "test":
            correctness = is_present((t_openke,r_openke), ans_openke, head_ans_in_test)
        else:
            correctness = is_present((t_openke,r_openke), ans_openke, head_ans_dict)

        Y_HEAD.append(int(correctness))

        if args.logans:
            h = model.dataset.entity_strings(filtered_answers[i])
            print(h, ",", r, ",", t, ";", filtered_answers[i], ",", rel, ",", tail, sep='', file = log)
    if args.logans:
        print("*" * 80, file = log)

triples['x_head' + suf] = postprocess(X_HEAD, (ent_dim*2) + rel_dim)
triples['y_head' + suf] = Y_HEAD
if args.mode == "test":
    triples['x_head' + "_raw"] = postprocess(X_HEAD, (ent_dim*2) + rel_dim)
    triples['y_head' + "_raw"] = Y_HEAD

if args.logans:
    log.close()

lstm_features_file = args.input_file
lstm_features_file = lstm_features_file.replace("transe", "complex")
lstm_features_file = lstm_features_file.replace("json", "pkl")
print("new file Name : ", lstm_features_file)
with open(lstm_features_file, "wb") as fout:
    pickle.dump(triples, fout, protocol = pickle.HIGHEST_PROTOCOL)
