import sys
import pickle
import numpy as np
from support.dataset_dbpedia50 import Dataset_dbpedia50

def convert(indata):
    dataset = Dataset_dbpedia50()
    x_head_fil = indata['x_head']
    n_features = len(x_head_fil[0]) - 5 + 1
    i = 0
    out_training_head = []
    while i < len(x_head_fil):
        X = np.zeros(shape=(10, n_features), dtype=np.float)
        Y = np.zeros(shape=(10), dtype=np.int)
        for j in range(10):
            emb = x_head_fil[i]
            meta = emb[:5]
            ent = meta[0]
            rel = meta[1]
            ans = meta[2]
            score = meta[3]
            emb = emb[5:]
            X[j] = np.concatenate([[score], emb])
            exists = dataset.exists_htr(ans, ent, rel)
            if exists:
                Y[j] = 1
            else:
                Y[j] = 0
            i += 1
        data_entry = {}
        data_entry['X'] = X
        data_entry['Y'] = Y
        out_training_head.append(data_entry)

    x_tail_fil = indata['x_tail']
    i = 0
    out_training_tail = []
    while i < len(x_tail_fil):
        X = np.zeros(shape=(10, n_features), dtype=np.float)
        Y = np.zeros(shape=(10), dtype=np.int)
        for j in range(10):
            emb = x_tail_fil[i]
            meta = emb[:5]
            ent = meta[0]
            rel = meta[1]
            ans = meta[2]
            score = meta[3]
            emb = emb[5:]
            X[j] = np.concatenate([[score], emb])
            exists = dataset.exists_htr(ent, ans, rel)
            if exists:
                Y[j] = 1
            else:
                Y[j] = 0
            i += 1
        data_entry = {}
        data_entry['X'] = X
        data_entry['Y'] = Y
        out_training_tail.append(data_entry)
    return out_training_head, out_training_tail

old_file = sys.argv[1]
indata = pickle.load(open(old_file, 'rb'))
out_training_head, out_training_tail = convert(indata)
new_answers_file = sys.argv[2]
new_file_head = new_answers_file + 'head-fil.pkl'
with open(new_file_head, 'wb') as fout:
    pickle.dump(out_training_head, fout)
new_file_tail = new_answers_file + 'tail-fil.pkl'
with open(new_file_tail, 'wb') as fout:
    pickle.dump(out_training_tail, fout)
