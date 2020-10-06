import os
import sys
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--true-out', dest ='true_out_file', type = str, help = 'File containing the true /expected answers.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-annotated-topk-10-tail.out')
    parser.add_argument('--lstm-out', dest ='lstm_out_file', type = str, help = 'File containing the output of lstm classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-transe-training-topk-10-tail-model-lstm-units-100-dropout-0.2.out')
    parser.add_argument('--mlp-out', dest ='mlp_out_file', type = str, help = 'File containing the output of mlp classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-transe-training-topk-10-tail-model-mlp-units-100-dropout-0.2.out')
    parser.add_argument('--path-out', dest ='path_out_file', type = str, help = 'File containing the output of path classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/path-classifier-tail.out')
    parser.add_argument('--sub-out', dest ='sub_out_file', type = str, help = 'File containing the output of sub classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-transe-subgraphs-tau-10-tail.out')
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/xxx/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--model', dest ='model',type = str, default = "transe", help = 'Embedding model name.')
    parser.add_argument('--pred', dest ='pred', type = str, required = True, choices = ['head', 'tail'], help = 'Prediction type (head/tail)', default='tail')
    return parser.parse_args()

args = parse_args()

result_dir =  args.result_dir + args.db + "/out/"
log_dir =  args.result_dir + args.db + "/logs/"
os.makedirs(result_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
queries_file_path = args.test_file

def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

entity_dict     = load_pickle(args.ent_dict)
relation_dict   = load_pickle(args.rel_dict)
lstm_results    = load_pickle(args.lstm_out_file)
mlp_results     = load_pickle(args.mlp_out_file)
path_results    = load_pickle(args.path_out_file)
sub_results     = load_pickle(args.sub_out_file)

lstm_y = np.array(lstm_results['fil']['predicted_y'])
mlp_y  = np.array(mlp_results['fil']['predicted_y'])
path_y = np.array(path_results['fil']['predicted_y'])
sub_y  = np.array(sub_results['fil']['predicted_y'])

len_y  = len(lstm_y)
#cnt_ones = int(len_y * 0.3)
#cnt_zeros = len_y - cnt_ones
#true_y = np.random.permutation([1] * cnt_ones + [0] * cnt_zeros)

indexes = []
true_y = np.empty(len_y, dtype = np.int)
true_y.fill(-1);
with open(args.true_out_file) as fin:
    lines = fin.readlines()
    for i, label in enumerate(lines):
        if label == "\n" or int(label) == -1:
            continue
        indexes.append(i)
        true_y[i] = int(label)

indexes_annotated = np.where(true_y != -1)

vs = np.vstack((lstm_y, mlp_y, path_y, sub_y))
sums = np.sum(vs, axis = 0)
max_voting_y = (sums > 2).astype(int)
min_voting_y = (sums > 0).astype(int)

test_queries = load_pickle(queries_file_path)
x_test_raw = np.array(test_queries['x_' + args.pred + "_raw"])
x_test_fil = np.array(test_queries['x_' + args.pred + "_fil"])

lstm_annotated = lstm_y[indexes_annotated]
mlp_annotated  = mlp_y[indexes_annotated]
sub_annotated  = sub_y[indexes_annotated]
path_annotated = path_y[indexes_annotated]
maxv_annotated = max_voting_y[indexes_annotated]
minv_annotated = min_voting_y[indexes_annotated]
true_annotated = true_y[indexes_annotated]

def print_results(y_true, y_predicted):
    conf = confusion_matrix(y_true, y_predicted)
    print(conf)
    result = classification_report(y_true, y_predicted)
    print(result)

print_results(true_annotated, lstm_annotated)
print_results(true_annotated, path_annotated)
print_results(true_annotated, maxv_annotated)
print_results(true_annotated, minv_annotated)

logfile = log_dir + args.pred + "-ensembled.log"
with open(logfile, "w") as log:
    for index, x in enumerate(tqdm(x_test_fil)):
        if index not in indexes_annotated[0]:
            continue
        e = int(x[0])
        r = int(x[1])
        a = int(x[2])
        head = e
        tail = a
        if args.pred == "head":
            head = a
            tail = e
        print("{}, {}, {} => (LSTM) {} :  (MLP) {} : (PATH) {} : (SUB) {} : (maxV) {} : (REAL) {}".format(entity_dict[head], relation_dict[r], entity_dict[tail], lstm_y[index], mlp_y[index], path_y[index], sub_y[index], max_voting_y[index], true_y[index]), file = log)
        if (index+1) % args.topk == 0:
            print("*" * 80, file = log)
