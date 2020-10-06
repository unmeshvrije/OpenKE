import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.',
    default = '/home/xxx/OpenKE/benchmarks/fb15k237/test2id.txt')
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/xxx/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = "fb15k237")
    return parser.parse_args()

args = parse_args()

log_dir =  args.result_dir + args.db + "/logs/"
os.makedirs(log_dir, exist_ok = True)
queries_file_path = args.test_file

x_test = []
with open(queries_file_path, "r") as fin:
    lines = fin.readlines()
    for line in lines[1:]:
        h = line.split()[0]
        t = line.split()[1]
        r = line.split()[2]
        x_test.append((h,r,t))


def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

entity_dict     = load_pickle(args.ent_dict)
relation_dict   = load_pickle(args.rel_dict)

logfile = log_dir + "test-queries.log"
with open(logfile, "w") as log:
    for index, x in enumerate(tqdm(x_test)):
        head = int(x[0])
        r = int(x[1])
        tail = int(x[2])
        print("{}, {}, {}".format(entity_dict[head], relation_dict[r], entity_dict[tail]), file = log)
