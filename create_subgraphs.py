import copy
import os
import sys
import json
from random import shuffle
import random
import argparse
import pickle
import numpy as np
from collections import defaultdict
sub_type_to_string = {SUBTYPE.SPO: "spo", SUBTYPE.POS: "pos"}

from subgraphs import Subgraph
from subgraphs import SubgraphFactory
from subgraphs import SUBTYPE

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read embeddings and prepare subgraphs.')
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--embfile', required = True, dest ='embfile', type = str, help = 'File containing embeddings.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--infile', dest ='infile', type = str, help = 'File containing training triples.', default = "/home/uji300/OpenKE/benchmarks/fb15k237/train2id.txt")
    parser.add_argument('--ms', dest = 'ms', type = int, default = 10, help = 'Minimum subgraph Size')
    return parser.parse_args()

args = parse_args()

ms = args.ms
db = args.db
result_dir = args.result_dir + db + "/"
os.makedirs(result_dir, exist_ok = True)

def read_embeddings(filename):
    with open(filename, "r") as fin:
        params = json.loads(fin.read())
    E = params['ent_embeddings.weight']
    R = params['rel_embeddings.weight']
    return E, R

def read_triples(filename):
    triples = []
    with open (filename, "r") as fin:
        lines = fin.readlines()
    for line in lines[1:]:
        h = int(line.split()[0])
        t = int(line.split()[1])
        r = int(line.split()[2])
        triples.append((h,t,r))

    return triples



triples = read_triples(args.infile)
E, R = read_embeddings(args.embfile)
sub_factory_spo = SubgraphFactory(args.db, int(args.ms), triples, E, SUBTYPE.SPO)
sub_factory_spo.make_subgraphs()
sub_factory_spo.save(result_dir, "transe")
sub_factory_pos = SubgraphFactory(args.db, int(args.ms), triples, E, SUBTYPE.POS)
sub_factory_pos.make_subgraphs()
sub_factory_pos.save(result_dir, "transe")
