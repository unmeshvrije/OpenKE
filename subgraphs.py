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
from enum import Enum
SUBTYPE = Enum('SUBTYPE', 'SPO POS')
sub_type_to_string = {SUBTYPE.SPO: "spo", SUBTYPE.POS: "pos"}


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

class Subgraph():
    def __init__(self, sid, st, sent, srel, ssize, entities):
        self.subType = st
        self.subId   = sid
        self.ent     = sent
        self.rel     = srel
        self.size    = ssize
        self.entities= copy.deepcopy(entities)



class SubgraphFactory():
    def __init__(self, db, min_subgraph_size, triples, ent_embeddings, sub_type):

        self.db = db
        self.min_subgraph_size = min_subgraph_size
        self.E = ent_embeddings
        self.triples = triples
        self.sub_type = sub_type

        self.subgraphs = []
        self.avg_embeddings = []
        self.var_embeddings = []

    def add_subgraphs(self, st, sent, srel, ssize, entities):
        subentities = copy.deepcopy(entities)
        sub = Subgraph(len(self.subgraphs), st, sent, srel, ssize, subentities)
        self.subgraphs.append(sub)

    def get_Nsubgraphs(self):
        return len(self.subgraphs)
    def save(self, outdir, emb_model_str, protocol=pickle.HIGHEST_PROTOCOL):
        filename = outdir + db + "-" + emb_model_str + "-" + sub_type_to_string[self.sub_type] + "-subgraphs-tau-" + str(self.min_subgraph_size) + ".pkl"
        with open(filename, 'wb') as fout:
            pickle.dump(self, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            subgraphs = pickle.load(fin)
        return subgraphs

    def make_subgraphs(self):

        E = self.E
        min_subgraph_size = self.min_subgraph_size
        sub_type = self.sub_type

        similar_entities = []
        current = np.zeros(len(E[0]), dtype=np.float64)
        count = 0
        prevo = -1
        prevp = -1
        subgraph_logfile="subgraphs-test.log"
        file_data = ""

        if sub_type == SUBTYPE.SPO:
            sorted_triples = sorted(self.triples, key = lambda l : (l[2], l[0]))
        elif sub_type == SUBTYPE.POS:
            sorted_triples = sorted(self.triples, key = lambda l : (l[2], l[1]))

        cntTriples = len(sorted_triples)
        for i, triple in enumerate(sorted_triples):
            sub = triple[0]
            obj = triple[1]
            rel = triple[2]
            ent = -1
            other_ent = -1
            ER = None
            if sub_type == SUBTYPE.POS:
                ent = obj
                other_ent = sub
            else:
                ent = sub
                other_ent = obj

            if ent != prevo or rel != prevp:
                if count > min_subgraph_size:
                    mean = current/count
                    self.avg_embeddings.append(mean)
                    columnsSquareDiff = np.zeros(len(E[0]), dtype=np.float64)
                    for se in similar_entities:
                        columnsSquareDiff += (E[se] - mean) * (E[se] - mean)
                    if count > 2:
                        columnsSquareDiff /= (count-1)
                    else:
                        columnsSquareDiff = mean
                    self.var_embeddings.append(columnsSquareDiff)
                    self.add_subgraphs(sub_type, prevo, prevp, count, similar_entities)
                    file_data += str(count) + " entities in the subgraph"
                count = 0
                prevo = ent
                prevp = rel
                current.fill(0.0)
                similar_entities.clear()
            count += 1
            current += E[other_ent]
            similar_entities.append(other_ent)
        # After looping over all triples, add remaining entities to a subgraph
        if count > min_subgraph_size:
            mean = current / count
            self.avg_embeddings.append(mean)
            columnsSquareDiff = np.zeros(self.args.ncomp, dtype=np.float64)
            for se in similar_entities:
                columnsSquareDiff += (E[se] - mean) * (E[se] - mean)
            if count > 2:
                columnsSquareDiff /= (count-1)
            else:
                columnsSquareDiff = mean
            self.var_embeddings.append(columnsSquareDiff)
            self.add_subgraphs(subType, prevo, prevp, count, similar_entities)
        print ("# of subgraphs : " , self.get_Nsubgraphs())
        file_data +=  "# of subgraphs : " + str(self.get_Nsubgraphs())
        with open(subgraph_logfile, "w") as fout:
            fout.write(file_data)


triples = read_triples(args.infile)
E, R = read_embeddings(args.embfile)
sub_factory_spo = SubgraphFactory(args.db, int(args.ms), triples, E, SUBTYPE.SPO)
sub_factory_spo.make_subgraphs()
sub_factory_spo.save(result_dir, "transe")
sub_factory_pos = SubgraphFactory(args.db, int(args.ms), triples, E, SUBTYPE.POS)
sub_factory_pos.make_subgraphs()
sub_factory_pos.save(result_dir, "transe")
