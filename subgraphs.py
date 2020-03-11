import pickle
import copy
import numpy as np
from enum import Enum
SUBTYPE = Enum('SUBTYPE', 'SPO POS')
sub_type_to_string = {SUBTYPE.SPO: "spo", SUBTYPE.POS: "pos"}

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
        self.data = {}
        self.data['subType'] = st
        self.data['subId']   = sid
        self.data['ent']     = sent
        self.data['rel']     = srel
        self.data['size']    = ssize
        self.data['entities']= copy.deepcopy(entities)

class SubgraphFactory():
    def __init__(self, db, min_subgraph_size, triples, ent_embeddings):
        self.db = db
        self.min_subgraph_size = min_subgraph_size
        self.E = ent_embeddings
        self.triples = triples

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
        filename = outdir + self.db + "-" + emb_model_str + "-subgraphs-tau-" + str(self.min_subgraph_size) + ".pkl"
        with open(filename, 'wb') as fout:
            pickle.dump(self.subgraphs, fout, protocol=protocol)

        filename = outdir + self.db + "-" + emb_model_str + "-avgemb-tau-" + str(self.min_subgraph_size) + ".pkl"
        with open(filename, 'wb') as fout:
            pickle.dump(self.avg_embeddings, fout, protocol=protocol)

        filename = outdir + self.db + "-" + emb_model_str + "-varemb-tau-" + str(self.min_subgraph_size) + ".pkl"
        with open(filename, 'wb') as fout:
            pickle.dump(self.var_embeddings, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            subgraphs = pickle.load(fin)
        return subgraphs

    # sub_type can be SPO or POS
    def make_subgraphs_per_type(self,sub_type):

        E = self.E
        min_subgraph_size = self.min_subgraph_size
        if sub_type == SUBTYPE.SPO:
            sorted_triples = sorted(self.triples, key = lambda l : (l[2], l[0]))
        elif sub_type == SUBTYPE.POS:
            sorted_triples = sorted(self.triples, key = lambda l : (l[2], l[1]))

        similar_entities = []
        current = np.zeros(len(E[0]), dtype = np.float64)
        count = 0
        prevo = -1
        prevp = -1
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

        print ("# of subgraphs ({}) : {}".format(sub_type_to_string[sub_type], self.get_Nsubgraphs()))

    def make_subgraphs(self):

        self.make_subgraphs_per_type(SUBTYPE.SPO)
        self.make_subgraphs_per_type(SUBTYPE.POS)
