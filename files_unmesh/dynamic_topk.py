import pickle
from openke.utils import DeepDict
from files_unmesh.subgraphs import read_triples

class DynamicTopk:
    def __init__(self, default=10):
        self.topk_dict_head = DeepDict()
        self.topk_dict_tail = DeepDict()
        self.default_topk = default

    def get_dyn_topk(self, ent, rel, type_prediction):
        if type_prediction == "head":
            if (ent, rel) in self.topk_dict_head:
                return self.topk_dict_head[(ent, rel)]
            else:
                return self.default_topk
        elif type_prediction == "tail":
            if (ent, rel) in self.topk_dict_tail:
                return self.topk_dict_tail[(ent, rel)]
            else:
                return self.default_topk

    def populate(self, triples_file):
        triples = read_triples(triples_file)
        for triple in triples:
            if (triple[0], triple[2]) in self.topk_dict_tail:
                self.topk_dict_tail[(triple[0],triple[2])] += 1
            else:
                self.topk_dict_tail[(triple[0],triple[2])] = 1

            if (triple[1], triple[2]) in self.topk_dict_head:
                self.topk_dict_head[(triple[1], triple[2])] += 1
            else:
                self.topk_dict_head[(triple[1], triple[2])] = 1

    def load(self, dyn_topk_head_filename, dyn_topk_tail_filename):
        with open(dyn_topk_tail_filename, 'rb') as fin:
            self.topk_dict_tail = pickle.load(fin)

        with open(dyn_topk_head_filename, 'rb') as fin:
            self.topk_dict_head = pickle.load(fin)

    def save(self, dyn_topk_head_filename, dyn_topk_tail_filename):
        with open(dyn_topk_tail_filename, 'wb') as fout:
            pickle.dump(self.topk_dict_tail, fout, protocol = pickle.HIGHEST_PROTOCOL)

        with open(dyn_topk_head_filename, 'wb') as fout:
            pickle.dump(self.topk_dict_head, fout, protocol = pickle.HIGHEST_PROTOCOL)
