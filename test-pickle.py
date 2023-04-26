import pickle
import sys
from enum import Enum

SUBTYPE = Enum('SUBTYPE', 'SPO, POS')
class Subgraph():
    def __init__(self, sid, st, sent, srel, ssize, entities):
        self.data = {}
        self.data['subType'] = st
        self.data['subId']   = sid
        self.data['ent']     = sent
        self.data['rel']     = srel
        self.data['size']    = ssize
        self.data['entities']= copy.deepcopy(entities)

with open(sys.argv[1], 'rb') as fin:
    data = pickle.load(fin)

print(len(data))
