import os
import json
import argparse

from files_unmesh.subgraphs import SubgraphFactory
from files_unmesh.subgraphs import read_triples

import kge.model

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read embeddings and prepare subgraphs.')
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--embfile', required = True, dest ='embfile', type = str, help = 'File containing embeddings.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--infile', dest ='infile', type = str, help = 'File containing training triples.', default = "/home/uji300/OpenKE/benchmarks/fb15k237/train2id.txt")
    parser.add_argument('--ms', dest = 'ms', type = int, default = 10, help = 'Minimum subgraph Size')
    parser.add_argument('--model', dest = 'model', type = str, default = "transe", help = 'Embedding model')
    return parser.parse_args()

args = parse_args()

ms = args.ms
db = args.db
result_dir = args.result_dir + db + "/subgraphs/"
os.makedirs(result_dir, exist_ok = True)

# read complex embeddings from the LibKGE
#'./local/fb15k-237-complex.pt'
def read_complex_embeddings(filename):
    model = kge.model.KgeModel.load_from_checkpoint(filename)
    E = model._entity_embedder._embeddings_all()
    R = model._relation_embedder._embeddings_all()
    return E.tolist(), R.tolist()

def read_embeddings(filename):
    with open(filename, "r") as fin:
        params = json.loads(fin.read())
    E = params['ent_embeddings.weight']
    R = params['rel_embeddings.weight']
    return E, R

triples = read_triples(args.infile)
if args.model == "complex":
    E, R = read_complex_embeddings(args.embfile)
else:
    E, R = read_embeddings(args.embfile)
#print(type(E))
#print(len(E))
#print(len(E[0]))
#print(type(R))
sub_factory = SubgraphFactory(args.db, int(args.ms), triples, E)
sub_factory.make_subgraphs()
sub_factory.save(result_dir, args.model)
