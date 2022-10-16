from tqdm import tqdm
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'Create files that translate fb15k237 subgraph ids to entity or relation names.')
    parser.add_argument('--idfile', dest = 'idfile', type = str, default = '/var/scratch/dvs254/kbs/fb15k237-id-to-entity.tsv', help = 'File containing existing subgraph id to entity data')
    parser.add_argument('--savedir', dest = 'save_dir', type = str, default = '/var/scratch/dvs254/kbs/', help = 'Directory in which id to entity and id to relation files will be created')
    return parser.parse_args()

args = parse_args()

efile = "benchmarks/fb15k237/entity2id.txt"

rfile = "benchmarks/fb15k237/relation2id.txt"

idfile = args.idfile
save_dir = args.save_dir

eid_to_fid = {}

fbdict = {}
with open(idfile, "r") as fin:
    lines = fin.readlines()
    for line in tqdm(lines):
        cols = line.split(maxsplit = 1)
        if len(cols) < 2:
            #print(line)
            continue
        key = cols[0]
        val = cols[1]
        fbdict[key] = val

cnt = 0
id_to_entity = {}
with open(efile, "r")as fin:
    lines = fin.readlines()
    for line in lines[1:]:
        fid = line.split()[0]
        eid = line.split()[1]
        if fid not in fbdict:
            cnt += 1
            id_to_entity[int(eid)] = "_"
        else:
            id_to_entity[int(eid)] = fbdict[fid].rstrip()

id_to_relation = {}
with open(rfile, "r") as fin:
    lines = fin.readlines()
    for line in tqdm(lines[1:]):
        cols = line.split(maxsplit = 1)
        val = cols[0]
        key = cols[1]
        id_to_relation[int(key)] = val.rstrip()

with open(save_dir + 'fb15k237-id-to-entity.pkl', 'wb') as fout:
    pickle.dump(id_to_entity, fout, protocol = pickle.HIGHEST_PROTOCOL)

with open(save_dir + 'fb15k237-id-to-relation.pkl', 'wb') as fout:
    pickle.dump(id_to_relation, fout, protocol = pickle.HIGHEST_PROTOCOL)
