from tqdm import tqdm
import pickle
import sys
import os
from pathlib import Path


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <database>")
    sys.exit()

db = sys.argv[1]
dir_prefix = "/home/uji300/OpenKE/benchmarks/"
efile = dir_prefix + db + "/entity2id.txt"
rfile = dir_prefix + db + "/relation2id.txt"

#entity2id.txt  n-1.txt  n-n.py  n-n.txt  relation2id.txt  test2id_all.txt  test2id.txt  train2id.txt  type_constrain.txt  valid2id.txt

train_file = dir_prefix + db + "/train2id.txt"
valid_file = dir_prefix + db + "/valid2id.txt"
test_file =  dir_prefix + db + "/test2id.txt"


eid_to_fid = {}
rid_to_rel = {}
id_to_relation = {}
id_to_entity = {}
'''
idfile = "/var/scratch2/uji300/kbs/fb15k237-id-to-entity.tsv"
fbdict = {}
with open(idfile, "r") as fin:
    lines = fin.readlines()
    for line in tqdm(lines):
        cols = line.split(maxsplit=1)
        if len(cols) < 2:
            #print(line)
            continue
        key = cols[0]
        val = cols[1]
        fbdict[key] = val
'''

cnt = 0
with open(efile, "r")as fin:
    lines = fin.readlines()
    for line in tqdm(lines[1:]):
        fid = line.split()[0].rstrip()
        eid = line.split()[1].rstrip()
        id_to_entity[int(eid)] = fid

with open(rfile, "r") as fin:
    lines = fin.readlines()
    for line in tqdm(lines[1:]):
        cols = line.split(maxsplit=1)
        val = cols[0]
        key = cols[1]
        id_to_relation[int(key)] = val.rstrip()

def expand_file(name):
    for x in ["train", "valid", "test"]:
        if x in name:
            outname = result_dir + x + ".txt"

    newlines = ""
    with open(name, "r") as fin:
        lines = fin.readlines()
        for line in lines[1:]:
            head = int(line.split()[0])
            tail = int(line.split()[1])
            rel  = int(line.split()[2].rstrip())
            this_line = eid_to_fid[head] +"\t"+rid_to_rel[rel]+"\t"+eid_to_fid[tail]+"\n"
            newlines += this_line
            #print(this_line)
    # make a new file with new lines
    with open(outname, "w") as fout:
        fout.write(newlines)

#expand_file(train_file)
#expand_file(valid_file)
#expand_file(test_file)
result_dir='/var/scratch2/uji300/OpenKE-results/' + db + '/misc/' 
Path(result_dir).mkdir(parents=True, exist_ok=True)

with open(result_dir + db + '-id-to-entity.pkl', 'wb') as fout:
    pickle.dump(id_to_entity, fout, protocol = pickle.HIGHEST_PROTOCOL)

with open(result_dir + db + '-id-to-relation.pkl', 'wb') as fout:
    pickle.dump(id_to_relation, fout, protocol = pickle.HIGHEST_PROTOCOL)
