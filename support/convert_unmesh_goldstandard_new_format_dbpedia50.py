import json
import pickle
from datetime import datetime

base_dir = "/var/scratch2/uji300/ijcai2021/binary-embeddings/dbpedia50"

annotations_unmesh_transe_head = "/var/scratch2/uji300/OpenKE-results/dbpedia50/out/dbpedia50-transe-annotated-topk-10-head.out-extended.json"
annotations_unmesh_transe_tail = "/var/scratch2/uji300/OpenKE-results/dbpedia50/out/dbpedia50-transe-annotated-topk-10-tail.out-extended.json"

annotations_head = json.load(open(annotations_unmesh_transe_head, 'rt'))
annotations_tail = json.load(open(annotations_unmesh_transe_tail, 'rt'))
annotations_out = 'misc/annotations_unmesh_dbpedia50.json'
out = []
for a in annotations_head:
    ent = a['e']
    rel = a['r']
    answers = []
    for answer in a['answers']:
        if answer['cor'] == 0:
            checked = False
        else:
            checked = True
        answers.append({'entity_id' : answer['answer'], 'checked' : checked, 'methods' : ['transe']})
    query = {'type' : 0, 'ent' : ent, 'rel' : rel}
    o = {'query' : query, 'annotated_answers' : answers, 'annotator' : 'U', 'date': str(datetime.now())}
    out.append(o)

for a in annotations_tail:
    ent = a['e']
    rel = a['r']
    answers = []
    for answer in a['answers']:
        if answer['cor'] == 0:
            checked = False
        else:
            checked = True
        answers.append({'entity_id' : answer['answer'], 'checked' : checked, 'methods' : ['transe']})
    query = {'type' : 1, 'ent' : ent, 'rel' : rel}
    o = {'query' : query, 'annotated_answers' : answers, 'annotator' : 'U', 'date': str(datetime.now())}
    out.append(o)

fout = open(base_dir + '/' + annotations_out, 'wt')
json.dump(out, fout)
fout.close()




