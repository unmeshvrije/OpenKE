import json
import pickle
from datetime import datetime

base_dir = 'binary-embeddings/fb15k237/'
annotations_unmesh_transe_head = 'old_data/fb15k237/out/fb15k237-transe-annotated-topk-10-head-extended.json'
annotations_unmesh_transe_tail = 'old_data/fb15k237/out/fb15k237-transe-annotated-topk-10-tail-extended.json'

annotations_head = json.load(open(annotations_unmesh_transe_head, 'rt'))
annotations_tail = json.load(open(annotations_unmesh_transe_tail, 'rt'))
annotations_out = 'misc/old_annotations.json'
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




