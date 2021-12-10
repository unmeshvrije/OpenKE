import json
import pickle
from datetime import datetime
import sys


base_dir = "dbpedia50"

annotations_out = 'misc/old_annotations.json'
out = []
for method  in ["transe", "complex", "rotate"]:
    annotations_unmesh_head = "dbpedia50/out/dbpedia50-"+method+"-annotated-topk-10-head.out-extended.json"
    annotations_unmesh_tail = "dbpedia50/out/dbpedia50-"+method+"-annotated-topk-10-tail.out-extended.json"

    annotations_head = json.load(open(annotations_unmesh_head, 'rt'))
    annotations_tail = json.load(open(annotations_unmesh_tail, 'rt'))
    for a in annotations_head:
        ent = a['e']
        rel = a['r']
        answers = []
        for answer in a['answers']:
            if answer['cor'] == 0:
                checked = False
            else:
                checked = True
            answers.append({'entity_id' : answer['answer'], 'checked' : checked, 'methods' : [method]})
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
            answers.append({'entity_id' : answer['answer'], 'checked' : checked, 'methods' : [method]})
        query = {'type' : 1, 'ent' : ent, 'rel' : rel}
        o = {'query' : query, 'annotated_answers' : answers, 'annotator' : 'U', 'date': str(datetime.now())}
        out.append(o)

fout = open(base_dir + '/' + annotations_out, 'wt')
json.dump(out, fout)
fout.close()




