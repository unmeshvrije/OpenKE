import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', required=True, type = str, help = 'Output dir.')
    parser.add_argument('--db', dest ='db', type = str, default = "fb15k237", choices=['fb15k237', 'dbpedia50'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    return parser.parse_args()

args = parse_args()
topk = 10
file_path = args.result_dir + '/' + args.db + '/annotations/gold-annotations.json'
facts_annotator = {}

n_annotations = 0
with open(file_path, encoding='utf-8') as fin:
    objects = json.load(fin)
    for key in objects.keys():
        ent = objects[key]['query']['ent']
        rel = objects[key]['query']['rel']
        typ = objects[key]['query']['type']
        if objects[key]['valid_annotations'] == False:
            continue
        annotations = objects[key]['annotated_answers']
        for annotation in annotations:
            a = annotation['entity_id']
            fact = (ent, rel, a, typ)
            for c in annotation['checked']:
                annotator = c['annotator']
                checked = c['checked']
                if fact not in facts_annotator:
                    facts_annotator[fact] = []
                facts_annotator[fact].append((annotator, checked))
            n_annotations += 1

n_annotations = 0
n_agreed = 0
n_yes1 = 0
n_yes2 = 0
for fact, an in facts_annotator.items():
    # Check that every fact was annotated by one annotator exactly once
    annotator_fact = set()
    is_testset = False
    for a,_ in an:
        annotator_fact.add(a)
        if a == 'Testset':
            is_testset = True
            break
    if is_testset:
        continue
    assert(len(annotator_fact) == len(an))
    if len(an) == 1:
        continue
    assert(len(an) == 2)
    if an[0][1] == an[1][1]:
        n_agreed += 1
    if an[0][1] == True:
        n_yes1 += 1
    if an[1][1] == True:
        n_yes2 += 1
    n_annotations += 1

p0 = n_agreed / n_annotations
pe = (n_yes1 / (n_annotations * 2)) * ((n_yes2 / (n_annotations * 2))) +\
     ((n_annotations - n_yes1) / (n_annotations * 2)) * (((n_annotations - n_yes2) / (n_annotations * 2)))
cohen_confidence = (p0 - pe) / (1 - pe)

print("N. agreed", n_agreed)
print("N. annotations", n_annotations)
print("p0", p0)
print("pe", pe)
print("Cohen_confidence", cohen_confidence)

#Coefficient between 0.81-0.99 => near perfect agreement

# With dbpedia50
#p0 0.8973214285714286
#pe 0.2114780970982143
#Cohen_confidence 0.8697834885109583

# With fb15k237
#p0 0.8513513513513513
#pe 0.20169831994156318
#Cohen_confidence 0.8137938922566624

