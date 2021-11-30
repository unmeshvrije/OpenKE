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
annotators_ids = {}

n_annotations = 0
with open(file_path, encoding='utf-8') as fin:
    objects = json.load(fin)
    for key in objects.keys():
        annotator = objects[key]['annotator']
        if annotator not in annotators_ids:
            annotators_ids[annotator] = set()
        annotators_ids[annotator].add((objects[key]['query']['ent'], objects[key]['query']['rel']))
        n_annotations += 1

print("N. annotators", len(annotators_ids))
print("N. annotations", n_annotations)
intersection = None
for annotator, annotations in annotators_ids.items():
    if intersection is None:
        intersection = annotations
    else:
        intersection = intersection.intersection(annotations)
print("Size intersection", len(intersection))