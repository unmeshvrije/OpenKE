import argparse
import json
from support.utils import *
from sklearn.model_selection import train_test_split
import os

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', required=True, type = str, help = 'Output dir.')
    parser.add_argument('--db', dest ='db', type = str, default = "fb15k237", choices=['fb15k237', 'dbpedia50'])
    return parser.parse_args()

args = parse_args()
topk = 10
testdata_raw_path = "../benchmarks/" + args.db + "/test2id.txt"
validdata_raw_path = "../benchmarks/" + args.db + "/valid2id.txt"

raw_test_triples = set()
with open(testdata_raw_path, 'rt') as f:
    nfacts = int(f.readline())
    for l in f:
        tkns = l.split(' ')
        h = int(tkns[0])
        t = int(tkns[1])
        r = int(tkns[2])
        raw_test_triples.add((h, t, r))
with open(validdata_raw_path, 'rt') as f:
    nfacts = int(f.readline())
    for l in f:
        tkns = l.split(' ')
        h = int(tkns[0])
        t = int(tkns[1])
        r = int(tkns[2])
        raw_test_triples.add((h, t, r))

# Load the gold standard
gold_dir = args.result_dir + '/' + args.db + '/annotations/'
gold_filename = get_filename_gold(args.db, topk)
new_gold_filename = gold_filename + ".new"
with open(gold_dir + gold_filename, 'rt') as fin:
    gold_annotations = json.load(fin)
new_annotations = {}
for key, query in gold_annotations.items():
    # Convert the annotators
    new_query = query
    typ = new_query['query']['type']
    ent = new_query['query']['ent']
    rel = new_query['query']['rel']
    annotator = new_query['annotator']
    date = new_query['date']
    del new_query['annotator']
    del new_query['date']
    new_annotated_answers = []
    add_old_annotator = False
    for annotated_answer in new_query['annotated_answers']:
        a = annotated_answer['entity_id']
        checked = annotated_answer['checked']
        found = False
        if typ == 0 and (a, ent, rel) in raw_test_triples:
            found = True
        if typ == 1 and (ent, a, rel) in raw_test_triples:
            found = True
        if 'enabled' in annotated_answer:
            enabled = annotated_answer['enabled']
        else:
            enabled = not found
        if found:
            enabled = False
        if not enabled:
            if found:
                if not checked:
                    print("true triple not checked")
                annotated_answer['checked'] = [{'checked' : True, 'annotator' : 'Testset'}]
            else:
                # This comes from a previous set of annotations
                annotated_answer['checked'] = [{'checked': checked, 'annotator': 'U'}]
                add_old_annotator = True
                annotated_answer['enabled'] = True
        else:
            annotated_answer['checked'] = [{'checked' : checked, 'annotator' : annotator}]
        new_annotated_answers.append(annotated_answer)
    new_query['annotated_answers'] = new_annotated_answers
    if not add_old_annotator:
        new_query['annotators'] = [{'name': annotator, 'date': date}]
    else:
        new_query['annotators'] = [{'name': annotator, 'date': date}, {'name' : 'U', 'date' : 'NA' }]
    new_annotations[key] = new_query
with open(gold_dir + '/' + new_gold_filename, 'wt') as fout:
    json.dump(new_annotations, fout)