import argparse
import pickle
import json
from support.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--classifier', dest='classifier', type=str, required=True, choices=['mlp'])
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--mode', dest='mode', type=str, default="test", choices=['train', 'valid', 'test'])
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    return parser.parse_args()

args = parse_args()

# Load annotations produced by a classifier
suf = '-' + args.classifier
answers_annotations_filename = args.result_dir + '/' + args.db + '/annotations/' + get_filename_answer_annotations(args.db, args.model, args.mode, args.topk, args.type_prediction, suf)
with open(answers_annotations_filename, 'rb') as fin:
    annotated_answers = pickle.load(fin)

# Load the gold standard
gold_dir = args.result_dir + '/' + args.db + '/annotations/'
gold_filename = get_filename_gold(args.db, args.topk)
with open(gold_dir + gold_filename, 'rt') as fin:
    gold_annotations = json.load(fin)
filter_queries = {}
n_annotations = 0
if args.type_prediction == 'head':
    accepted_query_type = 0
else:
    accepted_query_type = 1
for id, item in gold_annotations.items():
    query = item['query']
    type = query['type']
    if type == accepted_query_type and item['valid_annotations'] == True:
        ent = query['ent']
        rel = query['rel']
        filter_queries[(ent, rel)] = item['annotated_answers']
        n_annotations += len(item['annotated_answers'])

# Compute the various metrics
print("*********")
print("Dataset: {}". format(args.db))
print("Classifier: {}". format(args.classifier))
print("Type prediction: {}". format(args.type_prediction))
matched_answers = 0
for query_answers in annotated_answers:
    ent = query_answers['query']['ent']
    rel = query_answers['query']['rel']
    if (ent, rel) in filter_queries:
        true_annotated_answers = filter_queries[(ent, rel)]
        assert(query_answers['valid_annotations'])
        assert(query_answers['annotator'] == args.classifier)
        for ans in query_answers['annotated_answers']:
            entity_id = ans['entity_id']
            checked = ans['checked']
            found = False
            for true_answer in true_annotated_answers:
                if true_answer['entity_id'] == entity_id:
                    found = True
                    matched_answers += checked == true_answer['checked']
                    break
            assert(found)
print("Accuracy: {}".format(matched_answers / n_annotations))
print("*********")
