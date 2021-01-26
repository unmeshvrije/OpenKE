import argparse
import json
from support.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--classifier', dest='classifier', type=str, required=True, choices=['mlp', 'random', 'mlp_multi', 'lstm', 'conv', 'min', 'maj', 'snorkel', 'path', 'sub', 'threshold', 'supensemble', 'squid'])
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237', 'dbpedia50'])
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
gold_filename = get_filename_gold(args.db, args.topk, '-test')
with open(gold_dir + gold_filename, 'rt') as fin:
    gold_annotations = json.load(fin)
filter_queries = {}
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
        ans = []
        for a in item['annotated_answers']:
            methods = a['methods']
            for m in methods:
                if m == args.model:
                    ans.append(a)
                    break
        assert(len(ans) == args.topk)
        filter_queries[(ent, rel)] = ans

# Compute the various metrics
print("*********")
print("Dataset\t\t\t: {}". format(args.db))
print("Classifier\t\t: {}". format(args.classifier))
print("Type prediction\t: {}". format(args.type_prediction))
results = compute_metrics(args.classifier, args.type_prediction, args.db, annotated_answers, filter_queries)
results['test_size'] = len(gold_annotations)
suf = '-' + args.classifier
results_filename = args.result_dir + '/' + args.db +\
                   '/results/' +\
                   get_filename_results(args.db, args.model, args.mode, args.topk, args.type_prediction, suf)
fout = open(results_filename, 'wt')
json.dump(results, fout)
fout.close()