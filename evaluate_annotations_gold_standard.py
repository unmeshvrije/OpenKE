import argparse
import pickle
import json
from support.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--classifier', dest='classifier', type=str, required=True, choices=['mlp', 'random', 'mlp_multi', 'lstm', 'conv', 'min', 'maj', 'snorkel', 'path', 'sub'])
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
n_gold_annotations = 0
n_true_gold_annotations = 0
n_false_gold_annotations = 0
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
                    if a['checked']:
                        n_true_gold_annotations += 1
                    else:
                        n_false_gold_annotations += 1
                    break
        assert(len(ans) == args.topk)
        filter_queries[(ent, rel)] = ans
        n_gold_annotations += len(ans)

# Compute the various metrics
print("*********")
print("Dataset\t\t\t: {}". format(args.db))
print("Classifier\t\t: {}". format(args.classifier))
print("Type prediction\t: {}". format(args.type_prediction))
matched_answers = 0
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
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
                    if checked == True and true_answer['checked'] == True:
                        true_positives += 1
                    elif checked == True:
                        false_positives += 1
                    elif true_answer['checked'] == True:
                        false_negatives += 1
                    else:
                        true_negatives += 1
                    break
            assert(found)
acc = matched_answers / n_gold_annotations
rec = true_positives / (true_positives + false_negatives)
prec = true_positives / (true_positives + false_positives)
f1 = 2 * (prec * rec) / (prec + rec)
print("Accuracy\t\t: {:.3f}".format(acc))
print("Recall\t\t\t: {:.3f}".format(rec))
print("Precision\t\t: {:.3f}".format(prec))
print("F1\t\t\t\t: {:.3f}".format(f1))
print("*********")

results = { "F1" : f1, "REC" : rec, "PREC" : prec, "dataset" : args.db, "classifier" : args.classifier,
            "type_prediction" : args.type_prediction}
suf = '-' + args.classifier
results_filename = args.result_dir + '/' + args.db +\
                   '/results/' +\
                   get_filename_results(args.db, args.model, args.mode, args.topk, args.type_prediction, suf)
fout = open(results_filename, 'wt')
json.dump(results, fout)
fout.close()