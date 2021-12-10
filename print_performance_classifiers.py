import argparse
from support.utils import *
import json

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237', 'dbpedia50'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    return parser.parse_args()

args = parse_args()
classifiers = ['random', 'threshold', 'mlp_multi', 'lstm', 'conv', 'path', 'sub', 'min', 'maj', 'supensemble', 'snorkel', 'squid']

results_dir = args.result_dir + '/' + args.db + '/results/'

results = { 'head' : [], 'tail' : [] }
for type_prediction in ['head', 'tail']:
    for classifier in classifiers:
        suf = '-{}'.format(classifier)
        filename = get_filename_results(args.db, args.model, "test", args.topk, type_prediction, suf)
        path = results_dir + filename
        with open(path, 'rt') as fin:
            res = json.load(fin)
            results[type_prediction].append(res)

# Print the results
avg = {}

print("Type prediction: HEAD")
print("Method\t& Precision\t& Recall\t& F1\t& Accuracy")
for r in results['head']:
    if r['classifier'] not in avg:
        avg[r['classifier']] = [r]
    print("{}\t&{:.3f}\t&{:.3f}\t&{:.3f}\t&{:.3f}".format(r['classifier'], r['PREC'], r['REC'], r['F1'], r['accuracy']))

print("Type prediction: TAIL")
print("Method\t& Precision\t& Recall\t& F1\t& Accuracy")
for r in results['tail']:
    avg[r['classifier']].append(r)
    print("{}\t&{:.3f}\t&{:.3f}\t&{:.3f}\t&{:.3f}".format(r['classifier'], r['PREC'], r['REC'], r['F1'], r['accuracy']))

print("Type prediction: AVG")
print("Method\t& Precision\t& Recall\t& F1\t& Accuracy")
for name, r in avg.items():
    assert(len(r) == 2)
    avg_prec = (r[0]['PREC'] + r[1]['PREC']) / 2
    avg_rec = (r[0]['REC'] + r[1]['REC']) / 2
    avg_acc = (r[0]['accuracy'] + r[1]['accuracy']) / 2
    avg_f1 = (r[0]['F1'] + r[1]['F1']) / 2
    print("{}\t&{:.3f}\t&{:.3f}\t&{:.3f}\t&{:.3f}".format(name, avg_prec, avg_rec, avg_f1, avg_acc))



