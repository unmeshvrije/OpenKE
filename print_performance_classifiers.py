import argparse
from support.utils import *
import json

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    return parser.parse_args()

args = parse_args()
classifiers = ['random', 'mlp_multi', 'lstm', 'conv', 'path', 'sub', 'min', 'maj', 'snorkel']

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
print("Type prediction: HEAD")
print("Method\t\t\t\t\tPrecision\t\t\t\t\tRecall\t\t\tF1\t\t\t\t\tAccuracy")
for r in results['head']:
    print("{}\t\t\t\t\t{:.3f}\t\t\t\t\t{:.3f}\t\t\t\t\t{:.3f}\t\t\t\t\t{:.3f}".format(r['classifier'], r['PREC'], r['REC'], r['F1'], -1))

print("Type prediction: TAIL")
print("Method\t\t\t\t\tPrecision\t\t\t\t\tRecall\t\t\tF1\t\t\t\t\tAccuracy")
for r in results['tail']:
    print("{}\t\t\t\t\t{:.3f}\t\t\t\t\t{:.3f}\t\t\t\t\t{:.3f}\t\t\t\t\t{:.3f}".format(r['classifier'], r['PREC'], r['REC'], r['F1'], -1))



