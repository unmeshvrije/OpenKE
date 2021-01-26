import argparse
from os import listdir
import pickle
import json

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237','dbpedia50'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--filter_key', dest='filter_key', type=str, required=True)
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    return parser.parse_args()

args = parse_args()

print("Loading configurations ...")
configurations = []
dir = args.result_dir + '/' + args.db + '/paramtuning'
key = args.filter_key
onlyfiles = [f for f in listdir(dir) if args.type_prediction in f and args.model in f and str(args.topk) in f and key in f]
for f in onlyfiles:
    with open(dir + '/' + f, 'rb') as fin:
        conf = json.load(fin)
        configurations.append(conf)

# Sort the configurations by F1
sorted_list = sorted(configurations, key=lambda x: x['F1'], reverse=True)
print("Best F1 ({}):".format(args.type_prediction))
for c in sorted_list[:10]:
    print(" ", c)

# Sort the configurations by precision
sorted_list = sorted(configurations, key=lambda x: x['PREC'], reverse=True)
print("Best precision ({}):".format(args.type_prediction))
for c in sorted_list[:10]:
    print(" ", c)

# Sort the configurations by recall
sorted_list = sorted(configurations, key=lambda x: x['REC'], reverse=True)
print("Best recall ({}):".format(args.type_prediction))
for c in sorted_list[:10]:
    print(" ", c)

# Sort the configurations by accuracy
sorted_list = sorted(configurations, key=lambda x: x['accuracy'], reverse=True)
print("Best Accuracy ({}):".format(args.type_prediction))
for c in sorted_list[:10]:
    print(" ", c)
