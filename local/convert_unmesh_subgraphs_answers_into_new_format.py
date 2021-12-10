import sys
import json
import pickle
import argparse
import datetime
from support.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--db', dest='db', type=str, default="fb15k237", choices=['fb15k237', 'dbpedia50'])
    parser.add_argument('--mode', dest='mode', type=str, default="test", choices=['train', 'valid', 'test'])
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    parser.add_argument('--old_file', dest="old_file", type=str)
    return parser.parse_args()

args = parse_args()

# Load the answers
suf = ''
answers_filename = args.result_dir + '/' + args.db + '/misc/' + \
                   args.db + '-{}-test-topk-10.pkl'.format(args.model)
test_answers = pickle.load(open(answers_filename, 'rb'))
key = 'x_{}_fil'.format(args.type_prediction)
test_answers = test_answers[key]

testdata_classifier = pickle.load(open(args.old_file, 'rb'))
answers_classifier = testdata_classifier['fil']['predicted_y']
#answers_classifier_raw = testdata_classifier['raw']['predicted_y']
assert(len(answers_classifier) == len(test_answers))

if args.type_prediction == 'head':
    typ = 0
else:
    typ = 1

i = 0
output = []
while i < len(answers_classifier):
    predicted_answers = []
    answers_fil = []
    for _ in range(10):
        features = test_answers[i][:5]
        ent = features[0]
        rel = features[1]
        ans = features[2]
        value = answers_classifier[i]
        answers_fil.append(ans)
        if value == 1:
            checked = True
            score = 1
        else:
            checked = False
            score = 0
        predicted_answers.append({'entity_id' : ans, 'checked' : checked, 'score' : score})
        i += 1
    out = {}
    out['query'] = {'ent': ent, 'rel': rel, 'type' : typ, 'answers_fil' : answers_fil, 'answers_raw' : answers_fil }
    out['valid_annotations'] = True
    out['annotator'] = 'sub'
    out['date'] = str(datetime.datetime.now())
    out['annotated_answers'] = predicted_answers
    output.append(out)

# Store the output
suf = '-sub'
answers_annotations_filename = args.result_dir + '/' + args.db + '/annotations/' +\
                               get_filename_answer_annotations(args.db, args.model, args.mode, args.topk, args.type_prediction, suf)
with open(answers_annotations_filename, 'wb') as fout:
    pickle.dump(output, fout)
    fout.close()



