import argparse
from support.utils import *
from support.dataset_fb15k237 import Dataset_FB15k237
from support.dataset_dbpedia50 import Dataset_dbpedia50
import pickle
import datetime
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237', 'dbpedia50'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--mode', dest='mode', type=str, default="test", choices=['train', 'valid', 'test'])
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    return parser.parse_args()

args = parse_args()

dataset = None
if args.db == 'fb15k237':
    dataset = Dataset_FB15k237()
elif args.db == 'dbpedia50':
    dataset = Dataset_dbpedia50()
else:
    raise Exception("DB {} not supported!".format(args.db))

# Load the answers
answers_filename = args.result_dir + '/' + args.db + '/answers/' + get_filename_answers(args.db, args.model, args.mode, args.topk, args.type_prediction, '')
answers = pickle.load(open(answers_filename, 'rb'))

# Annotate every answer using the information that we have. This tool is meant to process only queries/answers that
# come from the training data. Thus, I only look at the 'raw' answers
assert(args.mode == 'train')
annotated_answers = []
for idx in tqdm(range(0, len(answers))):
    a = answers[idx]
    a_out = []
    answers_raw = a['answers_raw']
    ent = a['ent']
    rel = a['rel']
    for entity in answers_raw:
        entity_id = entity['entity_id']
        score = entity['score']
        # Check if exists
        exists = False
        if args.type_prediction == 'head':
            exists = dataset.exists_htr(entity_id, ent, rel)
        else:
            assert(args.type_prediction == 'tail')
            exists = dataset.exists_htr(ent, entity_id, rel)
        a_out.append({'entity_id': entity_id, 'checked': exists, 'score' : score})
    out = {}
    out['query'] = { 'type' : a['type'], 'rel' : a['rel'], 'ent': a['ent'] }
    out['valid_annotations'] = True
    out['annotator'] = 'CWA'
    out['date'] = str(datetime.datetime.now())
    out['annotated_answers'] = a_out
    annotated_answers.append(out)

# Write the annotations to a file
answers_annotations_filename = args.result_dir + '/' + args.db + '/annotations/' + get_filename_answer_annotations(args.db, args.model, args.mode, args.topk, args.type_prediction, '')
with open(answers_annotations_filename, 'wb') as fout:
    pickle.dump(annotated_answers, fout)