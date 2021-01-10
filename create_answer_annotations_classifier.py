import argparse
from support.dataset_fb15k237 import Dataset_FB15k237
from support.utils import *
from support.embedding_model import Embedding_Model
import pickle
import datetime
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--classifier', dest='classifier', type=str, required=True, choices=['mlp', 'random', 'mlp_multi', 'lstm', 'conv', 'min', 'maj', 'snorkel', 'path'])
    parser.add_argument('--name_signals', dest='name_signals', help='name of the signals (classifiers) to use when multiple signals should be combined', type=str, required=False, default="mlp_multi,lstm,conv")
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--mode', dest='mode', type=str, default="test", choices=['train', 'valid', 'test'])
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    return parser.parse_args()

args = parse_args()

# Load dataset
dataset = None
if args.db == 'fb15k237':
    dataset = Dataset_FB15k237()
else:
    raise Exception("DB {} not supported!".format(args.db))

# Load answers
suf = ''
answers_filename = args.result_dir + '/' + args.db + '/answers/' + get_filename_answers(args.db, args.model, args.mode, args.topk, args.type_prediction, suf)
queries_with_answers = pickle.load(open(answers_filename, 'rb'))

# Load embedding model
embedding_model_typ = args.model
embedding_model = Embedding_Model(args.result_dir, embedding_model_typ, dataset)

if args.classifier == 'mlp':
    from classifier_mlp import Classifier_MLP
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_MLP(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                None,
                                model_dir + '/' + model_filename)
elif args.classifier == 'mlp_multi':
    from classifier_mlp_multi import Classifier_MLP_Multi
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_MLP_Multi(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                None,
                                model_dir + '/' + model_filename)
elif args.classifier == 'lstm':
    from classifier_lstm import Classifier_LSTM
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_LSTM(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                None,
                                model_dir + '/' + model_filename)
elif args.classifier == 'conv':
    from classifier_conv import Classifier_Conv
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_Conv(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                None,
                                model_dir + '/' + model_filename)
elif args.classifier == 'random':
    from classifier_random import Classifier_Random
    classifier = Classifier_Random(dataset, args.type_prediction, args.result_dir)
elif args.classifier == 'path':
    from classifier_path import Classifier_Path
    classifier = Classifier_Path(dataset, args.type_prediction, args.result_dir)
elif args.classifier == 'min':
    from classifier_majmin import Classifier_MajMin
    signals = args.name_signals.split(",")
    classifier = Classifier_MajMin(dataset, args.type_prediction, args.topk, args.result_dir,
                                   args.model, signals, True)
elif args.classifier == 'maj':
    from classifier_majmin import Classifier_MajMin
    signals = args.name_signals.split(",")
    classifier = Classifier_MajMin(dataset, args.type_prediction, args.topk, args.result_dir,
                                   args.model, signals, False)
elif args.classifier == 'snorkel':
    from classifier_snorkel import Classifier_Snorkel
    signals = args.name_signals.split(",")
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_Snorkel(dataset, args.type_prediction, args.topk, args.result_dir,
                                    signals, embedding_model_typ, model_dir + '/' + model_filename)
else:
    raise Exception("Classifier {} not supported!".format(args.classifier))

# Launch the predictions
output = []
for item in tqdm(queries_with_answers):
    ent = item['ent']
    rel = item['rel']
    predicted_answers = classifier.predict(item)
    out = {}
    out['query'] = item
    out['valid_annotations'] = True
    out['annotator'] = args.classifier
    out['date'] = str(datetime.datetime.now())
    out['annotated_answers'] = predicted_answers
    output.append(out)

# Store the output
suf = '-' + args.classifier
answers_annotations_filename = args.result_dir + '/' + args.db + '/annotations/' + get_filename_answer_annotations(args.db, args.model, args.mode, args.topk, args.type_prediction, suf)
with open(answers_annotations_filename, 'wb') as fout:
    pickle.dump(output, fout)
    fout.close()