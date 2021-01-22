import argparse
from support.dataset_fb15k237 import Dataset_FB15k237
from support.dataset_dbpedia50 import Dataset_dbpedia50
from support.utils import *
from support.embedding_model import Embedding_Model
import pickle
import datetime
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--classifier', dest='classifier', type=str, required=True, choices=['mlp', 'random', 'mlp_multi', 'lstm', 'conv', 'min', 'maj', 'snorkel', 'path', 'sub', 'threshold', 'supensemble', 'squid'])
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest ='db', type = str, default = "fb15k237", choices=['fb15k237', 'dbpedia50'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--mode', dest='mode', type=str, default="test", choices=['train', 'valid', 'test'])
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])

    # Parameters for the various classifiers
    parser.add_argument('--name_signals', dest='name_signals', help='name of the signals (classifiers) to use when multiple signals should be combined', type=str, required=False, default="mlp_multi,lstm,conv,path,sub")
    parser.add_argument('--snorkel_low_threshold', dest='snorkel_low_threshold', type=str, default="0.2,0.2,0.2,0,0")
    parser.add_argument('--snorkel_high_threshold', dest='snorkel_high_threshold', type=str, default="0.6,0.6,0.6,0.6,0.5")
    parser.add_argument('--sub_k', dest='sub_k', type=int, default=3)
    parser.add_argument('--mlp_n_hidden_units', dest='mlp_n_hidden_units', type=int, default=100)
    parser.add_argument('--mlp_dropout', dest='mlp_dropout', type=float, default=0.2)
    parser.add_argument('--lstm_n_hidden_units', dest='lstm_n_hidden_units', type=int, default=100)
    parser.add_argument('--lstm_dropout', dest='lstm_dropout', type=float, default=0.2)
    parser.add_argument('--conv_kern_size1', dest='conv_kern_size1', type=int, default=4)
    parser.add_argument('--conv_kern_size2', dest='conv_kern_size2', type=int, default=2)
    parser.add_argument('--threshold_k', dest='threshold_k', type=int, default=10)

    return parser.parse_args()

args = parse_args()

# Load dataset
dataset = None
if args.db == 'fb15k237':
    dataset = Dataset_FB15k237()
elif args.db == 'dbpedia50':
    dataset = Dataset_dbpedia50()
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
    model_filename = get_filename_classifier_model(args.db, args.model, args.classifier, args.topk, args.type_prediction)
    hyper_params = {"n_units": args.mlp_n_hidden_units, "dropout": args.mlp_dropout}
    classifier = Classifier_MLP(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                hyper_params=hyper_params,
                                model_path=model_dir + '/' + model_filename)
elif args.classifier == 'mlp_multi':
    from classifier_mlp_multi import Classifier_MLP_Multi
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.model, args.classifier, args.topk, args.type_prediction)
    hyper_params = {"n_units": args.mlp_n_hidden_units, "dropout": args.mlp_dropout}
    classifier = Classifier_MLP_Multi(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                hyper_params=hyper_params,
                                model_path=model_dir + '/' + model_filename)
elif args.classifier == 'lstm':
    from classifier_lstm import Classifier_LSTM
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.model, args.classifier, args.topk, args.type_prediction)
    hyper_params = {"n_units": args.lstm_n_hidden_units, "dropout": args.lstm_dropout}
    classifier = Classifier_LSTM(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                hyper_params=hyper_params,
                                model_path=model_dir + '/' + model_filename)
elif args.classifier == 'conv':
    from classifier_conv import Classifier_Conv
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.model, args.classifier, args.topk, args.type_prediction)
    hyper_params = {"kernel_size1": args.conv_kern_size1, "kernel_size2": args.conv_kern_size2, "topk" : args.topk}
    classifier = Classifier_Conv(dataset,
                                args.type_prediction,
                                args.result_dir,
                                embedding_model,
                                args.topk,
                                hyper_params=hyper_params,
                                model_path=model_dir + '/' + model_filename)
elif args.classifier == 'random':
    from classifier_random import Classifier_Random
    classifier = Classifier_Random(dataset, args.type_prediction, args.result_dir)
elif args.classifier == 'threshold':
    from classifier_threshold import Classifier_Threshold
    classifier = Classifier_Threshold(dataset, args.type_prediction, args.result_dir, threshold=args.threshold_k)
elif args.classifier == 'path':
    from classifier_path import Classifier_Path
    classifier = Classifier_Path(dataset, args.type_prediction, args.result_dir)
elif args.classifier == 'sub':
    from classifier_subgraphs import Classifier_Subgraphs
    classifier = Classifier_Subgraphs(dataset, args.type_prediction, embedding_model, args.result_dir, args.sub_k)
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
    lows = args.snorkel_low_threshold.split(",")
    highs = args.snorkel_high_threshold.split(",")
    thresholds = []
    for i, l in enumerate(lows):
        h = highs[i]
        thresholds.append((float(l), float(h)))
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.model, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_Snorkel(dataset, args.type_prediction, args.topk, args.result_dir,
                                    signals, embedding_model_typ, model_path=model_dir + '/' + model_filename, abstain_scores=thresholds)
elif args.classifier == 'squid':
    from classifier_squid import Classifier_Squid
    signals = args.name_signals.split(",")
    lows = args.snorkel_low_threshold.split(",")
    highs = args.snorkel_high_threshold.split(",")
    thresholds = []
    for i, l in enumerate(lows):
        h = highs[i]
        thresholds.append((float(l), float(h)))
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.model, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_Squid(dataset, args.type_prediction, args.topk, args.result_dir,
                                    signals, embedding_model_typ, model_path=model_dir + '/' + model_filename, abstain_scores=thresholds)
elif args.classifier == 'supensemble':
    from classifier_supensemble import Classifier_SuperEnsemble
    signals = args.name_signals.split(",")
    model_dir = args.result_dir + '/' + args.db + '/models/'
    model_filename = get_filename_classifier_model(args.db, args.model, args.classifier, args.topk, args.type_prediction)
    classifier = Classifier_SuperEnsemble(dataset, args.type_prediction, args.topk, args.result_dir, signals,
                                          embedding_model_typ, model_path=model_dir + '/' + model_filename)
else:
    raise Exception("Classifier {} not supported!".format(args.classifier))

# Determine the type of answers
if args.mode == 'test':
    type_answers = 'answers_fil'
elif args.mode == 'train':
    type_answers = 'answers_raw'
else:
    type_answers = None
    raise Exception("Case not implemented")

# Launch the predictions
output = []
for item in tqdm(queries_with_answers):
    ent = item['ent']
    rel = item['rel']
    predicted_answers = classifier.predict(item, type_answers)
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