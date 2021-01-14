import argparse
from support.embedding_model import Embedding_Model
from support.dataset_fb15k237 import Dataset_FB15k237
from classifier_subgraphs import Classifier_Subgraphs
import pickle
import json
from support.utils import *
from sklearn.model_selection import train_test_split
import datetime
from classifier_lstm import Classifier_LSTM
from classifier_mlp_multi import Classifier_MLP_Multi


def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--tune_lstm', dest='tune_lstm', type=bool, default=False)
    parser.add_argument('--tune_mlp_multi', dest='tune_mlp_multi', type=bool, default=False)
    parser.add_argument('--tune_sub', dest='tune_sub', type=bool, default=False)
    return parser.parse_args()

args = parse_args()
use_valid_data = 0.05
tune_sub = args.tune_sub
tune_lstm = args.tune_lstm
tune_mlp_multi = args.tune_mlp_multi

# ***** PARAMS TO TUNE *****
sub_ks = [ 1, 3, 5, 10, 25, 50, 100 ]
lstm_nhid = [ 10, 100, 1000 ]
lstm_dropout = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
mlp_nhid = [ 10, 20, 50, 100, 500, 1000, 1500, 2000 ]
mlp_dropout = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

def tune_sub_classifier(type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir):
    for sub_k in sub_ks:
        print("Test {} sub with k={}".format(type_prediction, sub_k))
        classifier = Classifier_Subgraphs(dataset, type_prediction, embedding_model, args.result_dir, sub_k)
        output = []
        for item in tqdm(valid_data_to_test):
            predicted_answers = classifier.predict(item)
            out = {}
            out['query'] = item
            out['valid_annotations'] = True
            out['annotator'] = 'sub'
            out['date'] = str(datetime.datetime.now())
            out['annotated_answers'] = predicted_answers
            output.append(out)
        results = compute_metrics('sub', type_prediction, args.db, output, gold_valid_data)
        results['sub_k'] = sub_k
        # Store the output
        suf = '-sub-classifier-k-' + str(sub_k)
        answers_annotations_filename = out_dir + get_filename_answer_annotations(args.db, args.model, 'valid', args.topk, type_prediction, suf)
        with open(answers_annotations_filename, 'wb') as fout:
            pickle.dump(output, fout)
            fout.close()

def tune_lstm_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir):
    for hidden_units in lstm_nhid:
        for dropout in lstm_dropout:
            print("Test {} LSTM with nhidden={} dropout={}".format(type_prediction, hidden_units, dropout))
            hyper_params = {"n_units": hidden_units, "dropout": dropout}
            classifier = Classifier_LSTM(dataset, type_prediction, args.result_dir, embedding_model, hyper_params, None)

            # Create training data
            td = classifier.create_training_data(training_data)

            # Train a model
            classifier.train(td, None, None)

            # Test the model
            classifier.start_predict()
            output = []
            for item in tqdm(valid_data_to_test):
                predicted_answers = classifier.predict(item)
                out = {}
                out['query'] = item
                out['valid_annotations'] = True
                out['annotator'] = 'lstm'
                out['date'] = str(datetime.datetime.now())
                out['annotated_answers'] = predicted_answers
                output.append(out)
            results = compute_metrics('lstm', type_prediction, args.db, output, gold_valid_data)
            results['lstm_hiddenunits'] = hidden_units
            results['lstm_dropout'] = dropout
            # Store the output
            suf = '-lstm-hid-' + str(hidden_units) + '-dropout-' + str(dropout)
            results_filename = out_dir + get_filename_results(args.db, args.model, "valid", args.topk, type_prediction, suf)
            fout = open(results_filename, 'wt')
            json.dump(results, fout)
            fout.close()

def tune_mlp_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir):
    for hidden_units in mlp_nhid:
        for dropout in mlp_dropout:
            print("Test {} MLP with nhidden={} dropout={}".format(type_prediction, hidden_units, dropout))
            hyper_params = {"n_units": hidden_units, "dropout": dropout}
            classifier = Classifier_MLP_Multi(dataset, type_prediction, args.result_dir, embedding_model, hyper_params, None)

            # Create training data
            td = classifier.create_training_data(training_data)

            # Train a model
            classifier.train(td, None, None)

            # Test the model
            classifier.start_predict()
            output = []
            for item in tqdm(valid_data_to_test):
                predicted_answers = classifier.predict(item)
                out = {}
                out['query'] = item
                out['valid_annotations'] = True
                out['annotator'] = 'mlp_multi'
                out['date'] = str(datetime.datetime.now())
                out['annotated_answers'] = predicted_answers
                output.append(out)
            results = compute_metrics('mlp_multi', type_prediction, args.db, output, gold_valid_data)
            results['mlp_multi_hiddenunits'] = hidden_units
            results['mlp_multi_dropout'] = dropout
            # Store the stats
            suf = '-mlp-multi-hid-' + str(hidden_units) + '-dropout-' + str(dropout)
            results_filename = out_dir + get_filename_results(args.db, args.model, "valid", args.topk, type_prediction, suf)
            fout = open(results_filename, 'wt')
            json.dump(results, fout)
            fout.close()


# Load dataset
dataset = None
if args.db == 'fb15k237':
    dataset = Dataset_FB15k237()
else:
    raise Exception("DB {} not supported!".format(args.db))

# Load embedding model
embedding_model_typ = args.model
embedding_model = Embedding_Model(args.result_dir, embedding_model_typ, dataset)

out_dir = args.result_dir + '/' + args.db + '/paramtuning/'
for type_prediction in ['head', 'tail']:
    # Load some answers (*from the training set*. This is data annotated under closed world assumption)
    annotations_filename = get_filename_answer_annotations(args.db, args.model, 'train', args.topk, type_prediction)
    annotations_path = args.result_dir + '/' + args.db + '/annotations/' + annotations_filename
    queries_with_answers = pickle.load(open(annotations_path, 'rb'))

    # Split them between train and validation set
    n_data_points = len(queries_with_answers)
    training_data, valid_data = train_test_split(queries_with_answers, test_size=use_valid_data)
    gold_valid_data = {} # This is the format used for computing the metrics
    valid_data_to_test = [] # This stores the content of the valid dataset in a format that is readable as input by the classifiers
    for v in valid_data:
        ent = v['query']['ent']
        rel = v['query']['rel']
        typ = v['query']['type']
        gold_valid_data[(ent, rel)] = [a for a in v['annotated_answers']]
        valid_data_to_test.append({ 'ent' : ent, 'rel' : rel, 'type' : typ, 'answers_fil' : [a for a in v['annotated_answers']]})

    # 1- Subgraph classifier
    if tune_sub:
        tune_sub_classifier(type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir)

    # 2- LSTM-based classifier
    if tune_lstm:
        tune_lstm_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir)

    # 3- MLP-based classifier
    if tune_mlp_multi:
        tune_mlp_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir)