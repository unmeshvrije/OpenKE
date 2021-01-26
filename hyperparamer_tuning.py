import argparse
from support.embedding_model import Embedding_Model
from support.dataset_fb15k237 import Dataset_FB15k237
from support.dataset_dbpedia50 import Dataset_dbpedia50
from classifier_subgraphs import Classifier_Subgraphs
import pickle
import json
from support.utils import *
from sklearn.model_selection import train_test_split
import datetime
from classifier_lstm import Classifier_LSTM
from classifier_mlp_multi import Classifier_MLP_Multi
from classifier_conv import Classifier_Conv
from classifier_snorkel import Classifier_Snorkel
from classifier_supensemble import Classifier_SuperEnsemble
from classifier_threshold import Classifier_Threshold
from classifier_squid import Classifier_Squid

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237','dbpedia50'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--tune_lstm', dest='tune_lstm', type=bool, default=False)
    parser.add_argument('--tune_mlp_multi', dest='tune_mlp_multi', type=bool, default=False)
    parser.add_argument('--tune_sub', dest='tune_sub', type=bool, default=False)
    parser.add_argument('--tune_conv', dest='tune_conv', type=bool, default=False)
    parser.add_argument('--tune_snorkel', dest='tune_snorkel', type=bool, default=False)
    parser.add_argument('--tune_squid', dest='tune_squid', type=bool, default=False)
    parser.add_argument('--do_ablation_study', dest='do_ablation_study', type=bool, default=False)
    parser.add_argument('--test_different_k', dest='test_different_k', type=bool, default=False)
    parser.add_argument('--test_threshold_different_k', dest='test_threshold_different_k', type=bool, default=False)
    return parser.parse_args()

args = parse_args()
tune_sub = args.tune_sub
tune_lstm = args.tune_lstm
tune_mlp_multi = args.tune_mlp_multi
tune_conv = args.tune_conv
tune_snorkel = args.tune_snorkel
tune_squid = args.tune_squid
do_ablation_study = args.do_ablation_study
test_different_k = args.test_different_k
test_threshold_different_k = args.test_threshold_different_k

# ***** PARAMS TO TUNE *****
sub_ks = [ 1, 3, 5, 10, 25, 50, 100, 1000 ]
lstm_nhid = [ 10, 100, 200 ]
lstm_dropout = [ 0, 0.1, 0.2 ]
mlp_nhid = [ 10, 100, 200 ]
mlp_dropout = [ 0, 0.1, 0.2 ]
conv_k1 = [ 2, 3, 4, 6]
conv_k2 = [ 1, 2, 3, 4]
snorkel_tau = [ (0.01, 0.1), (0.05, 0.2), (0.01, 0.6), (0.05,0.6), (0.1, 0.6), (0.2,0.6), (0.2,0.7), (0.2,0.8), (0.3,0.6), (0.3,0.7), (0.3,0.8) ]
snorkel_classifiers = [ 'mlp_multi','lstm','conv','path','sub' ]
ks = [1, 2, 3, 5, 10]

def tune_sub_classifier(type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir):
    for sub_k in sub_ks:
        print("Test {} sub with k={}".format(type_prediction, sub_k))
        classifier = Classifier_Subgraphs(dataset, type_prediction, embedding_model, args.result_dir, sub_k)
        output = []
        for item in tqdm(valid_data_to_test):
            predicted_answers = classifier.predict(item, 'answers_fil')
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
        answers_annotations_filename = out_dir + get_filename_results(args.db, args.model, "valid", args.topk, type_prediction, suf)
        with open(answers_annotations_filename, 'wt') as fout:
            json.dump(results, fout)
            fout.close()

def tune_conv_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir):
    for k1 in conv_k1:
        for k2 in conv_k2:
            if k2 >= k1:
                continue
            print("Test {} CONV with kernsize1={} kernsize2={}".format(type_prediction, k1, k2))
            hyper_params = {"kernel_size1": k1, "kernel_size2": k2, "topk" : args.topk}
            classifier = Classifier_Conv(dataset, type_prediction, args.result_dir, embedding_model, args.topk, hyper_params, None)

            # Create training data
            td = classifier.create_training_data(training_data)

            # Train a model
            classifier.train(td, None, None)

            # Test the model
            classifier.start_predict()
            output = []
            for item in tqdm(valid_data_to_test):
                predicted_answers = classifier.predict(item, 'answers_fil')
                out = {}
                out['query'] = item
                out['valid_annotations'] = True
                out['annotator'] = 'conv'
                out['date'] = str(datetime.datetime.now())
                out['annotated_answers'] = predicted_answers
                output.append(out)
            results = compute_metrics('conv', type_prediction, args.db, output, gold_valid_data)
            results['conv_k1'] = k1
            results['conv_k2'] = k2
            # Store the output
            suf = '-conv-k1-' + str(k1) + '-k2-' + str(k2)
            results_filename = out_dir + get_filename_results(args.db, args.model, "valid", args.topk, type_prediction, suf)
            fout = open(results_filename, 'wt')
            json.dump(results, fout)
            fout.close()

def tune_snorkel_squid_classifier(is_snorkel, training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data_with_queries, gold_valid_data, out_dir):
    if is_snorkel:
        name = "SNORKEL"
    else:
        name = "SQUID"
    for l1, h1 in snorkel_tau:
        for l2, h2 in snorkel_tau:
            for l3, h3 in snorkel_tau:
                print("Test {} {} with l={},{},{} h={},{},{}".format(type_prediction, name, l1, l2, l3, h1, h2, h3))
                abstain_scores = []
                abstain_scores.append((l1, h1))
                abstain_scores.append((l2, h2))
                abstain_scores.append((l3, h3))
                abstain_scores.append((0, 0.5))
                abstain_scores.append((0, 0.5))
                if is_snorkel:
                    classifier = Classifier_Snorkel(dataset, type_prediction,  args.topk, args.result_dir, snorkel_classifiers, args.model, model_path=None, abstain_scores=abstain_scores)
                else:
                    classifier = Classifier_Squid(dataset, type_prediction, args.topk, args.result_dir,
                                                    snorkel_classifiers, args.model, model_path=None,
                                                    abstain_scores=abstain_scores)

                # Create training data
                td = classifier.create_training_data(training_data, valid_dataset=gold_valid_data_with_queries)

                # Train a model
                classifier.train(td, None, None)

                # Test the model
                classifier.start_predict()
                output = []
                for item in tqdm(valid_data_to_test):
                    predicted_answers = classifier.predict(item, 'answers_fil', provenance_test="test")
                    out = {}
                    out['query'] = item
                    out['valid_annotations'] = True
                    out['annotator'] = classifier.get_name()
                    out['date'] = str(datetime.datetime.now())
                    out['annotated_answers'] = predicted_answers
                    output.append(out)
                results = compute_metrics(classifier.get_name(), type_prediction, args.db, output, gold_valid_data)
                results['snorkel_l1'] = l1
                results['snorkel_h1'] = h1
                results['snorkel_l2'] = l2
                results['snorkel_h2'] = h2
                results['snorkel_l3'] = l3
                results['snorkel_h3 '] = h3
                # Store the output
                suf = '-{}-k1-{}-{}-{}-{}-{}-{}'.format(name, l1, h1, l2, h2, l3, h3)
                results_filename = out_dir + get_filename_results(args.db, args.model, "valid", args.topk, type_prediction, suf)
                fout = open(results_filename, 'wt')
                json.dump(results, fout)
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
                predicted_answers = classifier.predict(item, 'answers_fil')
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
                predicted_answers = classifier.predict(item, 'answers_fil')
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

def do_ablation_study_snorkel(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir):
    abstain_scores = []
    abstain_scores.append((0.2, 0.6))
    abstain_scores.append((0.2, 0.6))
    abstain_scores.append((0.2, 0.6))
    abstain_scores.append((0, 0.5))
    abstain_scores.append((0, 0.5))
    for i in range(len(snorkel_classifiers)):
        classifiers = []
        a_scores = []
        excluded = None
        for j in range(len(snorkel_classifiers)):
            if j != i:
                classifiers.append(snorkel_classifiers[j])
                a_scores.append(abstain_scores[j])
            else:
                excluded = snorkel_classifiers[j]
        print("Test {} SNORKEL with classifiers {}".format(type_prediction, classifiers))


        classifier = Classifier_Snorkel(dataset, type_prediction, args.topk, args.result_dir, classifiers,
                                        args.model, model_path=None, abstain_scores=a_scores)
        # Create training data
        td = classifier.create_training_data(training_data)
        # Train a model
        classifier.train(td, None, None)
        # Test the model
        classifier.start_predict()
        output = []
        for item in tqdm(valid_data_to_test):
            predicted_answers = classifier.predict(item, 'answers_fil')
            out = {}
            out['query'] = item
            out['valid_annotations'] = True
            out['annotator'] = 'snorkel'
            out['date'] = str(datetime.datetime.now())
            out['annotated_answers'] = predicted_answers
            output.append(out)
        results = compute_metrics('snorkel', type_prediction, args.db, output, gold_valid_data)
        results['snorkel_classifiers'] = str(classifiers)
        # Store the output
        suf = '-ablation-no-{}'.format(excluded)
        results_filename = out_dir + get_filename_results(args.db, args.model, "valid", args.topk, type_prediction, suf)
        fout = open(results_filename, 'wt')
        json.dump(results, fout)
        fout.close()

    print("Test {} SNORKEL with classifiers {}".format(type_prediction, snorkel_classifiers))

    classifier = Classifier_Snorkel(dataset, type_prediction, args.topk, args.result_dir, snorkel_classifiers,
                                    args.model, model_path=None, abstain_scores=abstain_scores)
    # Create training data
    td = classifier.create_training_data(training_data)
    # Train a model
    classifier.train(td, None, None)
    # Test the model
    classifier.start_predict()
    output = []
    for item in tqdm(valid_data_to_test):
        predicted_answers = classifier.predict(item, 'answers_fil')
        out = {}
        out['query'] = item
        out['valid_annotations'] = True
        out['annotator'] = 'snorkel'
        out['date'] = str(datetime.datetime.now())
        out['annotated_answers'] = predicted_answers
        output.append(out)
    results = compute_metrics('snorkel', type_prediction, args.db, output, gold_valid_data)
    results['snorkel_classifiers'] = str(snorkel_classifiers)
    # Store the output
    suf = '-ablation-all'
    results_filename = out_dir + get_filename_results(args.db, args.model, "valid", args.topk, type_prediction, suf)
    fout = open(results_filename, 'wt')
    json.dump(results, fout)
    fout.close()

def test_with_different_k(type_prediction, args, gold_valid_data, out_dir):
    # Load annotations
    suf = '-snorkel'
    answers_annotations_filename = args.result_dir + '/' + args.db + '/annotations/' + get_filename_answer_annotations(
        args.db, args.model, 'test', args.topk, type_prediction, suf)
    with open(answers_annotations_filename, 'rb') as fin:
        annotated_answers = pickle.load(fin)

    for k in ks:
        print("Test {} SNORKEL with k={}".format(type_prediction, k))
        results = compute_metrics('snorkel', type_prediction, args.db, annotated_answers, gold_valid_data, subset_k=k)
        results['ablation-k'] = k
        # Store the output
        suf = '-ablation-k-{}'.format(k)
        results_filename = out_dir + get_filename_results(args.db, args.model, "test", args.topk, type_prediction, suf)
        fout = open(results_filename, 'wt')
        json.dump(results, fout)
        fout.close()

def test_threshold_with_different_k(type_prediction, args, gold_valid_data, out_dir):
    for k in ks:
        print("Test {} THREHOLD with k={}".format(type_prediction, k))
        classifier = Classifier_Threshold(dataset, type_prediction, args.result_dir, k)
        output = []
        for item in tqdm(valid_data_to_test):
            predicted_answers = classifier.predict(item, 'answers_fil')
            out = {}
            out['query'] = item
            out['valid_annotations'] = True
            out['annotator'] = 'threshold'
            out['date'] = str(datetime.datetime.now())
            out['annotated_answers'] = predicted_answers
            output.append(out)
        results = compute_metrics('threshold', type_prediction, args.db, output, gold_valid_data)
        results['threshold-k'] = k
        # Store the output
        suf = '-threshold-k-' + str(k)
        answers_annotations_filename = out_dir + get_filename_answer_annotations(args.db, args.model, 'valid', args.topk, type_prediction, suf)
        with open(answers_annotations_filename, 'wb') as fout:
            pickle.dump(output, fout)
            fout.close()

# Load dataset
dataset = None
if args.db == 'fb15k237':
    dataset = Dataset_FB15k237()
elif args.db == 'dbpedia50':
    dataset = Dataset_dbpedia50()
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

    # Load test answers (used only for the ablation study)
    suf = ''
    test_answers_filename = args.result_dir + '/' + args.db + '/answers/' + get_filename_answers(args.db, args.model,
                                                                                                 "test", args.topk,
                                                                                                 type_prediction,
                                                                                                 suf)
    test_queries_with_answers = pickle.load(open(test_answers_filename, 'rb'))

    # Load the gold standard
    gold_dir = args.result_dir + '/' + args.db + '/annotations/'
    gold_filename = get_filename_gold(args.db, args.topk, '-valid')
    gold_valid_data_with_queries = {}
    with open(gold_dir + gold_filename, 'rt') as fin:
        gold_annotations = json.load(fin)
    filter_queries = {}
    if type_prediction == 'head':
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
            assert (len(ans) == args.topk)
            filter_queries[(ent, rel)] = ans
            gold_valid_data_with_queries[(ent, rel)] = item

    training_data = queries_with_answers
    gold_valid_data = filter_queries # This format is used to compute the metrics
    valid_data_to_test = [] # This stores the content of the valid dataset in a format that is readable as input by the classifiers
    for v in test_queries_with_answers:
        ent = v['ent']
        rel = v['rel']
        typ = v['type']
        if typ == accepted_query_type and (ent, rel) in filter_queries:
            valid_data_to_test.append(v)

    # 1- Subgraph classifier
    if tune_sub:
        tune_sub_classifier(type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir)

    # 2- LSTM-based classifier
    if tune_lstm:
        tune_lstm_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir)

    # 3- MLP-based classifier
    if tune_mlp_multi:
        tune_mlp_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir)

    # 4- Conv-based classifier
    if tune_conv:
        tune_conv_classifier(training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data, out_dir)

    # 5- Snorkel-based classifier
    if tune_snorkel:
        tune_snorkel_squid_classifier(True, training_data, type_prediction, dataset, embedding_model, args, valid_data_to_test, gold_valid_data_with_queries, gold_valid_data, out_dir)

    # 6- Snorkel ablation study
    if do_ablation_study:
        do_ablation_study_snorkel(training_data, type_prediction, dataset, embedding_model, args, test_queries_with_answers, gold_valid_data, out_dir)

    #7- Test different snorkel with ks
    if test_different_k:
        test_with_different_k(type_prediction, args, filter_queries, out_dir)

    #8- Test threshold classifier with different ks
    if test_threshold_different_k:
        test_threshold_with_different_k(type_prediction, args, gold_valid_data, out_dir)

    #9- SQUID-based classifier
    if tune_squid:
        tune_snorkel_squid_classifier(False, training_data, type_prediction, dataset, embedding_model, args,
                                      valid_data_to_test, gold_valid_data_with_queries, gold_valid_data, out_dir)