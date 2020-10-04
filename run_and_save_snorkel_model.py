import os
import sys
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model.baselines import RandomVoter
import logging
import json
from sklearn import svm
import random

logging.basicConfig(level = logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--true-out', dest ='true_out_file', type = str, help = 'File containing the true /expected answers.',
    default = 'OpenKE-results/fb15k237/out/fb15k237-transe-annotated-topk-10-tail.out')
    parser.add_argument('--true-out-extended', dest='true_out_ext_file', type=str, help='File containing the true /expected answers in JSON extended format.', default=None)
    parser.add_argument('--lstm-out', dest ='lstm_out_file', type = str, help = 'File containing the output of lstm classifier.',
    default = 'OpenKE-results/fb15k237/out/fb15k237-transe-training-topk-10-tail-model-lstm-units-100-dropout-0.2.out')
    parser.add_argument('--mlp-out', dest ='mlp_out_file', type = str, help = 'File containing the output of mlp classifier.',
    default = 'OpenKE-results/fb15k237/out/fb15k237-transe-training-topk-10-tail-model-mlp-units-100-dropout-0.2.out')
    parser.add_argument('--path-out', dest ='path_out_file', type = str, help = 'File containing the output of path classifier.',
    default = 'OpenKE-results/fb15k237/out/fb15k237-path-classifier-tail.out')
    parser.add_argument('--sub-out', dest ='sub_out_file', type = str, help = 'File containing the output of sub classifier.',
    default = 'OpenKE-results/fb15k237/out/fb15k237-transe-subgraphs-tau-10-tail.out')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--abstain', dest = 'abstain', default = False, action = 'store_true')
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--model', dest ='model',type = str, default = "transe", help = 'Embedding model name.')
    parser.add_argument('--pred', dest ='pred', type = str, required = True, choices = ['head', 'tail'], help = 'Prediction type (head/tail)', default='tail')
    return parser.parse_args()

args = parse_args()

'''
    1. Setup  out and log directories
'''
result_dir =  args.result_dir + args.db + "/out/"
log_dir =  args.result_dir + args.db + "/logs/"
models_dir = args.result_dir + args.db + "/models/"
os.makedirs(result_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
os.makedirs(models_dir, exist_ok = True)

'''
    2. Load pickle files for results of classifiers and ent/rel dictionaries
'''
def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

lstm_results    = load_pickle(args.lstm_out_file)
mlp_results     = load_pickle(args.mlp_out_file)
path_results    = load_pickle(args.path_out_file)
sub_results     = load_pickle(args.sub_out_file)

'''
    3. Extract the y/labelvalues for filtered setting
'''
y_label_str = "predicted_y"
if args.abstain:
    y_label_str += "_abs"
lstm_y = np.array(lstm_results['fil'][y_label_str])
mlp_y  = np.array(mlp_results['fil'][y_label_str])
path_y = np.array(path_results['fil'][y_label_str])
sub_y  = np.array(sub_results['fil'][y_label_str])

'''
    4. Extract true/gold y/label values from annotation file
'''
len_y  = len(lstm_y)
true_y = np.empty(len_y, dtype = np.int)
# TODO: fill with -2 and include -1 as ABSTAIN label ?
true_y.fill(-1);
if args.true_out_ext_file is None:
    with open(args.true_out_file) as fin:
        lines = fin.readlines()
        for i, label in enumerate(lines):
            if label == "\n" or int(label) == -1:
                continue
            true_y[i] = int(label)
else:
    queries = json.load(open(args.true_out_ext_file, 'rt'))
    #queries = random.sample(queries, k=200)
    #excludeQueryIds = []
    for queryId, query in enumerate(queries):
        #if queryId in excludeQueryIds:
        #    continue
        idx = query['idx']
        for j, answer in enumerate(query['answers']):
            true_y[idx + j] = answer['cor']

'''
    5. Find indexes of answer triples which are actually annotated
    For now, consider answers that are either labelled 0 or 1
'''
#indexes_annotated = np.array(indexes)
indexes_annotated = np.where(true_y != -1)[0]
print("# of annotated answers = ", len(indexes_annotated))

'''
    6. Compute simple max-voting and min-voting from 4 classifiers
'''
def count_ones(arr):
    ones = 0
    for a in arr:
        if a == 1:
            ones += 1
    return ones

'''
    function that accepts 4 classifiers y labels
    and annotated indexes, fills the out array with labels at those indexes
'''
def run_snorkel_model(lstm_y, mlp_y, sub_y, path_y, true_y, indexes_annotated):
    snorkel_y = np.empty(len(lstm_y), dtype = np.int)
    snorkel_y.fill(-1);

    kf = KFold(n_splits = 3, shuffle = False, random_state = 42)
    max_accuracy = 0.0
    best_model = None

    # JU: replacing cross validation with a larger training
    #best_model = LabelModel(verbose=False)
    #train_feature_list = [lstm_y, mlp_y, sub_y, path_y]
    #L_train = np.transpose(np.vstack(tuple(train_feature_list)))
    #best_model.fit(L_train, n_epochs=500, optimizer="adam")

    for train_split, test_split in kf.split(indexes_annotated):
        indexes_annotated_train = indexes_annotated[train_split]
        indexes_annotated_test  = indexes_annotated[test_split]

        lstm_annotated_test = lstm_y[indexes_annotated_test]
        mlp_annotated_test  = mlp_y[indexes_annotated_test]
        sub_annotated_test  = sub_y[indexes_annotated_test]
        path_annotated_test = path_y[indexes_annotated_test]
        true_annotated_test = true_y[indexes_annotated_test]

        lstm_annotated_train = lstm_y[indexes_annotated_train]
        mlp_annotated_train  = mlp_y[indexes_annotated_train]
        sub_annotated_train  = sub_y[indexes_annotated_train]
        path_annotated_train = path_y[indexes_annotated_train]
        true_annotated_train = true_y[indexes_annotated_train]

        label_model = LabelModel(verbose = False)
        train_feature_list = [lstm_annotated_train, mlp_annotated_train, sub_annotated_train, path_annotated_train]

        L_train = np.transpose(np.vstack(tuple(train_feature_list)))
        label_model.fit(L_train, Y_dev=true_annotated_train, n_epochs=500, optimizer="adam")

        test_feature_list = [lstm_annotated_test, mlp_annotated_test, sub_annotated_test, path_annotated_test]
        L_test = np.transpose(np.vstack(tuple(test_feature_list)))
        cv_accuracy = label_model.score(L = L_test, Y = true_annotated_test, tie_break_policy = "random")["accuracy"]
        if cv_accuracy > max_accuracy:
            max_accuracy = cv_accuracy
            best_model = label_model
            out_y = label_model.predict(L_test, tie_break_policy="random")
            for i,index in enumerate(indexes_annotated_test):
                snorkel_y[index] = out_y[i]

    # Write the best_model to disk
    a_str = ""
    if args.abstain:
        a_str = ".abs"
    snorkel_model_file_name = models_dir + args.db + "-" + args.model + "-" + args.pred + "-snorkel.model" + a_str
    print ("saving in ", snorkel_model_file_name)
    best_model.save(snorkel_model_file_name)


run_snorkel_model(lstm_y, mlp_y, sub_y, path_y, true_y, indexes_annotated)

