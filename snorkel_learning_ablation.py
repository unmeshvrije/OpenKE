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
import itertools

logging.basicConfig(level = logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--true-out', dest ='true_out_file', type = str, help = 'File containing the true /expected answers.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-transe-annotated-topk-10-tail.out')
    parser.add_argument('--lstm-out', dest ='lstm_out_file', type = str, help = 'File containing the output of lstm classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-transe-training-topk-10-tail-model-lstm-units-100-dropout-0.2.out')
    parser.add_argument('--mlp-out', dest ='mlp_out_file', type = str, help = 'File containing the output of mlp classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-transe-training-topk-10-tail-model-mlp-units-100-dropout-0.2.out')
    parser.add_argument('--path-out', dest ='path_out_file', type = str, help = 'File containing the output of path classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-path-classifier-tail.out')
    parser.add_argument('--sub-out', dest ='sub_out_file', type = str, help = 'File containing the output of sub classifier.',
    default = '/var/scratch2/xxx/OpenKE-results/fb15k237/out/fb15k237-transe-subgraphs-tau-10-tail.out')
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = '/var/scratch2/xxx/OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/xxx/OpenKE-results/",help = 'Output dir.')
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
os.makedirs(result_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)

'''
    2. Load pickle files for results of classifiers and ent/rel dictionaries
'''
def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

entity_dict     = load_pickle(args.ent_dict)
relation_dict   = load_pickle(args.rel_dict)
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
indexes = []
true_y = np.empty(len_y, dtype = np.int)
# TODO: fill with -2 and include -1 as ABSTAIN label ?
true_y.fill(-1);
with open(args.true_out_file) as fin:
    lines = fin.readlines()
    for i, label in enumerate(lines):
        if label == "\n" or int(label) == -1:
            continue
        indexes.append(i)
        true_y[i] = int(label)

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

vs = np.vstack((lstm_y, mlp_y, path_y, sub_y))
#sums = np.sum(vs, axis = 0)
sums = np.apply_along_axis(count_ones, 0, vs)
max_voting_y = (sums > 2).astype(int)
# TODO: check with abstain
min_voting_y = (sums > 0).astype(int)

'''
    function that accepts 4 classifiers y labels
    and annotated indexes, fills the out array with labels at those indexes
'''
def get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated, labels = {'lstm': True, 'mlp': True, 'sub' : True, 'path': True}):
    snorkel_y = np.empty(len(lstm_y), dtype = np.int)
    snorkel_y.fill(-1);

    kf = KFold(n_splits = 3, shuffle = False, random_state = 42)
    #kf.split(indexes_annotated)

    '''
    np.random.seed(24)
    np.random.shuffle(indexes_annotated)
    N_SPLIT = int(len(indexes_annotated)* 0.9)
    print("SPLIT: ", N_SPLIT)
    indexes_annotated_train = indexes_annotated[:N_SPLIT]
    indexes_annotated_test  = indexes_annotated[N_SPLIT:]
    '''
    max_accuracy = 0.0
    L_test_max = None
    indexes_annotated_test_max = None
    best_model = None
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

        train_feature_list = []
        if labels['lstm']:
            train_feature_list.append(lstm_annotated_train)
        if labels['mlp']:
            train_feature_list.append(mlp_annotated_train)
        if labels['sub']:
            train_feature_list.append(sub_annotated_train)
        if labels['path']:
            train_feature_list.append(path_annotated_train)

        L_train = np.transpose(np.vstack(tuple(train_feature_list)))
        label_model.fit(L_train, n_epochs=500, optimizer="adam")

        test_feature_list = []
        if labels['lstm']:
            test_feature_list.append(lstm_annotated_test)
        if labels['mlp']:
            test_feature_list.append(mlp_annotated_test)
        if labels['sub']:
            test_feature_list.append(sub_annotated_test)
        if labels['path']:
            test_feature_list.append(path_annotated_test)

        L_test = np.transpose(np.vstack(tuple(test_feature_list)))
        cv_accuracy = label_model.score(L = L_test, Y = true_annotated_test, tie_break_policy = "random")["accuracy"]
        if cv_accuracy > max_accuracy:
            max_accuracy = cv_accuracy
            L_test_max = L_test
            best_model = label_model
            indexes_annotated_test_max = indexes_annotated_test
            out_y = label_model.predict(L_test, tie_break_policy="random")
            for i,index in enumerate(indexes_annotated_test):
                snorkel_y[index] = out_y[i]

    '''
    apply best model to entire annotated set
    '''
    lstm_annotated= lstm_y[indexes_annotated]
    mlp_annotated= mlp_y[indexes_annotated]
    sub_annotated= sub_y[indexes_annotated]
    path_annotated= path_y[indexes_annotated]

    test_feature_list = []
    if labels['lstm']:
        test_feature_list.append(lstm_annotated)
    if labels['mlp']:
        test_feature_list.append(mlp_annotated)
    if labels['sub']:
        test_feature_list.append(sub_annotated)
    if labels['path']:
        test_feature_list.append(path_annotated)
    L_test_max = np.transpose(np.vstack(tuple(test_feature_list)))
    out_y = best_model.predict(L_test_max, tie_break_policy = "random")
    for i,index in enumerate(indexes_annotated):
        snorkel_y[index] = out_y[i]
    indexes_annotated_test_max = indexes_annotated

    return out_y, snorkel_y, indexes_annotated_test_max, L_test_max


def get_voter_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated, voter):
    voter_y = np.empty(len(lstm_y), dtype = np.int)
    voter_y.fill(-1);
    lstm_annotated_test = lstm_y[indexes_annotated]
    mlp_annotated_test  = mlp_y [indexes_annotated]
    sub_annotated_test  = sub_y [indexes_annotated]
    path_annotated_test = path_y[indexes_annotated]
    L_test = np.transpose(np.vstack((lstm_annotated_test, mlp_annotated_test, sub_annotated_test, path_annotated_test)))
    my_voter = voter()
    out_y = my_voter.predict(L_test, tie_break_policy="random")
    for i,index in enumerate(indexes_annotated):
        voter_y[index] = out_y[i]
    return out_y, voter_y

snorkel_y_annotated, snorkel_y, indexes_annotated_test, L_test = get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated)

withoutC4 = {
    'labels' : {
    'lstm' : True,
    'mlp' : True,
    'sub' : True,
    'path' : False
    },
    'snorkel_y_annotated': None,
    'snorkel_y' : None
}
withoutC4["snorkel_y_annotated"], withoutC4["snorkel_y"], indexes_annotated_test, L_test = get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated, withoutC4['labels'])

withoutC3 = {
    'labels' : {
    'lstm' : True,
    'mlp' : True,
    'sub' : False,
    'path' : True
    },
    'snorkel_y_annotated': None,
    'snorkel_y' : None
}
withoutC3["snorkel_y_annotated"], withoutC3["snorkel_y"], indexes_annotated_test, L_test = get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated, withoutC3['labels'])

withoutC2 = {
    'labels' : {
    'lstm' : True,
    'mlp' : False,
    'sub' : True,
    'path' : True
    },
    'snorkel_y_annotated': None,
    'snorkel_y' : None
}
withoutC2["snorkel_y_annotated"], withoutC2["snorkel_y"], indexes_annotated_test, L_test = get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated, withoutC2['labels'])

withoutC1 = {
    'labels' : {
    'lstm' : False,
    'mlp' : True,
    'sub' : True,
    'path' : True
    },
    'snorkel_y_annotated': None,
    'snorkel_y' : None
}
withoutC1["snorkel_y_annotated"], withoutC1["snorkel_y"], indexes_annotated_test, L_test = get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated, withoutC1['labels'])

major_y_annotated, major_y = get_voter_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated_test, MajorityLabelVoter)
random_y_annotated, random_y = get_voter_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated_test, RandomVoter)

lstm_annotated_test = lstm_y[indexes_annotated_test]
mlp_annotated_test  = mlp_y[indexes_annotated_test]
sub_annotated_test  = sub_y[indexes_annotated_test]
path_annotated_test = path_y[indexes_annotated_test]
maxv_annotated = max_voting_y[indexes_annotated_test]
minv_annotated = min_voting_y[indexes_annotated_test]
true_annotated = true_y[indexes_annotated_test]

def r2(num):
    return np.round(num, 2)

def get_results(y_true, y_predicted):
    #conf = confusion_matrix(y_true, y_predicted)
    result = classification_report(y_true, y_predicted, output_dict = True)
    #print ("Accuracy score: ", accuracy_score(y_true, y_predicted))
    return  "Precision = " + str(r2(result['1']['precision'])) + "," +\
            "Recall = "+str(r2(result['1']['recall']))         + "," +\
            "F1 score = "+str(r2(result['1']['f1-score']))         + "," +\
            "Accuracy(overall) = "+str(r2(result['accuracy']))

baseline = np.empty(len(true_annotated), dtype = np.int)
baseline.fill(1)
print("baseline  : ", get_results(true_annotated, baseline))
print("lstm  : ", get_results(true_annotated, lstm_annotated_test))
print("mlp   : ", get_results(true_annotated, mlp_annotated_test))
print("path  : ", get_results(true_annotated, path_annotated_test))
print("sub   : ", get_results(true_annotated, sub_annotated_test))
print("minv  : ", get_results(true_annotated, minv_annotated))
print("major : ", get_results(true_annotated, major_y_annotated))
print("maxv  : ", get_results(true_annotated, maxv_annotated))
print("random: ", get_results(true_annotated, random_y_annotated))
print("snork      : ", get_results(true_annotated, snorkel_y_annotated))
print("snork(-C1) : ", get_results(true_annotated, withoutC1["snorkel_y_annotated"]))
print("snork(-C2) : ", get_results(true_annotated, withoutC2["snorkel_y_annotated"]))
print("snork(-C3) : ", get_results(true_annotated, withoutC3["snorkel_y_annotated"]))
print("snork(-C4) : ", get_results(true_annotated, withoutC4["snorkel_y_annotated"]))

test_queries = load_pickle(args.test_file)
x_test_fil = np.array(test_queries['x_' + args.pred + "_fil"])
logfile = log_dir + args.model + "-" + args.db + "-" + args.pred + "-ensembled.log"
print("logfile = ", logfile)
with open(logfile, "w") as log:
    for index, x in enumerate(tqdm(x_test_fil)):
        if index not in indexes_annotated_test:
            continue
        e = int(x[0])
        r = int(x[1])
        a = int(x[2])
        head = e
        tail = a
        if args.pred == "head":
            head = a
            tail = e
        print("{}, {}, {},LSTM:{},MLP:{},PATH:{},SUB:{},maxV:{},Snorkel:{},REAL:{}".format(entity_dict[head], relation_dict[r], entity_dict[tail], lstm_y[index], mlp_y[index], path_y[index], sub_y[index], max_voting_y[index], snorkel_y[index], true_y[index]), file = log)
        if (index+1) % args.topk == 0:
            print("*" * 80, file = log)

