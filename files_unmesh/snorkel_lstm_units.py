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

logging.basicConfig(level = logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--true-out', dest ='true_out_file', type = str, help = 'File containing the true /expected answers.',
    default = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-complex-annotated-topk-10-tail.out')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--abstain', dest = 'abstain', default = False, action = 'store_true')
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--model', dest ='model',type = str, default = "complex", help = 'Embedding model name.')
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

lstm100 = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-complex-training-topk-10-tail-model-lstm-units-100-dropout-0.2.out'
mlp100  = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-complex-training-topk-10-tail-model-mlp-units-100-dropout-0.2.out'
lstm10 = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-complex-training-topk-10-tail-model-lstm-units-10-dropout-0.2.out'
mlp10  = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-complex-training-topk-10-tail-model-mlp-units-10-dropout-0.2.out'
lstm200 = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-complex-training-topk-10-tail-model-lstm-units-200-dropout-0.2.out'
mlp200  = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-complex-training-topk-10-tail-model-mlp-units-200-dropout-0.2.out'

lstm10_results  = load_pickle(lstm10)
lstm100_results = load_pickle(lstm100)
lstm200_results = load_pickle(lstm200)

mlp10_results  = load_pickle(mlp10)
mlp100_results = load_pickle(mlp100)
mlp200_results = load_pickle(mlp200)

'''
    3. Extract the y/labelvalues for filtered setting
'''
y_label_str = "predicted_y"
if args.abstain:
    y_label_str += "_abs"

lstm10_y = np.array(lstm10_results['fil'][y_label_str])
lstm100_y = np.array(lstm100_results['fil'][y_label_str])
lstm200_y = np.array(lstm200_results['fil'][y_label_str])

mlp10_y = np.array(mlp10_results['fil'][y_label_str])
mlp100_y = np.array(mlp100_results['fil'][y_label_str])
mlp200_y = np.array(mlp200_results['fil'][y_label_str])

'''
    4. Extract true/gold y/label values from annotation file
'''
len_y  = len(lstm10_y)
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


lstm10_annotated_test = lstm10_y[indexes_annotated]
lstm100_annotated_test = lstm100_y[indexes_annotated]
lstm200_annotated_test = lstm200_y[indexes_annotated]
mlp10_annotated_test =  mlp10_y[indexes_annotated]
mlp100_annotated_test = mlp100_y[indexes_annotated]
mlp200_annotated_test = mlp200_y[indexes_annotated]
true_annotated = true_y[indexes_annotated]


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

print("lstm 10", args.pred, " : ", get_results(true_annotated, lstm10_annotated_test))
print("lstm 100", args.pred, " : ", get_results(true_annotated, lstm100_annotated_test))
print("lstm 200", args.pred, " : ", get_results(true_annotated, lstm200_annotated_test))
print("mlp  10", args.pred, " : ", get_results(true_annotated, mlp10_annotated_test))
print("mlp  100", args.pred, " : ", get_results(true_annotated, mlp100_annotated_test))
print("mlp  200", args.pred, " : ", get_results(true_annotated, mlp200_annotated_test))
