import os
import pickle
import argparse
from sklearn.metrics import classification_report
from mlp_classifier import MLPClassifier
import numpy as np
import copy

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--true-out', dest ='true_out_file', type = str, help = 'File containing the true /expected answers.',
    default = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-transe-annotated-topk-10-tail.out')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--trainfile', dest ='train_file', type = str, help = 'File containing training triples.')
    parser.add_argument('--modelfile', dest ='model_file',type = str, help = 'File containing the model.')
    parser.add_argument('--classifier', dest ='classifier', required = True, type = str, help = 'Classifier LSTM/MLP.')
    parser.add_argument('--weightsfile', dest ='weights_file', type = str, help = 'File containing the model weights.')
    parser.add_argument('--embfile', dest ='emb_file', type = str, help = 'File containing entity embeddings.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--abs-low', required = True, dest = 'abs_low', type = float, default = 0.2)
    parser.add_argument('--abs-high', required = True, dest = 'abs_high', type = float, default = 0.6)
    parser.add_argument('--model', dest ='model',type = str, default = "transe", help = 'Embedding model name.')
    parser.add_argument('--pred', dest ='pred', type = str, required = True, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    return parser.parse_args()

def r2(num):
    return np.round(num, 2)

def get_results(y_true, y_predicted):
    result = classification_report(y_true, y_predicted, output_dict = True)
    return  result['1']['precision'], result['1']['recall'], result['1']['f1-score'], result['accuracy']

def get_results_str(y_true, y_predicted):
    result = classification_report(y_true, y_predicted, output_dict = True)
    return  "Precision = " + str(r2(result['1']['precision'])) + "," +\
            "Recall = "+str(r2(result['1']['recall']))         + "," +\
            "F1 score = "+str(r2(result['1']['f1-score']))         + "," +\
            "Accuracy(overall) = "+str(r2(result['accuracy']))

args = parse_args()

result_dir =  args.result_dir + args.db + "/out/"
log_dir =  args.result_dir + args.db + "/logs/"
os.makedirs(result_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
queries_file_path = args.test_file

model_file_path = args.model_file
model_weights_path = args.weights_file
# For this to work, queries_file_path must contain 10 (topk) answers present for each triple
myc = MLPClassifier(args.pred, args.db, args.topk, queries_file_path, args.model,
    model_file_path, model_weights_path, abs_low = float(args.abs_low), abs_high = float(args.abs_high))

myc.predict()
raw_result, fil_result = myc.results()

Y = np.array(fil_result['predicted_y'])
'''
    4. Extract true/gold y/label values from annotation file
'''
len_y  = len(Y)
indexes = []
true_y = np.empty(len_y, dtype = np.int)
# TODO: fill with -2 and include -1 as ABSTAIN label ?
true_y.fill(-1);
with open(args.true_out_file) as fin:
    lines = fin.readlines()
    for i, label in enumerate(lines):
        if label == "\n" or int(label) == -1:
            continue
        true_y[i] = int(label)

indexes_annotated = np.where(true_y != -1)[0]
true_annotated  = true_y[indexes_annotated]

results = {}
for low in [0.1, 0.2, 0.3, 0.4]:
    for high in [0.6, 0.7, 0.8, 0.9]:
        myc = MLPClassifier(args.pred, args.db, args.topk, queries_file_path, args.model,
            model_file_path, model_weights_path, abs_low = float(low), abs_high = float(high))

        myc.predict()
        raw_result, fil_result = myc.results()
        Y = np.array(fil_result['predicted_y_abs'])
        Y_annotated     = Y[indexes_annotated]
        a,b,c,d = get_results(true_annotated, Y_annotated)
        print("ASFSDF : ", r2(a), r2(b), r2(c), r2(d))
        results[(low, high)] = (r2(a), r2(b), r2(c), r2(d))

for k in results.keys():
    print(args.classifier,"(", k, ") : ", results[k])
