import os
import sys
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model.baselines import RandomVoter
import logging

logging.basicConfig(level = logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--true-out', dest ='true_out_file', type = str, help = 'File containing the true /expected answers.',
    default = '/var/scratch2/uji300/OpenKE-results/fb15k237/out/fb15k237-transe-annotated-topk-10-tail.out')
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = '/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = '/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', type = int, default = 10)
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

models = ["transe", "rotate", "complex"]
filemap = {}
for m in models:
    filemap[m] = {}
    filemap[m]["true"]  = result_dir + args.db + "-" + m + "-annotated-topk-10-tail.out"
    filemap[m]["lstm"]  = result_dir + args.db + "-" + m + "-training-topk-10-tail-model-lstm-units-100-dropout-0.2.out"
    filemap[m]["mlp"]   = result_dir + args.db + "-" + m + "-training-topk-10-tail-model-mlp-units-100-dropout-0.2.out"
    filemap[m]["sub"]   = result_dir + args.db + "-" + m + "-subgraphs-tau-10-tail.out"
    filemap[m]["path"]  = result_dir + args.db           + "-path-classifier-tail.out"
    filemap[m]["test"]  = args.result_dir + args.db + "/data/" + args.db + "-" + m + "-test-topk-10.pkl"

classifiers = ["true", "lstm", "mlp", "sub", "path", "test"]

#for m in models:
#    for c in classifiers:
#        print(filemap[m][c], " : ", os.path.isfile(filemap[m][c]))

'''
    2. Load pickle files for results of classifiers and ent/rel dictionaries
'''
def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

entity_dict     = load_pickle(args.ent_dict)
relation_dict   = load_pickle(args.rel_dict)

results_map = {}
for m in models:
    results_map[m] = {}
    results_map[m]["lstm"] = load_pickle(filemap[m]["lstm"])
    results_map[m]["mlp"]  = load_pickle(filemap[m]["mlp"])
    results_map[m]["sub"]  = load_pickle(filemap[m]["sub"])
    results_map[m]["path"] = load_pickle(filemap[m]["path"])


'''
    3. Extract the y/labelvalues for filtered setting
'''
y_label_str = "predicted_y"
if args.abstain:
    y_label_str += "_abs"

y_map = {}
for m in models:
    y_map[m] = {}
    y_map[m]["lstm"] = np.array(results_map[m]["lstm"]['fil'][y_label_str])
    y_map[m]["mlp"]  = np.array(results_map[m]["mlp"]['fil'][y_label_str])
    y_map[m]["sub"]  = np.array(results_map[m]["sub"]['fil'][y_label_str])
    y_map[m]["path"] = np.array(results_map[m]["path"]['fil'][y_label_str])

'''
    4. Extract true/gold y/label values from annotation file
'''

len_y  = len(y_map["transe"]["lstm"])

for m in models:
    y_map[m]["true"] = np.empty(len_y, dtype = np.int)
    y_map[m]["true"].fill(-1);
    with open(filemap[m]["true"]) as fin:
        lines = fin.readlines()
        for i, label in enumerate(lines):
            if label == "\n" or int(label) == -1:
                continue
            y_map[m]["true"][i] = int(label)

'''
len_y  = len(y_map["transe"]["lstm"])

y_map["transe"]["true"] = np.empty(len_y, dtype = np.int)
y_map["transe"]["true"].fill(-1);
y_map["rotate"]["true"] = np.empty(len_y, dtype = np.int)
y_map["rotate"]["true"].fill(-1);
y_map["complex"]["true"] = np.empty(len_y, dtype = np.int)
y_map["complex"]["true"].fill(-1);

with open (filemap["transe"]["true"]) as f1, open(filemap["rotate"]["true"]) as f2, open(filemap["complex"]["true"]) as f3:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines3 = f3.readlines()
    for i, _ in enumerate(lines1):
        label1 = lines1[i].rstrip()
        label2 = lines2[i].rstrip()
        label3 = lines3[i].rstrip()
        row = label1 + " " + label2 + " " + label3
        if (len(row.split()) == 2) or len(row.split()) == 1:
            print (i+1, " : ", "transe : (",label1,")", "rotate:(",label2,")", "complex:(",label3,")" )
        elif label2 == "-1" and label3 == "-1" and label1 != "-1":
            print (i+1, " : ", "transe : (",label1,")", "rotate:(",label2,")", "complex:(",label3,")" )
        #print("_ "*20)
#TODO
sys.exit()
'''

'''
    5. Find indexes of answer triples which are actually annotated
    For now, consider answers that are either labelled 0 or 1
'''
indexes_annotated1 = np.where(y_map["transe"]["true"] != -1)[0]
indexes_annotated2 = np.where(y_map["rotate"]["true"] != -1)[0]
indexes_annotated3 = np.where(y_map["complex"]["true"] != -1)[0]
indexes_annotated = indexes_annotated1
assert((indexes_annotated1 == indexes_annotated2).all())
assert((indexes_annotated3 == indexes_annotated2).all())


'''
    function that accepts 4 classifiers y labels
    and annotated indexes, fills the out array with labels at those indexes
'''
def get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated):
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
        print("Splitting training and test annotated data: ")

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
        L_train = np.transpose(np.vstack((lstm_annotated_train, mlp_annotated_train, sub_annotated_train, path_annotated_train)))
        label_model.fit(L_train, Y_dev=true_annotated_train, n_epochs=500, optimizer="adam")
        L_test = np.transpose(np.vstack((lstm_annotated_test, mlp_annotated_test, sub_annotated_test, path_annotated_test)))
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
    L_test_max = np.transpose(np.vstack((lstm_annotated, mlp_annotated, sub_annotated, path_annotated)))
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

def r2(num):
    return np.round(num, 2)

def get_results(y_true, y_predicted):
    #conf = confusion_matrix(y_true, y_predicted)
    result = classification_report(y_true, y_predicted, output_dict = True)
    return  "Precision = " + str(r2(result['1']['precision'])) + "," +\
            "Recall = "+str(r2(result['1']['recall']))         + "," +\
            "F1 score = "+str(r2(result['1']['f1-score']))         + "," +\
            "Accuracy(overall) = "+str(r2(result['accuracy']))

'''
snorkel_y_annotated, snorkel_y, indexes_annotated_test, L_test = get_snorkel_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated)
major_y_annotated, major_y = get_voter_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated_test, MajorityLabelVoter)
random_y_annotated, random_y = get_voter_labels(lstm_y, mlp_y, sub_y, path_y, indexes_annotated_test, RandomVoter)

lstm_annotated_test = lstm_y[indexes_annotated_test]
mlp_annotated_test  = mlp_y[indexes_annotated_test]
sub_annotated_test  = sub_y[indexes_annotated_test]
path_annotated_test = path_y[indexes_annotated_test]
maxv_annotated = max_voting_y[indexes_annotated_test]
minv_annotated = min_voting_y[indexes_annotated_test]
true_annotated = true_y[indexes_annotated_test]
print("lstm  : ", get_results(true_annotated, lstm_annotated_test))
print("mlp   : ", get_results(true_annotated, mlp_annotated_test))
print("path  : ", get_results(true_annotated, path_annotated_test))
print("sub   : ", get_results(true_annotated, sub_annotated_test))
print("minv  : ", get_results(true_annotated, minv_annotated))
print("snork : ", get_results(true_annotated, snorkel_y_annotated))
print("major : ", get_results(true_annotated, major_y_annotated))
print("maxv  : ", get_results(true_annotated, maxv_annotated))
print("random: ", get_results(true_annotated, random_y_annotated))
'''

q_map = {}
for m in models:
    q_map[m] = {}


transe_test_queries = load_pickle(filemap["transe"]["test"])
rotate_test_queries = load_pickle(filemap["rotate"]["test"])
complex_test_queries = load_pickle(filemap["complex"]["test"])

transe_x_test_fil = np.array(transe_test_queries['x_' + args.pred + "_fil"])[indexes_annotated]
rotate_x_test_fil = np.array(rotate_test_queries['x_' + args.pred + "_fil"])[indexes_annotated]
complex_x_test_fil = np.array(complex_test_queries['x_' + args.pred + "_fil"])[indexes_annotated]

logfile = log_dir + args.pred + "-combined-magic.log"
with open(logfile, "w") as log:
    for index, x in enumerate(tqdm(transe_x_test_fil)):
        e_transe  = int(x[0])
        r_transe  = int(x[1])
        a_transe  = int(x[2])
        e_rotate  = int(rotate_x_test_fil[index][0])
        r_rotate  = int(rotate_x_test_fil[index][1])
        a_rotate  = int(rotate_x_test_fil[index][2])
        e_complex = int(complex_x_test_fil[index][0])
        r_complex = int(complex_x_test_fil[index][1])
        a_complex = int(complex_x_test_fil[index][2])

        assert(e_transe == e_rotate)
        assert(e_transe == e_complex)
        assert(r_transe == r_rotate)
        assert(r_transe == r_complex)
        head = e_transe
        tail = a_transe
        #if args.pred == "head":
        #    head = a_transe
        #    tail = e_transe
        print("{}, {}, TransE:{}, RotatE:{}, ComplEx:{}".format(entity_dict[e_transe], relation_dict[r_transe], entity_dict[a_transe], entity_dict[a_rotate], entity_dict[a_complex]), file = log)
        #print("{}, {}, {},LSTM:{},MLP:{},PATH:{},SUB:{},maxV:{},Snorkel:{},REAL:{}".format(entity_dict[head], relation_dict[r], entity_dict[tail], lstm_y[index], mlp_y[index], path_y[index], sub_y[index], max_voting_y[index], snorkel_y[index], true_y[index]), file = log)
        if (index+1) % args.topk == 0:
            print("*" * 80, file = log)

