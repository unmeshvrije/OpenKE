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
    #filemap[m]["sub"]   = result_dir + args.db + "-" + m + "-subgraphs-tau-10-tail.out"
    #filemap[m]["path"]  = result_dir + args.db + "-" + m + "-path-classifier-tail.out"
    filemap[m]["test"]  = args.result_dir + args.db + "/data/" + args.db + "-" + m + "-test-topk-10.pkl"

classifiers = ["true", "lstm", "mlp"]#, "sub", "path"]

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
    #results_map[m]["sub"]  = load_pickle(filemap[m]["sub"])
    #results_map[m]["path"] = load_pickle(filemap[m]["path"])


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
    #y_map[m]["sub"]  = np.array(results_map[m]["sub"]['fil'][y_label_str])
    #y_map[m]["path"] = np.array(results_map[m]["path"]['fil'][y_label_str])

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
    5. Find indexes of answer triples which are actually annotated
    For now, consider answers that are either labelled 0 or 1
'''
indexes_annotated1 = np.where(y_map["transe"]["true"] != -1)[0]
indexes_annotated2 = np.where(y_map["rotate"]["true"] != -1)[0]
indexes_annotated3 = np.where(y_map["complex"]["true"] != -1)[0]
indexes_annotated = indexes_annotated1
assert((indexes_annotated1 == indexes_annotated2).all())
assert((indexes_annotated3 == indexes_annotated2).all())

y_annotated_seq = {}
for m in models:
    y_annotated_seq[m] = {}
    for c in classifiers:
        y_annotated_seq[m][c] = y_map[m][c][indexes_annotated]

transe_test_queries = load_pickle(filemap["transe"]["test"])
rotate_test_queries = load_pickle(filemap["rotate"]["test"])
complex_test_queries = load_pickle(filemap["complex"]["test"])

transe_x_test_fil = np.array(transe_test_queries['x_' + args.pred + "_fil"])[indexes_annotated]
rotate_x_test_fil = np.array(rotate_test_queries['x_' + args.pred + "_fil"])[indexes_annotated]
complex_x_test_fil = np.array(complex_test_queries['x_' + args.pred + "_fil"])[indexes_annotated]
q_map = {}
q_map["transe"] = transe_x_test_fil
q_map["rotate"] = rotate_x_test_fil
q_map["complex"] = complex_x_test_fil

'''
    Build a map that stores 1/0/-1 answers in a dictionary
    with key (ent, rel) for all three models
    map["transe"]["lstm"][34] = should give label for answer id 34 for lstm run with transe.
'''
y_annotated_map = {}
for m in models:
    y_annotated_map[m] = {}
    for c in classifiers:
        y_annotated_map[m][c] = {}#y_map[m][c][indexes_annotated]

for index, x in enumerate(tqdm(transe_x_test_fil)):
    for m in models:
        ent = int (q_map[m][index][0])
        rel = int (q_map[m][index][1])
        ans = int (q_map[m][index][2])
        for c in classifiers:
            y_annotated_map[m][c][ans] = y_annotated_seq[m][c][index]

'''
    function that accepts 4 classifiers y labels
    and annotated indexes, fills the out array with labels at those indexes
'''
def get_snorkel_labels(c1, c2, c3, c4, c5, c6, true_y, indexes_annotated):
    snorkel_y = np.empty(len(c1), dtype = np.int)
    snorkel_y.fill(-1);

    kf = KFold(n_splits = 5, shuffle = False, random_state = 12)
    #kf.split(indexes_annotated)

    max_accuracy = 0.0
    L_test_max = None
    indexes_annotated_test_max = None
    best_model = None
    for train_split, test_split in kf.split(indexes_annotated):
        print("Splitting training and test annotated data: ")

        indexes_annotated_train = indexes_annotated[train_split]
        indexes_annotated_test  = indexes_annotated[test_split]

        c1_annotated_test = c1[indexes_annotated_test]
        c2_annotated_test = c2[indexes_annotated_test]
        c3_annotated_test = c3[indexes_annotated_test]
        c4_annotated_test = c4[indexes_annotated_test]
        c5_annotated_test = c5[indexes_annotated_test]
        c6_annotated_test = c6[indexes_annotated_test]
        true_annotated_test = true_y[indexes_annotated_test]

        c1_annotated_train = c1[indexes_annotated_train]
        c2_annotated_train = c2[indexes_annotated_train]
        c3_annotated_train = c3[indexes_annotated_train]
        c4_annotated_train = c4[indexes_annotated_train]
        c5_annotated_train = c5[indexes_annotated_train]
        c6_annotated_train = c6[indexes_annotated_train]
        true_annotated_train = true_y[indexes_annotated_train]


        label_model = LabelModel(verbose = False)
        L_train = np.transpose(np.vstack((c1_annotated_train, c2_annotated_train, c3_annotated_train, c4_annotated_train, c5_annotated_train,
                                          c6_annotated_train )))
        label_model.fit(L_train, Y_dev=true_annotated_train, n_epochs=500, optimizer="adam")
        L_test = np.transpose(np.vstack(( c1_annotated_test, c2_annotated_test, c3_annotated_test, c4_annotated_test, c5_annotated_test,
                                          c6_annotated_test)))
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
    L_test_max = np.transpose(np.vstack((c1, c2, c3, c4, c5, c6)))
    snorkel_y = best_model.predict(L_test_max, tie_break_policy = "random")

    return best_model, snorkel_y

def r2(num):
    return np.round(num, 2)

def get_results(y_true, y_predicted):
    #conf = confusion_matrix(y_true, y_predicted)
    result = classification_report(y_true, y_predicted, output_dict = True)
    return  "Precision = " + str(r2(result['1']['precision'])) + "," +\
            "Recall = "+str(r2(result['1']['recall']))         + "," +\
            "F1 score = "+str(r2(result['1']['f1-score']))         + "," +\
            "Accuracy(overall) = "+str(r2(result['accuracy']))



ans_dict_map = {}
for m in models:
    ans_dict_map[m] = {}

def build_ans_dict(answer_triples):
    ans_dict = {}
    for triple in answer_triples:
        ent = int(triple[0])
        rel = int(triple[1])
        ans = int(triple[2])
        if (ent, rel) not in ans_dict.keys():
            ans_dict[(ent, rel)] = [ans]
        else:
            ans_dict[(ent, rel)].append(ans)
    return ans_dict

ans_dict_map["transe"] = build_ans_dict(transe_x_test_fil)
ans_dict_map["rotate"] = build_ans_dict(rotate_x_test_fil)
ans_dict_map["complex"] = build_ans_dict(complex_x_test_fil)

transe_lstm_y   = []
transe_mlp_y    = []
rotate_lstm_y   = []
rotate_mlp_y    = []
complex_lstm_y  = []
complex_mlp_y   = []


same_answer_count = 0
abstain_count     = 0

# 1. TransE
for index, x in enumerate(tqdm(transe_x_test_fil)):
    transe_lstm_y.append(y_annotated_seq["transe"]["lstm"][index])
    transe_mlp_y.append(y_annotated_seq["transe"]["mlp"][index])

    e_transe = int(x[0])
    r_transe = int(x[1])
    a_transe = int(x[2])

    if a_transe in ans_dict_map["rotate"][(e_transe, r_transe)]:
        rotate_lstm_y.append(y_annotated_map["rotate"]["lstm"][a_transe])
        rotate_mlp_y.append(y_annotated_map["rotate"]["mlp"][a_transe])
        same_answer_count += 1
    else:# abstain
        rotate_lstm_y.append(-1)
        rotate_mlp_y.append(-1)
        abstain_count += 1

    if a_transe in ans_dict_map["complex"][(e_transe, r_transe)]:
        complex_lstm_y.append(y_annotated_map["complex"]["lstm"][a_transe])
        complex_mlp_y.append(y_annotated_map["complex"]["mlp"][a_transe])
        same_answer_count += 1
    else:# abstain
        complex_lstm_y.append(-1)
        complex_mlp_y.append(-1)
        abstain_count += 1


# 2. Rotate Answers
for index, x in enumerate(tqdm(rotate_x_test_fil)):
    rotate_lstm_y.append(y_annotated_seq["rotate"]["lstm"][index])
    rotate_mlp_y.append(y_annotated_seq["rotate"]["mlp"][index])

    e_rotate = int(x[0])
    r_rotate = int(x[1])
    a_rotate = int(x[2])

    if a_rotate in ans_dict_map["transe"][(e_rotate, r_rotate)]:
        transe_lstm_y.append(y_annotated_map["transe"]["lstm"][a_rotate])
        transe_mlp_y.append(y_annotated_map["transe"]["mlp"][a_rotate])
        same_answer_count += 1
    else:# abstain
        transe_lstm_y.append(-1)
        transe_mlp_y.append(-1)
        abstain_count += 1

    if a_rotate in ans_dict_map["complex"][(e_rotate, r_rotate)]:
        complex_lstm_y.append(y_annotated_map["complex"]["lstm"][a_rotate])
        complex_mlp_y.append(y_annotated_map["complex"]["mlp"][a_rotate])
        same_answer_count += 1
    else:# abstain
        complex_lstm_y.append(-1)
        complex_mlp_y.append(-1)
        abstain_count += 1

# 3. Complex Answers
for index, x in enumerate(tqdm(complex_x_test_fil)):
    complex_lstm_y.append(y_annotated_seq["complex"]["lstm"][index])
    complex_mlp_y.append(y_annotated_seq["complex"]["mlp"][index])
    e_complex = int(x[0])
    r_complex = int(x[1])
    a_complex = int(x[2])

    if a_complex in ans_dict_map["transe"][(e_complex, r_complex)]:
        transe_lstm_y.append(y_annotated_map["transe"]["lstm"][a_complex])
        transe_mlp_y.append(y_annotated_map["transe"]["mlp"][a_complex])
        same_answer_count += 1
    else:# abstain
        transe_lstm_y.append(-1)
        transe_mlp_y.append(-1)
        abstain_count += 1

    if a_complex in ans_dict_map["rotate"][(e_complex, r_complex)]:
        rotate_lstm_y.append(y_annotated_map["rotate"]["lstm"][a_complex])
        rotate_mlp_y.append(y_annotated_map["rotate"]["mlp"][a_complex])
        same_answer_count += 1
    else:# abstain
        rotate_lstm_y.append(-1)
        rotate_mlp_y.append(-1)
        abstain_count += 1


gold_y  = []
gold_y.extend(y_annotated_seq["transe"]["true"])
gold_y.extend(y_annotated_seq["rotate"]["true"])
gold_y.extend(y_annotated_seq["complex"]["true"])

all_indexes = np.arange(len(transe_lstm_y))

label_model, snorkel_y = get_snorkel_labels(
                                        np.array(transe_lstm_y),  np.array(transe_mlp_y),
                                        np.array(rotate_lstm_y),  np.array(rotate_mlp_y),
                                        np.array(complex_lstm_y), np.array(complex_mlp_y),
                                        np.array(gold_y), all_indexes
                                        )
print("snork : ", get_results(gold_y, snorkel_y))

# 0, 2100
# 2100, 4200
for i, (si, ei) in enumerate(zip([0, 2100, 4200], [2100, 4200, 6300])):
    transe_test_snorkel = np.transpose(np.vstack ((
                                            np.array(transe_lstm_y[si:ei]),  np.array(transe_mlp_y[si:ei]),
                                            np.array(rotate_lstm_y[si:ei]),  np.array(rotate_mlp_y[si:ei]),
                                            np.array(complex_lstm_y[si:ei]), np.array(complex_mlp_y[si:ei])
                                                   )))
    snorkel_y_transe = label_model.predict(transe_test_snorkel, tie_break_policy="random")
    print("snork ( "+ models[i] + " answers ) : ", get_results(gold_y[si:ei], snorkel_y_transe))

major_voter = MajorityLabelVoter()
major_y = major_voter.predict(transe_test_snorkel, tie_break_policy="random")
print("major : ", get_results(gold_y[si:ei], major_y))

random_voter = RandomVoter()
random_y = random_voter.predict(transe_test_snorkel, tie_break_policy = "random")
print("random: ", get_results(gold_y[si:ei], random_y))

'''
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
'''
