import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from snorkel.labeling.model.label_model import LabelModel
import logging
import sys

logging.basicConfig(level=logging.DEBUG)

N = int(sys.argv[1])
lstm_y = (np.random.random(N) * 2 // 1).astype(int)
mlp_y  = (np.random.random(N) * 2 // 1).astype(int)
path_y = (np.random.random(N) * 2 // 1).astype(int)
sub_y  = (np.random.random(N) * 2 // 1).astype(int)

lstm_y_test = (np.random.random(N) * 2 // 1).astype(int)
mlp_y_test  = (np.random.random(N) * 2 // 1).astype(int)
path_y_test = (np.random.random(N) * 2 // 1).astype(int)
sub_y_test  = (np.random.random(N) * 2 // 1).astype(int)

vs_train = np.transpose(np.vstack((lstm_y, mlp_y, path_y, sub_y)))

len_y  = len(lstm_y)
cnt_ones = int(len_y * 0.3)
cnt_zeros = len_y - cnt_ones
true_y_train = np.random.permutation([1] * cnt_ones + [0] * cnt_zeros)

print(vs_train)

vs_test  = np.transpose(np.vstack((lstm_y_test, mlp_y_test, path_y_test, sub_y_test)))
len_y_test  = len(lstm_y_test)
cnt_ones_test = int(len_y_test * 0.3)
cnt_zeros_test = len_y_test - cnt_ones_test
true_y_test = np.random.permutation([1] * cnt_ones_test + [0] * cnt_zeros_test)

label_model = LabelModel(verbose = True)
label_model.fit(vs_train, Y_dev = true_y_train, n_epochs = N,lr = 0.001, lr_scheduler = "linear", optimizer="adam")

snorkel_y, probs_y = label_model.predict(vs_train, return_probs = True)

print(np.round(label_model.get_weights(),2))
print(snorkel_y)
print(probs_y)

#print(len(vs_test[0]))
#print(len(true_y_test))
#print("*" * 80)
#print(label_model.score(L=vs_test, Y=true_y_test))
