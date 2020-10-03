import sys
import pickle
import numpy as np

def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        pkl = pickle.load(fin)
    return pkl

input_file = sys.argv[1]
results = load_pickle(input_file)
y = np.array(results['fil']['predicted_y_abs'])

def make_results_per_topk(topk):
    new_y = []
    for i in range(0, len(y), 10):
        new_y.extend(y[i:topk])
    results['fil']['predicted_y_abs'] = new_y
    output_file = input_file.replace("topk-10", "topk-"+str(topk), 1)
    with open(output_file, "wb") as fout:
        pickle.dump(results, fout, protocol = pickle.HIGHEST_PROTOCOL)

for k in [1, 3, 5]:
    make_results_per_topk(k)
