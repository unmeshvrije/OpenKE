import argparse
import math
import pickle
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description = 'Run the experiments and extract the results in a format suitable for constructing latex tables.')
    parser.add_argument('-db', dest = 'db', type = str, default = 'fb15k237')
    parser.add_argument('-r', dest = 'r', type = int, default = '-1', help = 'Number of records to test')
    parser.add_argument('--metric', dest = 'metric', type = str, default = 'recall', help = 'The metric of interest, either "recall" or "reduction"')
    return parser.parse_args()

TEST_FILE_PATH = "test-subgraphs.sh"
DATA_DIR_PATH = "results/data/"
SUBGRAPH_COUNT = {          # used for k = 10% option
    "fb15k237-star": 7694,
    "fb15k237-diamond": 232216,
    "lubm-star": 1106,
    "lubm-diamond": 3270,
    "yago2-star": 8789,
    "yago2-diamond": 568,
    "dbpedia50-star": 326,
    "dbpedia50-diamond": 62
}


# Runs an experiment with particular metrics
#
# Arguments:
#    test_file_path - location of python script running a single experiment, usually test_subgraphs.py
#    database - one of databases "fb15k237", "lubm", "yago2", "dbpedia50"
#    model - one of models "transe", "rotate", "complex", "distmult", "hole"
#    r - number of records to test, usually 1000 or -1 (all the records)
#    k - threshold value, usually 10, -1 (dynamic k) or -2 (dynamic threshold)
#    s - score function. one of "avg", "kl", "nn"
#    metric - the metric of interest, either "recall" or "reduction"
#    subgraph_type - "star" or "diamond"
#    max_time - maximum permitted time per task in format "hh:mm:ss"
# Note: all arguments must be given as strings
#
# Results:
#    1st - reduction/recall (based on the 'metric' argument) value for Head (H)
#    2nd - reduction/recall value for Tail (T)
# In a case of unexpected behavior or a time limit the returned tuple is (-1, -1)
def run_test(test_file_path, database, model, r, k, s, metric = "Recall", subgraph_type = "star", max_time = "00:30:00"):
    # process k = 10% case
    if k.endswith('%'):
        k = SUBGRAPH_COUNT[database + "-" + subgraph_type] * int(k[:-1]) / 100
        k = str(math.floor(k))  # format k value

    proc = subprocess.Popen("prun -v -np 1 -t " + max_time + " -native '-C gpunode --gres=gpu:1' " + test_file_path + " -m " + model + " -d " + database + " -r " + r + " -k " + k + " -s " + s, stdout = subprocess.PIPE, shell = True)
    output = proc.stdout.readlines()

    recall_H = recall_T = red_H = red_T = 0
    # extract recall and reduction values from the output
    for line in output:
        line = line.decode('ascii')
        if line.startswith('Recall (H)'):
            recall_H = line[13:-1]    # ignore starting text "Recall (H) :" and the ending "\n" character
        elif line.startswith('Recall (T)'):
            recall_T = line[13:-1]
        elif line.startswith('%Red (H)'):
            red_H = line[14:-1]
        elif line.startswith('%Red (T)'):
            red_T = line[14:-1]

    # Check for undefined behavior
    if len(output) < 2:
        recall_H = recall_T = red_H = red_T = -1

    if metric.lower() == "recall":
        return str(round(float(recall_H), 2)), str(round(float(recall_T), 2))
    else:
        return str(round(float(red_H), 2)), str(round(float(red_T), 2))

# Generates a small portions of results with particular subgraph type and k value
def construct_model_cell_results(database, model, r, k, metric, subgraph_type):
    result_avg = run_test(TEST_FILE_PATH, database, model, r, k, "avg", metric, subgraph_type)
    result_kl = run_test(TEST_FILE_PATH, database, model, r, k, "kl", metric, subgraph_type)
    result_nn = run_test(TEST_FILE_PATH, database, model, r, k, "nn", metric, subgraph_type)
    experiment_results = dict()
    experiment_results["avg"] = {
        "H": result_avg[0],
        "T": result_avg[1]
    }
    experiment_results["kl"] = {
        "H": result_kl[0],
        "T": result_kl[1]
    }
    experiment_results["nn"] = {
        "H": result_nn[0],
        "T": result_nn[1]
    }
    return experiment_results

# Generates a portion of results with a particular subgraph type
def construct_model_subgraph_type_results(database, model, r, metric, subgraph_type):
    proc = subprocess.Popen("rm ../../../../var/scratch/dvs254/OpenKE-results/" + database + "/subgraphs/" + database + "-" + model + "-subgraphs-tau-10.pkl", stdout = subprocess.PIPE, shell = True)
    proc = subprocess.Popen("prun -v -np 1 -t 00:15:00 -native '-C gpunode --gres=gpu:1' ./step2-create-subgraphs.sh /var/scratch/dvs254/OpenKE-results/ " + model + " " + database + " " + subgraph_type, stdout = subprocess.PIPE, shell = True)
    creation_results = proc.stdout.read() # needed in order not to run two processes at the same time
    experiment_results = {

        "10": construct_model_cell_results(database, model, r, "10", metric, "star"),
        "10%": construct_model_cell_results(database, model, r, "10%", metric, "star"),
        "-1": construct_model_cell_results(database, model, r, "-1", metric, "star"),
        "-2": construct_model_cell_results(database, model, r, "-2", metric, "star")
    }
    return experiment_results

# Generates experiment results for a particular database and its model
def construct_model_results(database, model, r, metric):
    r = str(r)
    experiment_results = dict()
    experiment_results["star"] = construct_model_subgraph_type_results(database, model, r, metric, "star")
    experiment_results["diamond"] = construct_model_subgraph_type_results(database, model, r, metric, "diamond")
    return experiment_results

# Generates experiment results for a particular database
def construct_results(database, r, metric):
    experiment_results = dict()
    experiment_results["transe"] = construct_model_results(database, "transe", r, metric)
    experiment_results["hole"] = construct_model_results(database, "hole", r, metric)
    experiment_results["rotate"] = construct_model_results(database, "rotate", r, metric)
    experiment_results["distmult"] = construct_model_results(database, "distmult", r, metric)
    experiment_results["complex"] = construct_model_results(database, "complex", r, metric)
    return experiment_results

args = parse_args()

#generate_latex_table("dbpedia50", 1000, "Recall")
with open(DATA_DIR_PATH + args.db + '-r' + str(args.r) + '-' + args.metric + '-results.pkl', 'wb') as fout:
    pickle.dump(construct_results(args.db, args.r, args.metric), fout, protocol = pickle.HIGHEST_PROTOCOL)
