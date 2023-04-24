import argparse
import math
import pickle
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description = 'Run the experiments and extract the results in a format suitable for constructing latex tables.')
    parser.add_argument('-db', dest = 'db', type = str, default = 'fb15k237')
    parser.add_argument('-r', dest = 'r', type = int, default = '-1', help = 'Number of records to test')
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


# Runs an experiment with particular parameters
#
# Arguments:
#    test_file_path - location of python script running a single experiment, usually test_subgraphs.py
#    database - one of databases "fb15k237", "lubm", "yago2", "dbpedia50"
#    model - one of models "transe", "rotate", "complex", "distmult", "hole"
#    r - number of records to test, usually 1000 or -1 (all the records)
#    k - threshold value, usually 10, -1 (dynamic k) or -2 (dynamic threshold)
#    s - score function. one of "avg", "kl", "nn"
#    subgraph_type - "star" or "diamond"
#    max_time - maximum permitted time per task in format "hh:mm:ss"
# Note: all arguments must be given as strings
#
# Results:
#    proc - the process that has finished running an experiment. It should be later handled with process_results() function
# In a case of unexpected behavior or a time limit the returned tuple is (-1, -1)
def run_test(test_file_path, database, model, r, k, s, subgraph_type = "star", max_time = "02:30:00"):
    # process k = 10% case
    if k.endswith('%'):
        k = SUBGRAPH_COUNT[database + "-" + subgraph_type] * int(k[:-1]) / 100
        k = str(math.floor(k))  # format k value

    proc = subprocess.Popen("prun -v -np 1 -t " + max_time + " -native '-C gpunode --gres=gpu:1' " + test_file_path + " -m " + model + " -d " + database + " -type " + subgraph_type + " -r " + r + " -k " + k + " -s " + s, stdout = subprocess.PIPE, shell = True)
    return proc

# Processes the result output and extracts the recall and reduction metrics
#
# Arguments:
#    proc - process that run the test in run_test() function
#
# Results:
#    result dictionary consisting of recall_H, recall_T, red_H, red_T, runtime
# In a case of unexpected behavior or a time limit the returned tuple is (-1, -1, -1, -1, -1)
def process_results(result_proc):
    output = result_proc.stdout.readlines()

    recall_H = recall_T = red_H = red_T = runtime = 0
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
        elif line.startswith('Runtime'):
            runtime = line[10:-2]

    # Check for undefined behavior
    if len(output) < 2:
        recall_H = recall_T = red_H = red_T = runtime = -1

    results = dict()
    results["recall_H"] = str(round(float(recall_H), 2))
    results["recall_T"] = str(round(float(recall_T), 2))
    results["red_H"] = str(round(float(red_H), 2))
    results["red_T"] = str(round(float(red_T), 2))
    results["runtime"] = str(round(float(runtime)))
    return results

# Generates a small portions of results with particular subgraph type and k value
def construct_model_cell_results(database, model, r, k, subgraph_type):
    result_avg_proc = run_test(TEST_FILE_PATH, database, model, r, k, "avg", subgraph_type)
    result_kl_proc = run_test(TEST_FILE_PATH, database, model, r, k, "kl", subgraph_type)
    result_nn_proc = run_test(TEST_FILE_PATH, database, model, r, k, "nn", subgraph_type)
    experiment_results = dict()
    experiment_results["avg"] = process_results(result_avg_proc)
    experiment_results["kl"] = process_results(result_kl_proc)
    experiment_results["nn"] = process_results(result_nn_proc)
    return experiment_results

# Generates a portion of results with a particular subgraph type
def construct_model_subgraph_type_results(database, model, r, subgraph_type):
    experiment_results = {
        "10": construct_model_cell_results(database, model, r, "10", subgraph_type),
        "10%": construct_model_cell_results(database, model, r, "10%", subgraph_type),
        "-1": construct_model_cell_results(database, model, r, "-1", subgraph_type),
        "-2": construct_model_cell_results(database, model, r, "-2", subgraph_type)
    }
    return experiment_results

# Generates experiment results for a particular database and its model
def construct_model_results(database, model, r):
    r = str(r)
    experiment_results = dict()
    experiment_results["star"] = construct_model_subgraph_type_results(database, model, r, "star")
    experiment_results["diamond"] = construct_model_subgraph_type_results(database, model, r, "diamond")
    return experiment_results

# Generates experiment results for a particular database
def construct_results(database, r):
    experiment_results = dict()
    experiment_results["transe"] = construct_model_results(database, "transe", r)
    experiment_results["hole"] = construct_model_results(database, "hole", r)
    experiment_results["rotate"] = construct_model_results(database, "rotate", r)
    experiment_results["distmult"] = construct_model_results(database, "distmult", r)
    experiment_results["complex"] = construct_model_results(database, "complex", r)
    return experiment_results

args = parse_args()

with open(DATA_DIR_PATH + args.db + '-r' + str(args.r) + '-results.pkl', 'wb') as fout:
    pickle.dump(construct_results(args.db, args.r), fout, protocol = pickle.HIGHEST_PROTOCOL)
