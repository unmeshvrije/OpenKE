import argparse
import math
import pickle
import subprocess

"""
Runs tests with diferent parameters for a given database and the number of records to test (given as arguments to the script)

The results are stored in a pickle file with format "DATABASE-rRECORDS_TO_TEST-results.pkl"

Different parameters:
subgraph_type: single (no subgraphs), star, diamond
model: transe, rotate, complex, distmult, hole
k: 5, 10, 10%, -1 (dynamic k), -2 (dynamic threshold)
score function: avg, kl, nn
"""

def parse_args():
    parser = argparse.ArgumentParser(description = 'Run the experiments and extract the results in a format suitable for constructing latex tables.')
    parser.add_argument('-db', dest = 'db', type = str, default = 'fb15k237')
    parser.add_argument('-r', dest = 'r', type = int, default = '-1', help = 'Number of records to test')
    return parser.parse_args()

TEST_FILE_PATH = "test-subgraphs.sh"
TEST_FILE_PATH_SINGLE = "test-model.sh"
DATA_DIR_PATH = "results/data/"
SUBGRAPH_COUNT = {          # used for k = 10% option
    "fb15k237-star": 7694,
    "fb15k237-diamond": 116108,
    "fb15k237-single": 14541,
    "lubm-star": 1106,
    "lubm-diamond": 1635,
    "lubm-single": 17292,
    "yago2-star": 8789,
    "yago2-diamond": 284,
    "yago2-single": 397253,
    "dbpedia50-star": 326,
    "dbpedia50-diamond": 31,
    "dbpedia50-single": 24624
}


def run_test(test_file_path, database, model, r, k, s, subgraph_type = "star", max_time = "2:30:00"):
    """
    Runs an experiment with particular parameters

    Arguments
    ---------
    test_file_path: location of python script running a single experiment, usually test_subgraphs.py or test-model.sh
    database: one of databases "fb15k237", "lubm", "yago2", "dbpedia50"
    model: one of models "transe", "rotate", "complex", "distmult", "hole"
    r: number of records to test, usually 100, 1000 or -1 (all the records)
    k: threshold value, usually 10, -1 (dynamic k) or -2 (dynamic threshold)
    s: score function, one of "avg", "kl", "nn"
    subgraph_type: "star", "diamond" or "single" (tests on single entities do not use kl and nn score calculations)
    max_time: maximum permitted time per task in format "hh:mm:ss"
    
    Note: all arguments must be given as strings

    Return values
    -------
    proc - the process that has finished running an experiment. It should be later handled with process_results() function
    """
    if subgraph_type == "single" and ((s == "kl" or s == "nn") or (k == "-1" or k == "-2")):
        proc = subprocess.Popen(["sleep 0"], stdout = subprocess.PIPE, shell = True)
        return proc
    # process k = 10% case
    if k.endswith('%'):
        k = SUBGRAPH_COUNT[database + "-" + subgraph_type] * int(k[:-1]) / 100
        k = str(math.floor(k))  # format k value

    if subgraph_type == "single":
        proc = subprocess.Popen("prun -v -np 1 -t " + max_time + " -native '-C gpunode --gres=gpu:1' " + TEST_FILE_PATH_SINGLE + " " + database + " " + model + " " + k + " " + r, stdout = subprocess.PIPE, shell = True)
    else:
        proc = subprocess.Popen("prun -v -np 1 -t " + max_time + " -native '-C gpunode --gres=gpu:1' " + TEST_FILE_PATH + " -m " + model + " -d " + database + " --type " + subgraph_type + " -r " + r + " -k " + k + " -s " + s, stdout = subprocess.PIPE, shell = True)
    return proc

def process_results(result_proc):
    """
    Processes the result output and extracts the recall and reduction metrics

    Arguments
    ---------
    proc - process that run the test in run_test() function

    Return values
    -------
    results: dictionary consisting of recall_H, recall_T, red_H, red_T, runtime
        In a case of unexpected behavior or a time limit the returned dictionary tuple is (-1, -1, -1, -1, -1)
    """
    output = result_proc.stdout.readlines()

    recall_H = recall_T = precision_H = precision_T = red_H = red_T = runtime = 0
    # extract recall and reduction values from the output

    for line in output:
        line = line.decode('ascii')
        if line.startswith('Recall (H)'):
            recall_H = line[13:-1]    # ignore starting text "Recall (H) :" and the ending "\n" character
        elif line.startswith('Recall (T)'):
            recall_T = line[13:-1]
        elif line.startswith('Precision (H)'):
            precision_H = line[16:-1]
        elif line.startswith('Precision (T)'):
            precision_T = line[16:-1]
        elif line.startswith('%Red (H)'):
            red_H = line[14:-1]
        elif line.startswith('%Red (T)'):
            red_T = line[14:-1]
        elif line.startswith('Runtime'):
            runtime = line[10:-2]

    # Check for undefined behavior
    if len(output) < 2:
        recall_H = recall_T = precision_H = precision_T = red_H = red_T = runtime = -1

    results = dict()
    results["recall_H"] = str(round(float(recall_H), 2))
    results["recall_T"] = str(round(float(recall_T), 2))
    results["precision_H"] = str(round(float(precision_H), 3))
    results["precision_T"] = str(round(float(precision_T), 3))
    results["red_H"] = str(round(float(red_H), 2))
    results["red_T"] = str(round(float(red_T), 2))
    results["runtime"] = str(round(float(runtime)))
    return results

def construct_model_cell_results(database, model, r, k, subgraph_type):
    """Generates a small portions of results with particular subgraph type and k value"""
    result_avg_proc = run_test(TEST_FILE_PATH, database, model, r, k, "avg", subgraph_type)
    result_kl_proc = run_test(TEST_FILE_PATH, database, model, r, k, "kl", subgraph_type)
    result_nn_proc = run_test(TEST_FILE_PATH, database, model, r, k, "nn", subgraph_type)
    experiment_results = dict()
    experiment_results["avg"] = process_results(result_avg_proc)
    experiment_results["kl"] = process_results(result_kl_proc)
    experiment_results["nn"] = process_results(result_nn_proc)
    return experiment_results

def construct_model_subgraph_type_results(database, model, r, subgraph_type):
    """Generates a portion of results with a particular subgraph type"""
    experiment_results = {
        "5": construct_model_cell_results(database, model, r, "5", subgraph_type),
        "10": construct_model_cell_results(database, model, r, "10", subgraph_type),
        "10%": construct_model_cell_results(database, model, r, "10%", subgraph_type),
        "-1": construct_model_cell_results(database, model, r, "-1", subgraph_type),
        "-2": construct_model_cell_results(database, model, r, "-2", subgraph_type)
    }
    return experiment_results

def construct_model_results(database, model, r):
    """Generates experiment results for a particular database and its model"""
    r = str(r)
    experiment_results = dict()
    experiment_results["single"] = construct_model_subgraph_type_results(database, model, r, "single")
    experiment_results["star"] = construct_model_subgraph_type_results(database, model, r, "star")
    experiment_results["diamond"] = construct_model_subgraph_type_results(database, model, r, "diamond")
    return experiment_results

def construct_results(database, r):
    """Generates experiment results for a particular database"""
    experiment_results = dict()
    experiment_results["transe"] = construct_model_results(database, "transe", r)
    experiment_results["hole"] = construct_model_results(database, "hole", r)
    experiment_results["rotate"] = construct_model_results(database, "rotate", r)
    experiment_results["distmult"] = construct_model_results(database, "distmult", r)
    experiment_results["complex"] = construct_model_results(database, "complex", r)
    return experiment_results

if __name__ == "__main__":
    args = parse_args()

    with open(DATA_DIR_PATH + args.db + '-r' + str(args.r) + '-results.pkl', 'wb') as fout:
        pickle.dump(construct_results(args.db, args.r), fout, protocol = pickle.HIGHEST_PROTOCOL)
