import subprocess

TEST_FILE_PATH = "test-subgraphs.sh"

# Runs an experiment with particular metrics
#
# Arguments:
#    test_file_path - location of python script running a single experiment, usually test_subgraphs.py
#    model - one of models "transe", "rotate", "complex", "distmult", "hole"
#    database - one of databases "fb15k237", "lubm", "yago2", "dbpedia50"
#    r - number of records to test, usually 1000 or -1 (all the records)
#    k - threshold value, usually 10, -1 (dynamic k) or -2 (dynamic threshold)
#    s - score function. one of "avg", "kl", "nn"
#    metric - the metric of interest, either "Recall" or "Reduction"/"Red"
#    max_time - maximum permitted time per task in format "hh:mm:ss"
# Note: all arguments must be given as strings
#
# Results:
#    1st - reduction/recall (based on the 'metric' argument) value for Head (H)
#    2nd - reduction/recall value for Tail (T)
# In a case of unexpected behavior or a time limit the returned tuple is (-1, -1)
def run_test(test_file_path, model, database, r, k, s, metric = "Recall", max_time = "00:30:00"):
    proc = subprocess.Popen("prun -v -np 1 -t " + max_time + " -native '-C gpunode --gres=gpu:1' " + test_file_path + " -m " + model + " -d " + database + " -r " + r + " -k " + k + " -s " + s, stdout = subprocess.PIPE, shell = True)
    output = proc.stdout.readlines()

    recall_H = recall_T = red_H = red_T = 0
    # extract recall and reduction values from the output
    for line in output:
        line = line.decode('ascii')
        if line.startswith('Recall (H)'):
            recall_H = line[13:-1]    # ignore starting text (Recall (H) :) and the ending \n character
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
        return recall_H, recall_T
    else:
        return red_H, red_T

result_tuple = run_test(TEST_FILE_PATH, "transe", "dbpedia50", "1000", "10", "avg", "Recall")
print(result_tuple[0], result_tuple[1])
