import argparse
import sys
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description = 'Construct latex table comparing the performance between normal, star and diamond subgraphs from the experiment results')
    parser.add_argument('-r', dest = 'r', type = int, default = '100', help = 'Number of records that were tested')
    return parser.parse_args()

DATABASES = ["fb15k237", "lubm", "yago2", "dbpedia50"]
MODELS = ["transe", "rotate", "complex", "distmult", "hole"]
METRICS = ["Recall", "Reduction", "Runtime"]
SUBGRAPH_TYPES = ["normal", "star", "diamond"]
END_TYPES = ["H", "T"]
K_VALUES = ["10", "10%", "-1", "-2"]
K_VALUE_TRANSLATIONS = {"10": "10", "10%": "10\%", "-1": "$Dyn$", "-2": "$Dyn^T$"}
SCORE = "avg"
SUBGRAPH_SYMBOLS = {"normal": "\\bullet", "star": "\star", "diamond": "\diamond"}

DATA_DIR_PATH = "results/data/"
TABLE_DIR_PATH = "results/tables/"

def generate_line(data, model, metric, subgraph_type, end_type):
    line = "& " + metric.capitalize() + "(" + end_type + ")$" + SUBGRAPH_SYMBOLS[subgraph_type] + "$ & "
    if metric == "Runtime":
        line = "& " + metric.capitalize() + "$\ " + SUBGRAPH_SYMBOLS[subgraph_type] + "$ & "
    for database in DATABASES:
        for k in K_VALUES:
            type_str = metric.lower() + "_" + end_type
            if metric == "Reduction":
                type_str = "red_" + end_type
            if metric == "Runtime":
                type_str = "runtime"
            line += str(data[database][model][subgraph_type][k][SCORE][type_str]) + " & "
    line = line[:-2] + "\\\\"  # Remove ending & and add \\
    return line

def generate_latex_table(r):
    table = "\\begingroup\n"
    table += "\\setlength{\\tabcolsep}{6pt} % Default value: 6pt\n"
    table += "\\footnotesize\n"
    table += "\\begin{tabular}{p{0.3em} p{6em} || cccc | cccc | cccc | cccc}\n"

    # line for databases
    table += "& \\em Database" 
    for database in DATABASES:
        table += "& \\multicolumn{" + str(len(K_VALUES)) + "}{c}{$" + database + "$} "
    table += "\\\\\n"

    # line for k values
    table += "& \em K"
    for i in range(len(K_VALUES)):
        for k in K_VALUES:
            table += "& " + K_VALUE_TRANSLATIONS[k] + " "
    table += "\\\\\n"

    for model in MODELS:
        table += "\\cline{2-18}\n"
        table += "\\multirow{15}{*}{\\rotatebox{90}{\\" + model + "}}\n"
        for metric in METRICS:
            for subgraph_type in SUBGRAPH_TYPES:
                if metric == "Runtime":
                    table += generate_line(data, model, metric, subgraph_type, "") + "\n"
                    table += "\\cline{2-18}\n"
                else:    
                    for end_type in END_TYPES:
                        table += generate_line(data, model, metric, subgraph_type, end_type) + "\n"
                        table += "\\cline{2-18}\n"
        table += "\\\\[-5pt]"

    table += "\\end{tabular}\n"
    table += "\\endgroup"
    return table

args = parse_args()

data = dict()

for database in DATABASES:
    with open(DATA_DIR_PATH + database + '-r' + str(args.r) + '-results.pkl', 'rb') as fin:
        data[database] = pickle.load(fin)

with open(TABLE_DIR_PATH + 'comparison-r' + str(args.r) + '-results.tex', 'w') as fout:
    print(generate_latex_table(args.r), file = fout)