import argparse
import os
import pickle5 as pickle
from run_tests_with_and_without_subgraphs import run_test, process_results
from subgraphs import read_triples

MODELS = ["transe", "rotate"]
K = "10"

def parse_args():
    parser = argparse.ArgumentParser(description = 'Run tests with different relation types')
    parser.add_argument('--minN', dest = 'minN', type = int, required = False, default = 20, help = 'minimum size of considered relations')
    parser.add_argument('--maxRelations', dest = 'maxRelations', type = int, default = 100, help = 'maximum number of relations to consider')
    parser.add_argument('--dataset', dest = 'dataset', type = str, required = True, default = 'yago2', help = 'dataset to run tests on')
    parser.add_argument('--reldict', dest = 'rel_dict', type = str, default = '/var/scratch/dvs254/OpenKE-results/', help = 'relation id dictionary')
    parser.add_argument('--testdict', dest = 'test_dict', type = str, default = './benchmarks/', help = 'test triple dictionary')
    parser.add_argument('--createfiles', default = False, action = 'store_true', help = "create relation test files")
    parser.add_argument('--no-createfiles', dest = 'createfiles', action = 'store_false', help = "do not create relation test files")
    parser.add_argument('--createdfilesdict', dest = 'created_dict', type = str, default = './relation_test_triples/', help = 'relation id dictionary')
    parser.add_argument('--resultdict', dest = 'result_dict', type = str, default = './results/data/', help = "dictionary for saving relation statistics")
    return parser.parse_args()

# For k relation types that have at least minN entities, the tests are run and recall, reduction and precision statistics are printed
def print_relation_statistics(triples, minN, k, created_dict, dataset, result_dict):
    relation_frequency = {}
    for triple in triples:
        r = triple[2]
        if r not in relation_frequency:
            relation_frequency[r] = 1
        else:
            relation_frequency[r] = relation_frequency[r] + 1
    sorted_relation_frequency = sorted(relation_frequency.items(), key = lambda x:x[1], reverse = True)
    experiment_results_star = {}
    experiment_results_diamond = {}
    for i in range(min(len(relation_frequency), k)):
        if sorted_relation_frequency[i][1] < minN:
            break
        r = sorted_relation_frequency[i][0]
        experiment_results_star[r] = {}
        experiment_results_diamond[r] = {}
        for model in MODELS:
            relation_file = created_dict + dataset + "/" + "r-" + str(r) + ".txt"
            experiment_results_star[r][model] = process_results(run_test(relation_file, dataset, model, str(sorted_relation_frequency[i][1]), K, "avg", "star"))
            experiment_results_diamond[r][model] = process_results(run_test(relation_file, dataset, model, str(sorted_relation_frequency[i][1]), K, "avg", "diamond"))
    print_statistics_per_type("star", sorted_relation_frequency, k, minN, experiment_results_star, dataset, result_dict)
    print_statistics_per_type("diamond", sorted_relation_frequency, k, minN, experiment_results_diamond, dataset, result_dict)


def print_statistics_per_type(subgraph_type, sorted_relation_frequency, k, minN, experiment_results, dataset, result_dict):
    with open(result_dict + dataset + "-relation-comparison-" + subgraph_type + ".txt", "w") as fout:
        print("           |            " + '{:<19}'.format(MODELS[0] + "(h)") + " |            " + '{:<19}'.format(MODELS[0] + "(t)") + " |            " + '{:<19}'.format(MODELS[1] + "(h)") + " |            " + '{:<19}'.format(MODELS[1] + "(t)") + " |", file = fout)
        print("  r | freq | recall | reduction | precision | recall | reduction | precision | recall | reduction | precision | recall | reduction | precision | translation", file = fout)
        for i in range(min(len(sorted_relation_frequency), k)):
            if sorted_relation_frequency[i][1] < minN:
                break
            r = sorted_relation_frequency[i][0]
            print('{:>3}'.format(r), end = "  ", file = fout)
            print('{:>5}'.format(sorted_relation_frequency[i][1]), end = "   ", file = fout)
            for model in MODELS:
                results = experiment_results[r][model]
                print('{:>6}'.format(round(float(results['recall_H']), 3)), end = "   ", file = fout)
                print('{:>9}'.format(round(float(results['red_H']), 3)), end = "   ", file = fout)
                print('{:>9}'.format(round(float(results['precision_H']), 3)), end = "   ", file = fout)
                print('{:>6}'.format(round(float(results['recall_T']), 3)), end = "   ", file = fout)
                print('{:>9}'.format(round(float(results['red_T']), 3)), end = "   ", file = fout)
                print('{:>9}'.format(round(float(results['precision_T']), 3)), end = "   ", file = fout)
            print(rdict[sorted_relation_frequency[i][0]], file = fout)
        print(file = fout)


# For each relation type r, a file is created containing triples only with relation r
def create_relation_test_files(created_dict, dataset, triples):
    created_relation_dict = created_dict + dataset + "/"
    relation_triples_dict = {}
    for triple in triples:
        r = triple[2]
        if r not in relation_triples_dict:
            relation_triples_dict[r] = []
        relation_triples_dict[r].append(triple)
    for relation in relation_triples_dict:
        relation_file = created_relation_dict + "r-" + str(relation) + ".txt"
        if not os.path.exists(created_relation_dict):
            os.makedirs(created_relation_dict)
        with open(relation_file, "w") as fout:
            print(len(relation_triples_dict[relation]), file = fout)
            for triple in relation_triples_dict[relation]:
                print(str(triple[0]) + " " + str(triple[1]) + " " + str(triple[2]), file = fout)


args = parse_args()

TEST_FILE = args.test_dict + args.dataset + "/test2id.txt"
RDICT_FILE = args.rel_dict + args.dataset + "/misc/" + args.dataset + "-id-to-relation.pkl"

triples = read_triples(TEST_FILE)
with open(RDICT_FILE, 'rb') as fin:
    rdict = pickle.load(fin)
if args.createfiles:
    create_relation_test_files(args.created_dict, args.dataset, triples)
print_relation_statistics(triples, args.minN, args.maxRelations, args.created_dict, args.dataset, args.result_dict)
