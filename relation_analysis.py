import argparse
import os
import pickle5 as pickle
from subgraphs import read_triples

MODELS = ["transe", "rotate"]

def parse_args():
    parser = argparse.ArgumentParser(description = 'Run tests with different relation types')
    parser.add_argument('--dataset', dest = 'dataset', type = str, required = True, default = 'yago2', help = 'dataset to run tests on')
    parser.add_argument('--reldict', dest = 'rel_dict', type = str, default = '/var/scratch/dvs254/OpenKE-results/', help = 'relation id dictionary')
    parser.add_argument('--testdict', dest = 'test_dict', type = str, default = './benchmarks/', help = 'test triple dictionary')
    parser.add_argument('--createfiles', default = False, action = 'store_true', help = "create relation test files")
    parser.add_argument('--no-createfiles', dest = 'createfiles', action = 'store_false', help = "do not create relation test files")
    parser.add_argument('--createdfilesdict', dest = 'created_dict', type = str, default = './relation_test_triples/', help = 'relation id dictionary')
    return parser.parse_args()

# Prints first k relation types that have at least minN entities
def print_relation_statistics(triples, rdict, minN, k):
    relation_frequency = {}
    for triple in triples:
        r = triple[2]
        if r not in relation_frequency:
            relation_frequency[r] = 1
        else:
            relation_frequency[r] = relation_frequency[r] + 1
    sorted_relation_frequency = sorted(relation_frequency.items(), key = lambda x:x[1], reverse = True)
    print("  r |   freq | translation")
    for i in range(min(len(relation_frequency), k)):
        if sorted_relation_frequency[i][1] < minN:
            break
        print('{:>3}'.format(sorted_relation_frequency[i][0]), end = "   ")
        print('{:>6}'.format(sorted_relation_frequency[i][1]), end = "   ")
        print(rdict[sorted_relation_frequency[i][0]])

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
print_relation_statistics(triples, rdict, 0, 100)
