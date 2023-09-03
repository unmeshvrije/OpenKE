import pickle5 as pickle
from subgraphs import read_triples

DATASET = "yago2"
MODELS = ["transe", "rotate"]

TEST_FILE = "./benchmarks/" + DATASET + "/test2id.txt"
RDICT_FILE = "/var/scratch/dvs254/OpenKE-results/" + DATASET + "/misc/" + DATASET + "-id-to-relation.pkl"

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


triples = read_triples(TEST_FILE)
with open(RDICT_FILE, 'rb') as fin:
    rdict = pickle.load(fin)
print_relation_statistics(triples, rdict, 0, 100)


