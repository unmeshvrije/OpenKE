import sys
import json
import pickle

answer_file1 = sys.argv[1]
answer_file2 = sys.argv[2]

# Load the answers from the old model
testdata = json.load(open(answer_file1, 'rt'))
queries_head = {}
queries_tail = {}
for t in testdata:
    head = t['head']
    tail = t['tail']
    rel = t['rel']
    queries_tail[(head, rel)] = t['tail_predictions_fil']['entities']
    queries_head[(tail, rel)] = t['head_predictions_fil']['entities']

# Check the answers from the new model (head)
n_answers = 0
n_shared_answers = 0
with open(answer_file2 + 'head-fil.pkl', 'rb') as fin:
    query_answers = pickle.load(fin)
    for query in query_answers:
        q = (query['ent'], query['rel'])
        answers2 = query['answers_fil']
        answers1 = queries_head[q]
        a1 = set(answers1)
        a2 = set(answers2)
        inter = a1.intersection(a2)
        n_shared_answers += len(inter)
        n_answers += len(answers2)
print("HEAD: Shared answers {} total answers {}".format(n_shared_answers, n_answers))

# Check the answers from the new model (tail)
n_answers = 0
n_shared_answers = 0
with open(answer_file2 + 'tail-fil.pkl', 'rb') as fin:
    query_answers = pickle.load(fin)
    for query in query_answers:
        q = (query['ent'], query['rel'])
        answers2 = query['answers_fil']
        answers1 = queries_tail[q]
        a1 = set(answers1)
        a2 = set(answers2)
        inter = a1.intersection(a2)
        n_shared_answers += len(inter)
        n_answers += len(answers2)
print("TAIL: Shared answers {} total answers {}".format(n_shared_answers, n_answers))