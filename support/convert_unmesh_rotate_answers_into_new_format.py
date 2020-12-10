import sys
import json
import pickle

old_file = sys.argv[1]
new_file = sys.argv[2]

testdata = json.load(open(old_file, 'rt'))
queries_head = {}
queries_tail = {}
out_head = []
out_tail = []
for t in testdata:
    head = t['head']
    tail = t['tail']
    rel = t['rel']
    answer_fil_tail = t['tail_predictions_fil']['entities']
    answer_fil_head = t['head_predictions_fil']['entities']
    answer_raw_tail = t['tail_predictions_raw']['entities']
    answer_raw_head = t['head_predictions_raw']['entities']
    if (head, rel) not in queries_tail:
        queries_tail[(head, rel)] = None
        q = { 'type' : 1, 'ent': head, 'rel': rel }
        q['answers_fil'] = answer_fil_tail
        q['answers_raw'] = answer_raw_tail
        out_tail.append(q)
    if (tail, rel) not in queries_head:
        queries_head[(tail, rel)] = None
        q = {'type': 0, 'ent': tail, 'rel': rel}
        q['answers_fil'] = answer_fil_head
        q['answers_raw'] = answer_raw_head
        out_head.append(q)

new_file_head = new_file + 'head-fil.pkl'
with open(new_file_head, 'wb') as fout:
    pickle.dump(out_head, fout)
new_file_tail = new_file + 'tail-fil.pkl'
with open(new_file_tail, 'wb') as fout:
    pickle.dump(out_tail, fout)

