import sys
import pickle


def convert(indata, is_test = True):
    if is_test:
        x_head_fil = indata['x_head_fil']
        x_tail_fil = indata['x_tail_fil']
    else:
        x_head_fil = indata['x_head']
        x_tail_fil = indata['x_tail']

    i = 0
    out_head = []
    while i < len(x_head_fil):
        entry = None
        answers = []
        for j in range(10):
            emb = x_head_fil[i]
            meta = emb[:5]
            ent = meta[0]
            rel = meta[1]
            ans = meta[2]
            score = meta[3]
            emb = emb[5:]
            answers.append({'entity_id' : ans, 'score' : score })
            if j == 0:
                entry = {'type': 0, 'ent': ent, 'rel': rel}
            i += 1
        entry['answers_fil'] = answers
        entry['answers_raw'] = answers
        out_head.append(entry)

    i = 0
    out_head = []
    while i < len(x_head_fil):
        entry = None
        answers = []
        for j in range(10):
            emb = x_head_fil[i]
            meta = emb[:5]
            ent = meta[0]
            rel = meta[1]
            ans = meta[2]
            score = meta[3]
            emb = emb[5:]
            answers.append({'entity_id': ans, 'score': score})
            if j == 0:
                entry = {'type': 0, 'ent': ent, 'rel': rel}
            i += 1
        entry['answers_fil'] = answers
        entry['answers_raw'] = answers
        out_head.append(entry)


    queries_head = {}
    queries_tail = {}
    out_head = []
    out_tail = []
    for t in indata['x']:
        head = t['head']
        tail = t['tail']
        rel = t['rel']
        answer_fil_tail = t['tail_predictions_fil']['entities']
        answer_fil_head = t['head_predictions_fil']['entities']
        answer_raw_tail = t['tail_predictions_raw']['entities']
        answer_raw_head = t['head_predictions_raw']['entities']
        if (head, rel) not in queries_tail:
            queries_tail[(head, rel)] = None
            q = {'type': 1, 'ent': head, 'rel': rel}
            q['answers_fil'] = answer_fil_tail
            q['answers_raw'] = answer_raw_tail
            out_tail.append(q)
        if (tail, rel) not in queries_head:
            queries_head[(tail, rel)] = None
            q = {'type': 0, 'ent': tail, 'rel': rel}
            q['answers_fil'] = answer_fil_head
            q['answers_raw'] = answer_raw_head
            out_head.append(q)


old_test_file = sys.argv[1]
testdata = pickle.load(open(old_test_file, 'rb'))
convert(testdata)


new_file_head = new_file + 'head-fil.pkl'
with open(new_file_head, 'wb') as fout:
    pickle.dump(out_head, fout)
new_file_tail = new_file + 'tail-fil.pkl'
with open(new_file_tail, 'wb') as fout:
    pickle.dump(out_tail, fout)
