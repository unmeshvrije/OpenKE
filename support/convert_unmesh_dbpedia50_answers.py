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
    out_training_head = []
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
    out_tail = []
    out_training_tail = []
    while i < len(x_tail_fil):
        entry = None
        answers = []
        for j in range(10):
            emb = x_tail_fil[i]
            meta = emb[:5]
            ent = meta[0]
            rel = meta[1]
            ans = meta[2]
            score = meta[3]
            emb = emb[5:]
            answers.append({'entity_id': ans, 'score': score})
            if j == 0:
                entry = {'type': 1, 'ent': ent, 'rel': rel}
            i += 1
        entry['answers_fil'] = answers
        entry['answers_raw'] = answers
        out_tail.append(entry)
    return out_head, out_training_head, out_tail, out_training_tail

old_file = sys.argv[1]
testdata = pickle.load(open(old_file, 'rb'))
out_answer_head, out_training_head, out_answer_tail, out_training_tail = convert(testdata, is_test=False)
new_answers_file = sys.argv[2]
new_file_head = new_answers_file + 'head-fil.pkl'
with open(new_file_head, 'wb') as fout:
    pickle.dump(out_answer_head, fout)
new_file_tail = new_answers_file + 'tail-fil.pkl'
with open(new_file_tail, 'wb') as fout:
    pickle.dump(out_answer_tail, fout)

#new_training_data_file = sys.argv[3]
#new_file_head = new_training_data_file + 'head-fil.pkl'
#with open(new_file_head, 'wb') as fout:
#    pickle.dump(out_training_head, fout)
#new_file_tail = new_training_data_file + 'tail-fil.pkl'
#with open(new_file_tail, 'wb') as fout:
#    pickle.dump(out_training_tail, fout)