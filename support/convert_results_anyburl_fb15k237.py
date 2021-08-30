import sys
from support.dataset_fb15k237 import Dataset_FB15k237
import json

predictions = sys.argv[1]
test_gold_standard = sys.argv[2]

def return_metrics(m):
    if m['tp'] + m['fp'] == 0:
        prec = 0
    else:
        prec = (m['tp'] / (m['tp'] + m['fp']))
    if m['tp'] + m['fn'] == 0:
        rec = 0
    else:
        rec = (m['tp'] / (m['tp'] + m['fn']))
    if prec == 0 and rec == 0:
        f1 = 0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1

def update_maps(method, type, ent, rel, answer, checked):
    t = (ent, rel)
    if method == 'transe':
        if type == 'head':
            m = all_annotations_transe_head
        else:
            m = all_annotations_transe_tail
    elif method == 'complex':
        if type == 'head':
            m = all_annotations_complex_head
        else:
            m = all_annotations_complex_tail
    elif method == 'rotate':
        if type == 'head':
            m = all_annotations_rotate_head
        else:
            m = all_annotations_rotate_tail
    if t in m:
        m[t].append((answer, checked))
    else:
        m[t] = [(answer, checked)]


db = Dataset_FB15k237()

fin = open(predictions, 'rt')
lines = fin.readlines()
head_queries = {}
tail_queries = {}
for i in range(0, len(lines), 3):
    query = lines[i][:-1]
    heads = lines[i + 1][7:-1]
    tails = lines[i + 1][7:-1]

    tokens = query.split(' ')
    h = db.get_entity_id(tokens[0])
    r = db.get_relation_id(tokens[1])
    t = db.get_entity_id(tokens[2])
    if heads != '':
        # Process the heads
        answers_scores = heads.split('\t')
        answers = []
        for i in range(0, 19, 2):
            if i == len(answers_scores) - 1:
                break
            a = answers_scores[i]
            answers.append(db.get_entity_id(a))
        tail_queries[(t, r)] = answers
    if tails != '':
        # Process the tails
        answers_scores = tails.split('\t')
        answers = []
        for i in range(0, 19, 2):
            if i == len(answers_scores) - 1:
                break
            a = answers_scores[i]
            answers.append(db.get_entity_id(a))
        head_queries[(h, r)] = answers

metrics = {}
metrics['transe_head'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['complex_head'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['rotate_head'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['transe_tail'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['complex_tail'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['rotate_tail'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}

queries = json.load(open(test_gold_standard, 'rt'))
for _, query in queries.items():
    rel = query['query']['rel']
    typ = query['query']['type']
    ent = query['query']['ent']
    t = (ent, rel)
    print(db.get_entity_text(ent), db.get_relation_text(rel))
    if typ == 0:
        suf = '_head'
    else:
        suf = '_tail'
    for answer in query['annotated_answers']:
        ans = answer['entity_id']
        real_checked = answer['checked']
        for m in answer['methods']:
            cla_answers = None
            if typ == 1:
                if t in head_queries:
                    cla_answers = head_queries[t]
                else:
                    print("Not found")
            else:
                if t in tail_queries:
                    cla_answers = tail_queries[t]
                else:
                    print("Not found")
            resp = False
            if cla_answers is not None:
                for cla_answer in cla_answers:
                    if cla_answer == ans:
                        resp = True

            if resp == True and real_checked == True:
                cnt = metrics[m + suf]["tp"]
                cnt += 1
                metrics[m + suf]["tp"] = cnt
            elif resp == True and real_checked == False:
                cnt = metrics[m + suf]["fp"]
                cnt += 1
                metrics[m + suf]["fp"] = cnt
            elif resp == False and real_checked == True:
                cnt = metrics[m + suf]["fn"]
                cnt += 1
                metrics[m + suf]["fn"] = cnt
            elif resp == False and real_checked == False:
                cnt = metrics[m + suf]["tn"]
                cnt += 1
                metrics[m + suf]["tn"] = cnt


transe_head = return_metrics(metrics['transe_head'])
transe_tail = return_metrics(metrics['transe_tail'])
print("TRANSE HEAD: PREC {} REC {} F1 {}".format(transe_head[0], transe_head[1], transe_head[2]))
print("TRANSE TAIL: PREC {} REC {} F1 {}".format(transe_tail[0], transe_tail[1], transe_tail[2]))
print("TRANSE: PREC {} REC {} F1 {}".format((transe_head[0] + transe_tail[0])/ 2, (transe_head[1] + transe_tail[1])/ 2, (transe_head[2] + transe_tail[2])/ 2))
complex_head = return_metrics(metrics['complex_head'])
complex_tail = return_metrics(metrics['complex_tail'])
print("COMPLEX HEAD: PREC {} REC {} F1 {}".format(complex_head[0], complex_head[1], complex_head[2]))
print("COMPLEX TAIL: PREC {} REC {} F1 {}".format(complex_tail[0], complex_tail[1], complex_tail[2]))
print("COMPLEX: PREC {} REC {} F1 {}".format((complex_head[0] + complex_tail[0])/ 2, (complex_head[1] + complex_tail[1])/ 2, (complex_head[2] + complex_tail[2])/ 2))
rotate_head = return_metrics(metrics['rotate_head'])
rotate_tail = return_metrics(metrics['rotate_tail'])
print("ROTATE HEAD: PREC {} REC {} F1 {}".format(rotate_head[0], rotate_head[1], rotate_head[2]))
print("ROTATE TAIL: PREC {} REC {} F1 {}".format(rotate_tail[0], rotate_tail[1], rotate_tail[2]))
print("ROTATE: PREC {} REC {} F1 {}".format((rotate_head[0] + rotate_tail[0])/ 2, (rotate_head[1] + rotate_tail[1])/ 2, (rotate_head[2] + rotate_tail[2])/ 2))
