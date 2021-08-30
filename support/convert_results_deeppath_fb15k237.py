import sys
from support.dataset_fb15k237 import Dataset_FB15k237
import os
import json

inputdir = sys.argv[1]
gold_annotation_file = sys.argv[2]
dataset = Dataset_FB15k237(include_valid=True)

all_annotations_transe_head = {}
all_annotations_transe_tail = {}
all_annotations_complex_head = {}
all_annotations_complex_tail = {}
all_annotations_rotate_head = {}
all_annotations_rotate_tail = {}

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

for f in os.listdir(inputdir):
    if f.startswith('results-'):
        if 'transe' in f:
            method = 'transe'
        elif 'rotate' in f:
            method = 'rotate'
        elif 'complex' in f:
            method = 'complex'
        relid = int(f[9 + len(method):f.rfind('.')])
        with open(inputdir + '/' + f, 'rt') as fin:
            results = json.load(fin)
            print(len(results), f, dataset.get_relation_text(relid))
            ts = set()
            to = set()
            for annotation in results:
                s_txt = annotation['query'][0]
                s_txt = '/' + s_txt.replace('m_','m/', 1)
                o_txt = annotation['query'][1]
                o_txt = '/' + o_txt.replace('m_', 'm/', 1)
                sid = dataset.get_entity_id(s_txt)
                oid = dataset.get_entity_id(o_txt)
                checked = annotation['checked']
                ts.add((oid,relid))
                to.add((sid, relid))
                update_maps(method, 'head', oid, relid, sid, checked)
                update_maps(method, 'tail', sid, relid, oid, checked)

metrics = {}
metrics['transe_head'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['complex_head'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['rotate_head'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['transe_tail'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['complex_tail'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
metrics['rotate_tail'] = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}

# Read the test gold annotation
with open(gold_annotation_file, 'rt') as fin:
    queries = json.load(fin)
    queries = [q for _, q in queries.items()]
    for query in queries:
        if query['valid_annotations'] == False:
            continue
        ent = query['query']['ent']
        rel = query['query']['rel']
        typ = query['query']['type']
        t = (ent, rel)
        if typ == 0:
            suf = '_head'
        else:
            suf = '_tail'
        for answer in query['annotated_answers']:
            ans = answer['entity_id']
            real_checked = answer['checked']
            for m in answer['methods']:
                cla_answers = None
                if m == 'transe':
                    if type == 0:
                        if t in all_annotations_transe_head:
                            cla_answers = all_annotations_transe_head[t]
                    else:
                        if t in all_annotations_transe_tail:
                            cla_answers = all_annotations_transe_tail[t]
                elif m == 'complex':
                    if type == 0:
                        if t in all_annotations_complex_head:
                            cla_answers = all_annotations_complex_head[t]
                    else:
                        if t in all_annotations_complex_tail:
                            cla_answers = all_annotations_complex_tail[t]
                elif m == 'rotate':
                    if type == 0:
                        if t in all_annotations_rotate_head:
                            cla_answers = all_annotations_rotate_head[t]
                    else:
                        if t in all_annotations_rotate_tail:
                            cla_answers = all_annotations_rotate_tail[t]
                resp = False
                if cla_answers is not None:
                    for cla_answer in cla_answers:
                        if cla_answer[0] == ans:
                            resp = cla_answer[1]
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
print(metrics)

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




