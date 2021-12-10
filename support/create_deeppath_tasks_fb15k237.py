import sys
import os
import shutil
import json
from support.dataset_fb15k237 import Dataset_FB15k237

dbpath = sys.argv[1]
taskdir = dbpath + '/tasks'
relationsfile = sys.argv[2]
test_gold_standard = sys.argv[3]

dataset = Dataset_FB15k237(include_valid=True)

def create_graph(relname):
    graph = []
    train_pos = []
    for fact in dataset.get_facts():
        head_txt = dataset.get_entity_text(fact[0])
        tail_txt = dataset.get_entity_text(fact[1])
        rel_txt = dataset.get_relation_text(fact[2])
        rel_inv_txt = rel_txt + '_inv'
        line = "{}\t{}\t{}\n".format(head_txt, tail_txt, rel_txt)
        if rel_txt == relname:
            train_pos.append(line)
        else:
            graph.append(line)
            line = "{}\t{}\t{}\n".format(tail_txt, head_txt, rel_inv_txt)
            graph.append(line)
    return graph, train_pos


def create_task(relname, relid, test_facts_complex, test_facts_rotate, test_facts_transe):
    dir_relname = relname[1:]
    dir_relname = dir_relname.replace("/", "@")
    reldir = taskdir + '/' + dir_relname
    if os.path.exists(reldir):
        print("Dir {} already existing".format(reldir))
    else:
        # Create a directory
        os.makedirs(reldir)

    #Copy entity2id.txt
    old_filepath = dbpath + '/entity2id.txt'
    new_filepath = reldir + '/entity2id.txt'
    shutil.copy(old_filepath, new_filepath)

    # Copy relation2id.txt
    old_filepath = dbpath + '/relation2id.txt'
    new_filepath = reldir + '/relation2id.txt'
    shutil.copy(old_filepath, new_filepath)
    graph, train_pos = create_graph(relname)

    fout_graph = reldir + '/graph.txt'
    with open(fout_graph, 'wt') as fout:
        for l in graph:
            fout.write(l)
        fout.close()

    fout_train_pos = reldir + '/train_pos'
    with open(fout_train_pos, 'wt') as fout:
        for l in train_pos:
            fout.write(l)
        fout.close()

    # Write test queries
    fout_test = reldir + '/test.pairs-{}-complex'.format(relid)
    with open(fout_test, 'wt') as fout:
        for pair in test_facts_complex:
            h = pair[0][1:].replace('/','_')
            h_new = "thing$" + h
            t = pair[1][1:].replace('/','_')
            t_new = "thing$" + t
            if pair[2] == 1:
                l_new = '+'
            else:
                l_new = '-'
            line = "{},{}: {}\n".format(h_new, t_new, l_new)
            fout.write(line)
        fout.close()
    fout_test = reldir + '/test.pairs-{}-rotate'.format(relid)
    with open(fout_test, 'wt') as fout:
        for pair in test_facts_rotate:
            h = pair[0][1:].replace('/', '_')
            h_new = "thing$" + h
            t = pair[1][1:].replace('/', '_')
            t_new = "thing$" + t
            if pair[2] == 1:
                l_new = '+'
            else:
                l_new = '-'
            line = "{},{}: {}\n".format(h_new, t_new, l_new)
            fout.write(line)
        fout.close()
    fout_test = reldir + '/test.pairs-{}-transe'.format(relid)
    with open(fout_test, 'wt') as fout:
        for pair in test_facts_transe:
            h = pair[0][1:].replace('/', '_')
            h_new = "thing$" + h
            t = pair[1][1:].replace('/', '_')
            t_new = "thing$" + t
            if pair[2] == 1:
                l_new = '+'
            else:
                l_new = '-'
            line = "{},{}: {}\n".format(h_new, t_new, l_new)
            fout.write(line)
        fout.close()


if __name__ == "__main__":
    # Load the gold standard
    queries = json.load(open(test_gold_standard, 'rt'))
    rel_queries_transe = {}
    rel_queries_complex = {}
    rel_queries_rotate = {}
    for _, query in queries.items():
        rel = query['query']['rel']
        typ = query['query']['type']
        ent = query['query']['ent']
        rel_txt = dataset.get_relation_text(rel)
        print(rel_txt)
        qent_txt = dataset.get_entity_text(ent)
        if rel_txt in rel_queries_transe:
            pairs_transe = rel_queries_transe[rel_txt]
            pairs_complex = rel_queries_complex[rel_txt]
            pairs_rotate = rel_queries_rotate[rel_txt]
        else:
            pairs_transe = []
            pairs_complex = []
            pairs_rotate = []
        for answer in query['annotated_answers']:
            ent_txt = dataset.get_entity_text(answer['entity_id'])
            if answer['checked'] == False:
                label = 0
            else:
                label = 1
            if typ == 0:
                tuple = (ent_txt, qent_txt, label)
            else:
                tuple = (qent_txt, ent_txt, label)
            for m in answer['methods']:
                if m == 'transe':
                    pairs_transe.append(tuple)
                if m == 'complex':
                    pairs_complex.append(tuple)
                if m == 'rotate':
                    pairs_rotate.append(tuple)
        rel_queries_transe[rel_txt] = pairs_transe
        rel_queries_complex[rel_txt] = pairs_complex
        rel_queries_rotate[rel_txt] = pairs_rotate

    with open(relationsfile, 'rt') as fin:
        for rel in fin:
            rel = rel[:-1]
            print("Processing {}".format(rel))
            relid = dataset.get_relation_id(rel)
            create_task(rel, relid, rel_queries_complex[rel], rel_queries_rotate[rel], rel_queries_transe[rel])

