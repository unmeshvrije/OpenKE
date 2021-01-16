import argparse
import json
from support.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--mode', dest = 'mode', type = str, default = "test", choices = ['train', 'valid', 'test'])
    return parser.parse_args()
args = parse_args()

queries_head = set()
queries_tail = set()

if args.db == 'fb15k237':
    if args.mode == 'test':
        input_file = 'benchmarks/fb15k237/test2id.txt'
        with open(input_file, 'rt') as fin:
            _ = int(fin.readline())
            for l in fin:
                tkns = l.split(' ')
                h = int(tkns[0])
                t = int(tkns[1])
                r = int(tkns[2])
                queries_head.add((t, r))
                queries_tail.add((h, r))

    elif args.mode == 'train':
        input_file = 'benchmarks/fb15k237/train2id.txt'
        with open(input_file, 'rt') as fin:
            _ = int(fin.readline())
            for l in fin:
                tkns = l.split(' ')
                h = int(tkns[0])
                t = int(tkns[1])
                r = int(tkns[2])
                queries_head.add((t, r))
                queries_tail.add((h, r))
        input_file = 'benchmarks/fb15k237/valid2id.txt'
        with open(input_file, 'rt') as fin:
            _ = int(fin.readline())
            for l in fin:
                tkns = l.split(' ')
                h = int(tkns[0])
                t = int(tkns[1])
                r = int(tkns[2])
                queries_head.add((t, r))
                queries_tail.add((h, r))
    else:
        raise Exception("Not supported")
elif args.db == 'dbpedia50':
    if args.mode == 'test':
        input_file = 'benchmarks/dbpedia50/test2id.txt'
        with open(input_file, 'rt') as fin:
            _ = int(fin.readline())
            for l in fin:
                tkns = l.split(' ')
                h = int(tkns[0])
                t = int(tkns[1])
                r = int(tkns[2])
                queries_head.add((t, r))
                queries_tail.add((h, r))

    elif args.mode == 'train':
        input_file = 'benchmarks/dbpedia50/train2id.txt'
        with open(input_file, 'rt') as fin:
            _ = int(fin.readline())
            for l in fin:
                tkns = l.split(' ')
                h = int(tkns[0])
                t = int(tkns[1])
                r = int(tkns[2])
                queries_head.add((t, r))
                queries_tail.add((h, r))
        input_file = 'benchmarks/dbpedia50/valid2id.txt'
        with open(input_file, 'rt') as fin:
            _ = int(fin.readline())
            for l in fin:
                tkns = l.split(' ')
                h = int(tkns[0])
                t = int(tkns[1])
                r = int(tkns[2])
                queries_head.add((t, r))
                queries_tail.add((h, r))
    else:
        raise Exception("Not supported")
else:
    raise Exception("Not supported")


queries_head = list(queries_head)
queries_tail = list(queries_tail)
output_dir = args.result_dir + '/' + args.db + '/queries'
out_file_head = get_filename_queries(args.db, args.mode, 'head')
with open(output_dir + '/' + out_file_head, 'wt') as fout:
    o = []
    for i, q in enumerate(queries_head):
        o.append({ 'type' : 0, 'ent' : q[0], 'rel' : q[1] })
    json.dump(o, fout)
out_file_tail = get_filename_queries(args.db, args.mode, 'tail')
with open(output_dir + '/' + out_file_tail, 'wt') as fout:
    o = []
    for i, q in enumerate(queries_tail):
        o.append({ 'type' : 1, 'ent' : q[0], 'rel' : q[1] })
    json.dump(o, fout)
