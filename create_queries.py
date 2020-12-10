import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--test_file', dest='test_file', type=str, help='File containing the triples to be converted into queries.')
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237")
    parser.add_argument('--mode', dest = 'mode', type = str, default = "test", choices = ['train', 'valid', 'test'])
    return parser.parse_args()
args = parse_args()

queries_head = set()
queries_tail = set()
with open(args.test_file, 'rt') as fin:
    ntriples = int(fin.readline())
    for l in fin:
        tkns = l.split(' ')
        h = int(tkns[0])
        t = int(tkns[1])
        r = int(tkns[2])
        queries_head.add((t, r))
        queries_tail.add((h, r))

queries_head = list(queries_head)
queries_tail = list(queries_tail)
output_dir = args.result_dir + '/' + args.db + '/queries'
out_file_head = args.db + '-' + args.mode + '-head.json'
with open(output_dir + '/' + out_file_head, 'wt') as fout:
    o = []
    for i, q in enumerate(queries_head):
        o.append({'type' : 0, 'ent' : q[0], 'rel' : q[1] })
    json.dump(o, fout)
out_file_tail = args.db + '-' + args.mode + '-tail.json'
with open(output_dir + '/' + out_file_tail, 'wt') as fout:
    o = []
    for i, q in enumerate(queries_tail):
        o.append({'type' : 1, 'ent' : q[0], 'rel' : q[1] })
    json.dump(o, fout)
