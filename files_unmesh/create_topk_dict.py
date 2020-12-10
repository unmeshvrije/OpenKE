import os
import argparse
from files_unmesh.dynamic_topk import DynamicTopk

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training triples and prepare an estimate of topk for head and tail predictions.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--infile', dest ='infile', type = str, help = 'File containing training triples.', default = "/home/uji300/OpenKE/benchmarks/fb15k237/train2id.txt")
    return parser.parse_args()

args = parse_args()
db = args.db
result_dir = args.result_dir + db + "/"
os.makedirs(result_dir, exist_ok = True)

dyntop = DynamicTopk()
dyntop.populate(args.infile)

dyn_topk_head_filename = result_dir + db + "-dynamic-topk-head.pkl"
dyn_topk_tail_filename = result_dir + db + "-dynamic-topk-tail.pkl"

dyntop.save(dyn_topk_head_filename, dyn_topk_tail_filename)
