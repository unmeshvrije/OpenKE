import argparse
from subgraphs import Subgraph
from subgraphs import SUBTYPE
from mlp_classifier import MLPClassifier
from subgraph_classifier import SubgraphClassifier

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', required = True, type = str, help = 'File containing test data.')
    parser.add_argument('--modelfile', dest ='model_file',type = str, help = 'File containing test data.')
    parser.add_argument('--classifier', dest ='classifier',type = str, help = 'Classifier.')
    parser.add_argument('--weightsfile', dest ='weights_file', type = str, help = 'File containing test data.')
    parser.add_argument('--subfile', dest ='sub_file', type = str, help = 'File containing subgraphs metadata.')
    parser.add_argument('--subembfile', dest ='subemb_file', type = str, help = 'File containing subgraphs embeddings.')
    parser.add_argument('--embfile', dest ='emb_file', type = str, help = 'File containing entity embeddings.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--pred', dest ='pred', type = str, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    return parser.parse_args()

args = parse_args()

queries_file_path = args.test_file

if args.classifier == "mlp":
    model_file_path = args.model_file
    model_weights_path = args.weights_file
    # For this to work, queries_file_path must contain 10 (topk) answers present for each triple
    myc = MLPClassifier(args.pred, args.topk, queries_file_path, model_file_path, model_weights_path)
    myc.predict()
    myc.results()
elif args.classifier == "sub":
    emb_file = args.emb_file
    sub_file = args.sub_file
    subemb_file = args.subemb_file
    mys = SubgraphClassifier(args.pred, args.topk, queries_file_path, emb_file, sub_file, subemb_file)
    mys.predict()
    mys.results()
