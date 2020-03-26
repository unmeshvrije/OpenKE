import argparse
from subgraphs import Subgraph
from subgraphs import SUBTYPE
from mlp_classifier import MLPClassifier
from subgraph_classifier import SubgraphClassifier

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', type = str, help = 'File containing test queries.')
    parser.add_argument('--trainfile', dest ='train_file', type = str, help = 'File containing training triples.')
    parser.add_argument('--modelfile', dest ='model_file',type = str, help = 'File containing test data.')
    parser.add_argument('--classifier', dest ='classifier', required = True, type = str, help = 'Classifier.')
    parser.add_argument('--weightsfile', dest ='weights_file', type = str, help = 'File containing test data.')
    parser.add_argument('--subfile', dest ='sub_file', type = str, help = 'File containing subgraphs metadata.')
    parser.add_argument('--subembfile', dest ='subemb_file', type = str, help = 'File containing subgraphs embeddings.')
    parser.add_argument('--embfile', dest ='emb_file', type = str, help = 'File containing entity embeddings.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--model', dest ='model',type = str, default = "transe", help = 'Embedding model name.')
    parser.add_argument('--pred', dest ='pred', type = str, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    parser.add_argument('-stp', '--subgraph-threshold-percentage',dest ='sub_threshold', type = float, help = '% of top subgraphs to check the correctness of answers')
    return parser.parse_args()

args = parse_args()

queries_file_path = args.test_file

if args.classifier == "mlp" or args.classifier == "lstm":
    model_file_path = args.model_file
    model_weights_path = args.weights_file
    # For this to work, queries_file_path must contain 10 (topk) answers present for each triple
    myc = MLPClassifier(args.pred, args.topk, queries_file_path, model_file_path, model_weights_path)
    myc.init_entity_dict('/var/scratch2/uji300/kbs/fb15k237-id-to-entity.pkl', '/var/scratch2/uji300/kbs/fb15k237-id-to-relation.pkl')
    myc.predict()
    myc.results()
elif args.classifier == "sub":
    emb_file = args.emb_file
    sub_file = args.sub_file
    subemb_file = args.subemb_file
    mys = SubgraphClassifier(args.pred, args.topk, queries_file_path, emb_file, sub_file, subemb_file, args.model, args.train_file, args.sub_threshold)
    mys.predict()
    mys.results()
    print("fil 1s = {}, 0s = {}".format(mys.y_predicted_fil.count(1), mys.y_predicted_fil.count(0)))
    #print("raw 1s = {}, 0s = {}".format(mys.y_predicted_raw.count(1), mys.y_predicted_raw.count(0)))
