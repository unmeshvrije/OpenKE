import os
import pickle
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
    parser.add_argument('--entdict', dest ='ent_dict', type = str, default = '/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-entity.pkl',help = 'entity id dictionary.')
    parser.add_argument('--reldict', dest ='rel_dict', type = str, default = '/var/scratch2/uji300/OpenKE-results/fb15k237/misc/fb15k237-id-to-relation.pkl',help = 'relation id dictionary.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--model', dest ='model',type = str, default = "transe", help = 'Embedding model name.')
    parser.add_argument('--pred', dest ='pred', type = str, required = True, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    parser.add_argument('-stp', '--subgraph-threshold-percentage', dest ='sub_threshold', default = 0.1, type = float, help = '% of top subgraphs to check the correctness of answers.')
    parser.add_argument('-th', '--threshold',dest ='threshold', type = float, default = 0.5, help = 'Probability value that decides the boundary between class 0 and 1.')
    return parser.parse_args()

args = parse_args()

result_dir =  args.result_dir + args.db + "/out/"
log_dir =  args.result_dir + args.db + "/logs/"
os.makedirs(result_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
queries_file_path = args.test_file

if args.classifier == "mlp" or args.classifier == "lstm":
    model_file_path = args.model_file
    model_weights_path = args.weights_file
    # For this to work, queries_file_path must contain 10 (topk) answers present for each triple
    myc = MLPClassifier(args.pred, args.topk, queries_file_path, model_file_path, model_weights_path, threshold=args.threshold)

    # entity dict is the id to string dictionary for entities
    myc.init_entity_dict(args.ent_dict, args.rel_dict)

    # set log file
    base_name = os.path.basename(model_file_path).rsplit('.', maxsplit=1)[0]
    logfile = log_dir + base_name + ".log"
    myc.set_logfile(logfile)

    # prediction
    myc.predict()
    raw_result, fil_result = myc.results()

    # Pickle the output
    output_file = result_dir + base_name + ".out"
    result_dict = {}
    result_dict['raw'] = raw_result
    result_dict['fil'] = fil_result
    with open(output_file, 'wb') as fout:
        pickle.dump(result_dict, fout, protocol = pickle.HIGHEST_PROTOCOL)
elif args.classifier == "sub":
    emb_file = args.emb_file
    sub_file = args.sub_file
    subemb_file = args.subemb_file
    mys = SubgraphClassifier(args.pred, args.topk, queries_file_path, emb_file, sub_file, subemb_file, args.model, args.train_file, args.sub_threshold)

    # entity dict is the id to string dictionary for entities
    mys.init_entity_dict(args.ent_dict, args.rel_dict)

    # set log file
    base_name = os.path.basename(sub_file).rsplit('.', maxsplit=1)[0]
    logfile = log_dir + base_name + ".log"
    mys.set_logfile(logfile)

    mys.predict()
    raw_result, fil_result = mys.results()
