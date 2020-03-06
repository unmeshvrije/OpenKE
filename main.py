import argparse
from mlp_classifier import MLPClassifier

def parse_args():
    parser = argparse.ArgumentParser(description = 'Read training/test file and run LSTM training or test.')
    parser.add_argument('--testfile', dest ='test_file', required = True, type = str, help = 'File containing test data.')
    parser.add_argument('--modelfile', dest ='model_file', required = True, type = str, help = 'File containing test data.')
    parser.add_argument('--weightsfile', dest ='weights_file', required = True, type = str, help = 'File containing test data.')
    parser.add_argument('-rd', '--result-dir', dest ='result_dir', type = str, default = "/var/scratch2/uji300/OpenKE-results/",help = 'Output dir.')
    parser.add_argument('--topk', dest = 'topk', required = True, type = int, default = 10)
    parser.add_argument('--db', required = True, dest = 'db', type = str, default = None)
    parser.add_argument('--pred', dest ='pred', type = str, choices = ['head', 'tail'], help = 'Prediction type (head/tail)')
    return parser.parse_args()

args = parse_args()

triples_file_path = args.test_file
model_file_path = args.model_file
model_weights_path = args.weights_file
# For this to work, triples_file_path must contain 10 (topk) answers present for each triple
myc = MLPClassifier("tail", 10, triples_file_path, model_file_path, model_weights_path)
myc.predict()
myc.results()

