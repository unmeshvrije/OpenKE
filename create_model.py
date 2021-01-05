import argparse
from support.dataset_fb15k237 import Dataset_FB15k237
from support.utils import *
from support.embedding_model import  Embedding_Model
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--classifier', dest='classifier', type=str, choices=['mlp','mlp_multi'])
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])
    return parser.parse_args()

args = parse_args()

# Load the dataset
dataset = None
training_data_dir = args.result_dir + '/' + args.db + '/training_data/'
if args.db == 'fb15k237':
    dataset = Dataset_FB15k237()
    annotations_filename = get_filename_training_data(args.db, args.classifier, args.topk, args.type_prediction)
    with open(training_data_dir + '/' + annotations_filename, 'rb') as fin:
        training_data = pickle.load(fin)

# Load the embedding model
embedding_model_typ = args.model
embedding_model = Embedding_Model(args.result_dir, embedding_model_typ, dataset)

# Load the classifier
if args.classifier == 'mlp':
    from classifier_mlp import Classifier_MLP
    classifier = Classifier_MLP(dataset, args.type_prediction, args.result_dir, embedding_model)
elif args.classifier == 'mlp_multi':
    from classifier_mlp_multi import Classifier_MLP_Multi
    classifier = Classifier_MLP_Multi(dataset, args.type_prediction, args.result_dir, embedding_model)
else:
    raise Exception('Not supported')

# Get the output file for the model
model_dir = args.result_dir + '/' + args.db + '/models'
model_filename = get_filename_classifier_model(args.db, args.classifier, args.topk, args.type_prediction)
model_full_path_name = model_dir + '/' + model_filename

# Train a model
classifier.train(training_data, model_full_path_name)