import argparse
from support.dataset_fb15k237 import Dataset_FB15k237
from support.utils import *
from support.embedding_model import  Embedding_Model
import pickle
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--classifier', dest='classifier', type=str, choices=['mlp','mlp_multi','lstm','conv', 'snorkel', 'trans'])
    parser.add_argument('--name_signals', dest='name_signals', help='name of the signals (classifiers) to use when multiple signals should be combined', type=str, required=False, default="mlp_multi,lstm,conv,path,sub")
    parser.add_argument('--result_dir', dest ='result_dir', type = str, help = 'Output dir.')
    parser.add_argument('--db', dest = 'db', type = str, default = "fb15k237", choices=['fb15k237'])
    parser.add_argument('--topk', dest='topk', type=int, default=10)
    parser.add_argument('--model', dest='model', type=str, default="transe", choices=['complex', 'rotate', 'transe'])
    parser.add_argument('--type_prediction', dest='type_prediction', type=str, default="head", choices=['head', 'tail'])

    parser.add_argument('--snorkel_low_threshold', dest='snorkel_low_threshold', type=str,
                        default="0.2,0.2,0.2,0,0")
    parser.add_argument('--snorkel_high_threshold', dest='snorkel_high_threshold', type=str,
                        default="0.6,0.6,0.6,0.5,0.5")
    parser.add_argument('--mlp_n_hidden_units', dest='mlp_n_hidden_units', type=int, default=100)
    parser.add_argument('--mlp_dropout', dest='mlp_dropout', type=float, default=0.2)
    parser.add_argument('--lstm_n_hidden_units', dest='lstm_n_hidden_units', type=int, default=100)
    parser.add_argument('--lstm_dropout', dest='lstm_dropout', type=float, default=0.2)

    return parser.parse_args()

args = parse_args()

# How much data should I use as validation dataset?
use_valid_data = 0.05

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
if args.classifier != 'snorkel':
    embedding_model = Embedding_Model(args.result_dir, embedding_model_typ, dataset)

# Load the classifier
if args.classifier == 'mlp':
    from classifier_mlp import Classifier_MLP
    hyper_params = { "n_units" : args.mlp_n_hidden_units, "dropout" : args.mlp_dropout }
    classifier = Classifier_MLP(dataset, args.type_prediction, args.result_dir, embedding_model, hyper_params=hyper_params)
elif args.classifier == 'mlp_multi':
    from classifier_mlp_multi import Classifier_MLP_Multi
    hyper_params = {"n_units": args.mlp_n_hidden_units, "dropout": args.mlp_dropout}
    classifier = Classifier_MLP_Multi(dataset, args.type_prediction, args.result_dir, embedding_model, hyper_params=hyper_params)
elif args.classifier == 'lstm':
    from classifier_lstm import Classifier_LSTM
    hyper_params = {"n_units": args.lstm_n_hidden_units, "dropout": args.lstm_dropout}
    classifier = Classifier_LSTM(dataset, args.type_prediction, args.result_dir, embedding_model, hyper_params=hyper_params)
elif args.classifier == 'conv':
    from classifier_conv import Classifier_Conv
    classifier = Classifier_Conv(dataset, args.type_prediction, args.result_dir, embedding_model)
elif args.classifier == 'trans':
    from classifier_transformer import Classifier_Transformer
    classifier = Classifier_Transformer(dataset, args.type_prediction, args.result_dir, embedding_model)
elif args.classifier == 'snorkel':
    from classifier_snorkel import Classifier_Snorkel
    signals = args.name_signals.split(",")
    use_valid_data = 0 # With snorkel, I don't need validation data
    classifier = Classifier_Snorkel(dataset, args.type_prediction, args.topk, args.result_dir, signals, embedding_model_typ)
else:
    raise Exception('Not supported')

# Get the output file for the model
model_dir = args.result_dir + '/' + args.db + '/models'
model_filename = get_filename_classifier_model(args.db, args.classifier, args.topk, args.type_prediction)
model_full_path_name = model_dir + '/' + model_filename

# Take 10% out and use it for validation
if use_valid_data == 0.0:
    valid_data = None
else:
    n_data_points = len(training_data)
    training_data, valid_data = train_test_split(training_data, test_size=use_valid_data)

# Train a model
classifier.train(training_data, valid_data, model_full_path_name)