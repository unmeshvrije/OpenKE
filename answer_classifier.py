from abc import ABC, abstractmethod
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import glob
from data_generator import DataGenerator

class AnswerClassifier(ABC):

    def __init__(self, type_prediction, test_queries_file, db, emb_model, topk):
        self.x_test_raw = None
        self.y_test_raw = None
        self.x_test_fil = None
        self.y_test_fil = None
        self.entity_dict = None
        self.relation_dict = None
        self.logfile = None

        self.y_predicted_raw = []
        self.y_predicted_fil = []
        self.y_predicted_fil_abs = []

        self.type_prediction   = type_prediction
        self.db                = db
        self.emb_model         = emb_model
        self.topk              = topk
        self.use_generator     = os.path.isdir(test_queries_file)
        self.test_queries_file = test_queries_file
        self.test_generator = {}
        self.test_labels    = {}
        self.test_queries_answers = {}
        self.init_test_triples(test_queries_file)

    def get_labels(self, list_IDs_temp, folder, type_pred, db="fb15k237", emb_model="transe", topk=10, batch_size=10, dim_x=(1000,10,605), dim_y=(1000,10,1), n_classes=2, type_data="test", shuffle=False, filtered=""):
        y = []

        if type_data == "test":
            filtered = "_"+filtered
        # Generate data
        assert(dim_x[1] == topk)
        N_FEATURES = dim_x[2]
        for i, ID in enumerate(list_IDs_temp):
            # type_data = {"training", "test"}
            # batch_data/fb15k237-transe-test-topk-50-tail_fil-batch-13.pkl
            batch_file = folder + db + "-" + emb_model + "-test-topk-" + str(topk) + "-" + type_pred + filtered+ "-batch-"+str(ID) + ".pkl"

            with open(batch_file, 'rb') as fin:
                training_data = pickle.load(fin)

            yi = np.array(training_data['y_' + type_pred + filtered], dtype = np.int32)
            N = len(yi)
            if N < dim_y[0] * dim_y[1]:
                # padding
                diff = (dim_x[0] * dim_y[1]) - N
                yi = np.vstack([yi, np.zeros([diff])])
                N = dim_y[0] * dim_y[1]

            y.extend(yi)

        return np.array(y)

    def get_queries_answers(self, list_IDs_temp, folder, type_pred, db="fb15k237", emb_model="transe", topk=10, batch_size=10, dim_x=(1000,10,605), dim_y=(1000,10,1), n_classes=2, type_data="test", shuffle=False, filtered=""):
        x = []

        if type_data == "test":
            filtered = "_"+filtered
        # Generate data
        assert(dim_x[1] == topk)
        N_FEATURES = dim_x[2]
        for i, ID in enumerate(list_IDs_temp):
            # type_data = {"training", "test"}
            # batch_data/fb15k237-transe-test-topk-50-tail_fil-batch-13.pkl
            batch_file = folder + db + "-" + emb_model + "-test-topk-" + str(topk) + "-" + type_pred + filtered+ "-batch-"+str(ID) + ".pkl"

            with open(batch_file, 'rb') as fin:
                training_data = pickle.load(fin)

            xi = training_data['x_' + type_pred + filtered]
            N = len(xi)
            #if N < dim_y[0] * dim_y[1]:
                # padding
            #    diff = (dim_x[0] * dim_y[1]) - N
            #    yi = np.vstack([yi, np.zeros([diff])])
            #    N = dim_y[0] * dim_y[1]
            # add [ent, rel, ans]
            for q in xi:
                x.append([q[0], q[1], q[2]])

        return np.array(x)

    def init_batch_test_triples(self):
        N_FEATURES=605
        for rf in ["raw", "fil"]:
            all_files = glob.glob(self.test_queries_file + "*topk-"+str(self.topk) +"*" + self.type_prediction +"_"+ rf + "*.pkl")
            num_files = len(all_files)
            test_ids = np.arange(1, num_files + 1)

            params = {
                'db'        : self.db,
                'emb_model' : self.emb_model,
                'topk'      : self.topk,
                'dim_x'     : (1000, self.topk, N_FEATURES),
                'dim_y'     : (1000, self.topk, 1),
                'batch_size': 1, # TODO : always 1 file so that any number of files would do. int(args.batch_size),
                'n_classes' : 2,
                'type_data' : "test",
                'shuffle'   : False,
                'filtered'  : rf
                }

            self.test_generator[rf] = DataGenerator(test_ids, self.test_queries_file, self.type_prediction, **params)
            self.test_labels[rf]    = self.get_labels(test_ids, self.test_queries_file, self.type_prediction, **params)
            self.test_queries_answers[rf]    = self.get_queries_answers(test_ids, self.test_queries_file, self.type_prediction, **params)
            self.cnt_test_triples = len(self.test_queries_answers[rf])
            print("Size of all test triples = ", self.cnt_test_triples)

    def init_test_triples(self, test_queries_file):
        # Read the test file
        # A test triples file has 3 + 2 +  200*3 features where first three features are
        # (h + r + ans1) + rank  + score + followed by h_bar + r_bar + ans1_bar
        if self.use_generator:
            # open folder and create data generator
            self.init_batch_test_triples()
        else:
            with open(test_queries_file, "rb") as fin:
                data = pickle.load(fin)
            self.x_test_raw = np.array(data['x_' + self.type_prediction + "_raw"])
            self.y_test_raw = np.array(data['y_' + self.type_prediction + "_raw"], dtype = np.int32)
            self.x_test_fil = np.array(data['x_' + self.type_prediction + "_fil"])
            self.y_test_fil = np.array(data['y_' + self.type_prediction + "_fil"], dtype = np.int32)

            self.test_queries_answers["raw"] = self.x_test_raw
            self.test_queries_answers["fil"] = self.x_test_fil
            self.cnt_test_triples = len(self.x_test_raw)
            print("Size of all test triples = ", self.cnt_test_triples)
            self.emb_dim = len(self.x_test_raw[0])

    def init_entity_dict(self, entity_dict_file, rel_dict_file):
        with open(entity_dict_file, 'rb') as fin:
            self.entity_dict = pickle.load(fin)

        with open(rel_dict_file, 'rb') as fin:
            self.relation_dict = pickle.load(fin)


    @abstractmethod
    def set_logfile(self, logfile):
        pass

    @abstractmethod
    def print_answer_entities(self):
        '''
            print stringized entities from ids in the log
        '''
        pass

    @abstractmethod
    def predict(self):
        '''
        For a single test triple (h, r, ans1):

        1. SubgraphClassifier would work as follows
        Use x_test , compute h * r (where * is the operator used by the embedding model)
        Find topK subgraphs, and check if they contain the ans1

        2. LSTMClassifier would work as follows
        Use pre-trained model using training triples

        Both fill y_predicted_raw and y_predicted_fil in the end
        and also y_predicted_fil_abs => to abstain
        '''
        pass

    def results(self):
        print("#" * 20 + "   RAW   " + "#" * 20)
        raw_cnt_tuple = np.unique(self.y_predicted_raw, return_counts = True)
        print("# of predicted : ", raw_cnt_tuple)
        raw_conf_mat = confusion_matrix(self.y_test_raw, self.y_predicted_raw)
        print(raw_conf_mat)
        raw_result = classification_report(self.y_test_raw, self.y_predicted_raw, output_dict = True)
        raw_result['predicted_cnt'] = {}
        raw_result['predicted_cnt']['0'] = raw_cnt_tuple[1][0]
        raw_result['predicted_cnt']['1'] = raw_cnt_tuple[1][1]
        raw_result['predicted_y'] = self.y_predicted_raw
        raw_result['TP'] = raw_conf_mat[0][0]
        raw_result['FP'] = raw_conf_mat[0][1]
        raw_result['FN'] = raw_conf_mat[1][0]
        raw_result['TN'] = raw_conf_mat[1][1]
        print(classification_report(self.y_test_raw, self.y_predicted_raw))

        print("#" * 20 + "   FILTERED   " + "#" * 20)
        fil_cnt_tuple = np.unique(self.y_predicted_fil, return_counts = True)
        print("# of predicted : ", fil_cnt_tuple)
        fil_conf_mat = confusion_matrix(self.y_test_fil, self.y_predicted_fil)
        print(fil_conf_mat)
        filtered_result = classification_report(self.y_test_fil, self.y_predicted_fil, output_dict = True)
        filtered_result['predicted_cnt'] = {}
        filtered_result['predicted_cnt']['0'] = fil_cnt_tuple[1][0]
        filtered_result['predicted_cnt']['1'] = fil_cnt_tuple[1][1]
        filtered_result['predicted_y'] = self.y_predicted_fil
        filtered_result['predicted_y_abs'] = self.y_predicted_fil_abs
        filtered_result['TP'] = fil_conf_mat[0][0]
        filtered_result['FP'] = fil_conf_mat[0][1]
        filtered_result['FN'] = fil_conf_mat[1][0]
        filtered_result['TN'] = fil_conf_mat[1][1]
        print(classification_report(self.y_test_fil, self.y_predicted_fil))

        print("*" * 80)
        if self.entity_dict is not None and self.relation_dict is not None:
            self.print_answer_entities()
        return raw_result, filtered_result
