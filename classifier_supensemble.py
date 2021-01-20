import supervised_classifier
from support.utils import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

FALSE=0
TRUE=1

class Classifier_SuperEnsemble(supervised_classifier.Supervised_Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 topk,
                 results_dir,
                 classifiers,
                 embedding_model_name,
                 model_path=None):
        self.classifiers = classifiers
        self.topk = topk
        self.dataset_name = dataset.get_name()
        self.embedding_model_name = embedding_model_name
        self.type_prediction = type_prediction
        self.result_dir = results_dir
        self.test_annotations = None
        if model_path is not None:
            print("Loading existing model {} ...".format(model_path))
            model = joblib.load(model_path)
            self.set_model(model)
        else:
            self.init_model(embedding_model_name, None)

    def get_name(self):
        return "SupEnsemble"

    def init_model(self, embedding_model, hyper_params):
        self.model = RandomForestClassifier(max_depth=2, random_state=0)
        #self.model = SVC()

    def create_training_data(self, queries_with_answers):
        classifiers_annotations = load_classifier_annotations(self.classifiers,
                                                                    self.result_dir,
                                                                    self.dataset_name,
                                                                    self.embedding_model_name,
                                                                    "train",
                                                                    self.topk,
                                                                    self.type_prediction,
                                                                    return_scores=True)
        print("  Creating training data ...")
        training_data_X = []
        training_data_Y = []
        for query in queries_with_answers:
            typ = query['query']['type']
            ent = query['query']['ent']
            rel = query['query']['rel']
            ans = query['annotated_answers']
            assert(query['valid_annotations'] == True)
            assert((ent, rel) in classifiers_annotations)
            answers_X = classifiers_annotations[(ent, rel)]
            for _, a in enumerate(ans):
                ans_id = a['entity_id']
                if a['checked']:
                    Y = 1
                else:
                    Y = 0
                assert(ans_id in answers_X)
                X = answers_X[ans_id]
                training_data_X.append(X)
                training_data_Y.append(Y)
        training_data_X = np.asarray(training_data_X)
        training_data_Y = np.asarray(training_data_Y)
        count_true = np.zeros(len(self.classifiers), dtype=int)
        count_false = np.zeros(len(self.classifiers), dtype=int)
        for t in training_data_X:
            for i, a in enumerate(t):
                if a == TRUE:
                    count_true[i] += 1
                else:
                    count_false[i] += 1
        for i, classifier in enumerate(self.classifiers):
            print("Classifier {} TRUE {} FALSE {}".format(classifier, count_true[i], count_false[i]))
        return {'X' : training_data_X, 'Y' : training_data_Y }

    def train(self, training_data, valid_data, model_path):
        #X, y = make_classification(n_samples=1000,
        #                           n_features=4,
        #                           n_informative = 2,
        #                           n_redundant = 0,
        #                           random_state = 0,
        #                           shuffle = False)
        X = training_data['X']
        Y = training_data['Y']
        self.model.fit(X, Y)
        if model_path is not None:
            joblib.dump(self.model, model_path)

    def predict(self, query_with_answers, type_answers, provenance_test="test"):
        if self.test_annotations is None:
            self.test_annotations = load_classifier_annotations(self.classifiers,
                                                                self.result_dir,
                                                                self.dataset_name,
                                                                self.embedding_model_name,
                                                                provenance_test,
                                                                self.topk,
                                                                self.type_prediction,
                                                                return_scores=True)
        # for q in query_with_answers:
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 0 or self.type_prediction == 'tail')
        assert (typ == 1 or self.type_prediction == 'head')
        assert ((ent, rel) in self.test_annotations)

        # Check that the output matches the filtered answers
        filtered_answers = set()
        for a in query_with_answers[type_answers]:
            filtered_answers.add(a['entity_id'])

        annotated_answers = []
        for entity_id, labels in self.test_annotations[(ent, rel)].items():
            assert (entity_id in filtered_answers)
            assert (len(labels) == len(self.classifiers))
            #for i, _ in enumerate(labels):
                #if labels[i] == True:
                #    labels[i] = 1
                #else:
                #    labels[i] = 0
            l = np.asarray(labels)
            lr = l.reshape(1, -1)
            out = self.get_model().predict(lr)
            checked = out[0] == 1
            annotated_answers.append({'entity_id': entity_id, 'checked': checked, 'score': out[0]})
        return annotated_answers