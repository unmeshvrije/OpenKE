import pickle
from tqdm import tqdm

def get_filename_queries(db, mode, type_prediction):
    return db + '-' + mode + '-' + type_prediction + '.json'

def get_filename_model(db, model):
    return db + "-" + model + ".pt"

def get_filename_answers(db, model, mode, topk, type_prediction, suf = ''):
    return db + "-answers-" + model + "-" + mode + "-" + str(topk) + "-" + type_prediction + suf + ".pkl"

def get_filename_answer_annotations(db, model, mode, topk, type_prediction, suf = ''):
    return db + "-annotated-answers-" + model + "-" + mode + "-" + str(topk) + "-" + type_prediction + suf + ".pkl"

def get_filename_training_data(db, classifier, topk, type_prediction):
    return db + '-' + classifier + '-' + str(topk) + '-' + type_prediction + ".pkl"

def get_filename_classifier_model(db, classifier, topk, type_prediction, extra_params=""):
    if extra_params == '':
        return db + '-' + classifier + '-' + str(topk) + '-' + type_prediction + ".pt"
    else:
        return db + '-' + classifier + '-' + str(topk) + '-' + type_prediction + '-' + extra_params + ".pt"

def get_filename_classifier_labels(db, classifier, topk, type_prediction, extra_params=""):
    if extra_params == "":
        return db + '-' + classifier + '-' + str(topk) + '-' + type_prediction + ".json"
    else:
        return db + '-' + classifier + '-' + str(topk) + '-' + type_prediction + '-' + extra_params + ".json"

def get_filename_gold(db, topk):
    return "gold-annotations.json"

def get_filename_results(db, model, mode, topk, type_prediction, suf = ''):
    return db + "-gold-results-" + model + "-" + mode + "-" + str(topk) + "-" + type_prediction + suf + ".json"


def load_classifier_annotations(classifiers, result_dir, dataset_name, embedding_model_name, mode, topk, type_prediction):
    classifiers_annotations = {}
    for idx, classifier in enumerate(classifiers):
        print("  Loading annotations from classifier {} ...".format(classifier))
        suf = '-' + classifier
        file_name = get_filename_answer_annotations(dataset_name, embedding_model_name, "test", topk,
                                                    type_prediction, suf)
        file_path = result_dir + '/' + dataset_name + '/annotations/' + file_name
        classifier_annotations = pickle.load(open(file_path, 'rb'))
        for a in tqdm(classifier_annotations):
            ent = a['query']['ent']
            rel = a['query']['rel']
            typ = a['query']['type']
            assert (typ == 0 or type_prediction == 'tail')
            assert (typ == 1 or type_prediction == 'head')
            if (ent, rel) not in classifiers_annotations:
                assert (idx == 0)
                classifiers_annotations[(ent, rel)] = {}
            else:
                assert (idx != 0)
            answer_set = classifiers_annotations[(ent, rel)]
            for answer in a['annotated_answers']:
                if answer['entity_id'] not in answer_set:
                    assert (idx == 0)
                    answer_set[answer['entity_id']] = [answer['checked']]
                else:
                    assert (idx != 0)
                    answer_set[answer['entity_id']].append(answer['checked'])
    for key, ans in classifiers_annotations.items():
        assert (len(ans) == topk)
    return classifiers_annotations