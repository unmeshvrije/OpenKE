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