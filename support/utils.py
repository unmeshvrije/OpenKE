import pickle
from tqdm import tqdm

def get_filename_queries(db, mode, type_prediction):
    return db + '-' + mode + '-' + type_prediction + '.json'

def get_filename_model(db, model, suf = '.pt'):
    return db + "-" + model + suf

def get_filename_answers(db, model, mode, topk, type_prediction, suf = ''):
    return db + "-answers-" + model + "-" + mode + "-" + str(topk) + "-" + type_prediction + suf + ".pkl"

def get_filename_answer_annotations(db, model, mode, topk, type_prediction, suf = ''):
    return db + "-annotated-answers-" + model + "-" + mode + "-" + str(topk) + "-" + type_prediction + suf + ".pkl"

def get_filename_training_data(db, model, classifier, topk, type_prediction):
    return db + '-' + model + '-' + classifier + '-' + str(topk) + '-' + type_prediction + ".pkl"

def get_filename_classifier_model(db, model, classifier, topk, type_prediction, extra_params=""):
    if extra_params == '':
        return db + '-' + model + '-' + classifier + '-' + str(topk) + '-' + type_prediction + ".pt"
    else:
        return db + '-' + model + '-' + classifier + '-' + str(topk) + '-' + type_prediction + '-' + extra_params + ".pt"

def get_filename_classifier_labels(db, classifier, topk, type_prediction, extra_params=""):
    if extra_params == "":
        return db + '-' + classifier + '-' + str(topk) + '-' + type_prediction + ".json"
    else:
        return db + '-' + classifier + '-' + str(topk) + '-' + type_prediction + '-' + extra_params + ".json"

def get_filename_gold(db, topk, suf=''):
    return "gold-annotations{}.json".format(suf)

def get_filename_results(db, model, mode, topk, type_prediction, suf = ''):
    return db + "-gold-results-" + model + "-" + mode + "-" + str(topk) + "-" + type_prediction + suf + ".json"

def compute_metrics(classifier, type_prediction, db, annotated_answers, true_answers, subset_k=None):
    matched_answers = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    n_gold_annotations = 0
    for query_answers in annotated_answers:
        ent = query_answers['query']['ent']
        rel = query_answers['query']['rel']
        if (ent, rel) in true_answers:
            true_annotated_answers = true_answers[(ent, rel)]
            assert (query_answers['valid_annotations'])
            assert (query_answers['annotator'] == classifier)
            for idx_ans, ans in enumerate(query_answers['annotated_answers']):
                if subset_k is not None and idx_ans >= subset_k:
                    break
                entity_id = ans['entity_id']
                checked = ans['checked']
                found = False
                for true_answer in true_annotated_answers:
                    if true_answer['entity_id'] == entity_id:
                        checked_by_annotators = true_answer['checked']
                        true_answer_checked = checked_by_annotators[0]['checked']  # Pick the first one
                        found = True
                        n_gold_annotations += 1
                        matched_answers += checked == true_answer_checked
                        if checked == True and true_answer_checked == True:
                            true_positives += 1
                        elif checked == True:
                            false_positives += 1
                        elif true_answer_checked == True:
                            false_negatives += 1
                        else:
                            true_negatives += 1
                        break
                if not found:
                    if entity_id == ent:
                        if checked:
                            false_positives += 1
                        else:
                            true_negatives += 1
                            matched_answers += 1
                        found = True
                if not found:
                    print("Answer {} for query {} {} not found!".format(entity_id, ent, rel))

    acc = matched_answers / n_gold_annotations
    if true_positives + false_negatives == 0:
        rec = 0
    else:
        rec = true_positives / (true_positives + false_negatives)
    if true_positives + false_positives == 0:
        prec = 0
    else:
        prec = true_positives / (true_positives + false_positives)
    if prec == 0 and rec == 0:
        f1 = 0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    print("Accuracy\t\t: {:.3f}".format(acc))
    print("Recall\t\t\t: {:.3f}".format(rec))
    print("Precision\t\t: {:.3f}".format(prec))
    print("F1\t\t\t\t: {:.3f}".format(f1))
    print("*********")
    results = {"F1": f1, "REC": rec, "PREC": prec, "dataset": db, "classifier": classifier,
               "type_prediction": type_prediction, "accuracy" : acc}
    return results


def load_classifier_annotations(classifiers, result_dir, dataset_name, embedding_model_name, mode, topk,
                                type_prediction, return_scores = False):
    classifiers_annotations = {}
    for idx, classifier in enumerate(classifiers):
        print("  Loading annotations from classifier {} ...".format(classifier))
        suf = '-' + classifier
        file_name = get_filename_answer_annotations(dataset_name, embedding_model_name, mode, topk,
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
                    if return_scores:
                        answer_set[answer['entity_id']] = [answer['score']]
                    else:
                        answer_set[answer['entity_id']] = [answer['checked']]
                else:
                    assert (idx != 0)
                    if return_scores:
                        answer_set[answer['entity_id']].append(answer['score'])
                    else:
                        answer_set[answer['entity_id']].append(answer['checked'])
    for key, ans in classifiers_annotations.items():
        assert (len(ans) == topk)
    return classifiers_annotations
