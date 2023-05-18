import sys
import pickle5 as pl
import argparse
import numpy as np
import orjson


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read training/test file and run LSTM training or test."
    )
    parser.add_argument(
        "--answerfile",
        dest="annotated_answers_file",
        type=str,
        help="File containing test queries.",
        default="/var/scratch2/uji300/OpenKE-results/dbpedia50/annotations/dbpedia50-annotated-answers-transe-test-10-head-snorkel.pkl",
    )
    parser.add_argument(
        "--embfile",
        dest="emb_file",
        type=str,
        help="File containing entity embeddings.",
    )
    parser.add_argument(
        "--entdict",
        dest="ent_dict",
        type=str,
        default="/var/scratch2/uji300/OpenKE-results/dbpedia50/misc/dbpedia50-id-to-entity.pkl",
        help="entity id dictionary.",
    )
    parser.add_argument(
        "--reldict",
        dest="rel_dict",
        type=str,
        default="/var/scratch2/uji300/OpenKE-results/dbpedia50/misc/dbpedia50-id-to-relation.pkl",
        help="relation id dictionary.",
    )
    parser.add_argument("--topk", dest="topk", type=int, default=10)
    parser.add_argument("--db", dest="db", type=str, default="dbpedia50")
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default="transe",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--pred",
        dest="pred",
        type=str,
        choices=["head", "tail"],
        help="Prediction type (head/tail)",
    )
    parser.add_argument(
        "--outfile",
        dest="out_file",
        type=str,
        default="./out_temp.json"
    )
    return parser.parse_args()


args = parse_args()


def load_pickle(file_name):
    with open(file_name, "rb") as fin:
        pkl = pl.load(fin)
    return pkl


entity_map = load_pickle(args.ent_dict)
relation_map = load_pickle(args.rel_dict)
data = load_pickle(args.annotated_answers_file)


def get_query_str(q: dict) -> str:
    """
    Prints query as a string
    q is a dictionary of following keys
    type: 0 means head prediction query, 1 means tail prediction query
    ent: id of the entity
    rel: id of the relation
    ...
    """
    return (
        f"?, {relation_map[q['rel']]}, {entity_map[q['ent']]}"
        if not q["type"]
        else f"{entity_map[q['ent']]}, {relation_map[q['rel']]}, ?"
    )

queries = []
for d in data:
    # Print all answers that Snorkel thinks are true
    '''
    if any(x['checked'] for x in d['annotated_answers']):
        print(get_query_str(d['query']))
        answers = [entity_map[x['entity_id']] for x in d['annotated_answers'] if x['checked']]
        print(answers)
        print("*" * 80)
    '''
    # Print all answers that are actually true
    if any(x['entity_id'] in d['query']['answers_test_file'] for x in d['annotated_answers']):
        answers = [entity_map[x['entity_id']] for x in d['annotated_answers'] if x['checked'] and x['entity_id'] in d['query']['answers_test_file']]
        if answers:
            query = dict(
                query=get_query_str(d['query']),
                answers=answers
            )
            queries.append(query)

with open(args.out_file, 'wb') as fout:
    fout.write(orjson.dumps(queries))
