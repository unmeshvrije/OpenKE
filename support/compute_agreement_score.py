#cat /home/jurbani/data2/binary-embeddings/fb15k237/annotations/gold-annotations.json-2021-01-26\ 23\:30\:02.286156  | python -c "import sys, json; print(json.load(sys.stdin)['757']['annotator'])"
import os
import sys
import json
#dir_path = "/home/jurbani/data2/binary-embeddings/fb15k237/annotations/"
dir_path = "/home/jurbani/data2/binary-embeddings/dbpedia50/annotations/"

json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
jacopo_query_ids = set()
unmesh_query_ids = set()

for fil in json_files:
    print(os.path.join(dir_path,fil))
    with open(os.path.join(dir_path, fil), encoding='utf-8') as fin:
        objects = json.load(fin)
        for key in objects.keys():
            if objects[key]['annotator'] == 'J':
                #jacopo_query_ids.add(objects[key]['query']['id'])
                jacopo_query_ids.add((objects[key]['query']['ent'], objects[key]['query']['rel']))
            elif objects[key]['annotator'] == 'U':
                #unmesh_query_ids.add(objects[key]['query']['id'])
                unmesh_query_ids.add((objects[key]['query']['ent'], objects[key]['query']['rel']))

    print(jacopo_query_ids.intersection(unmesh_query_ids))
    print(sorted(list(unmesh_query_ids)))
    print(sorted(list(jacopo_query_ids)))
