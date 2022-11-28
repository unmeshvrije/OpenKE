ent_dict = {}
rel_dict = {}
n_triples = 0
n_training = 0
n_valid = 0
with open("myago.tsv", "r") as fin:
    lines = fin.readlines()
    n_triples = len(lines)
    n_training = int(0.8 * n_triples)
    n_valid    = int(0.9 * n_triples)
    train_data = ""
    valid_data = ""
    test_data  = ""
    ent_dict_data = ""
    rel_dict_data = ""
    ent_id = 0
    rel_id = 0
    for i, line in enumerate(lines):
        head_str = line.split()[0]
        rel_str  = line.split()[1]
        tail_str = line.split()[2]
        if head_str not in ent_dict:
            ent_dict[head_str] = ent_id
            ent_dict_data += head_str + " " + str(ent_id) + "\n"
            ent_id += 1
        if tail_str not in ent_dict:
            ent_dict[tail_str] = ent_id
            ent_dict_data += tail_str + " " + str(ent_id) + "\n"
            ent_id += 1
        if rel_str not in rel_dict:
            rel_dict[rel_str] = rel_id
            rel_dict_data += rel_str + " " + str(rel_id) + "\n"
            rel_id += 1
        if i < n_training:
            train_data += str(ent_dict[head_str]) + " " + str(ent_dict[tail_str]) + " " + str(rel_dict[rel_str]) + "\n"
        elif i < n_valid:
            valid_data += str(ent_dict[head_str]) + " " + str(ent_dict[tail_str]) + " " + str(rel_dict[rel_str]) + "\n"
        else:
            test_data += str(ent_dict[head_str]) + " " + str(ent_dict[tail_str]) + " " + str(rel_dict[rel_str]) + "\n"
# make files

n_entities = len(ent_dict.keys())
with open("entity2id.txt", "w") as fout:
    fout.write(str(n_entities) + "\n" + ent_dict_data)

n_relations = len(rel_dict.keys())
with open("relation2id.txt", "w") as fout:
    fout.write(str(n_relations) + "\n" + rel_dict_data)

with open("train2id.txt", "w") as fout:
    fout.write(str(n_training) + "\n" + train_data)

valid_count = n_valid - n_training
with open("valid2id.txt", "w") as fout:
    fout.write(str(valid_count) + "\n" + valid_data)

test_count = n_triples - n_valid
with open("test2id.txt", "w") as fout:
    fout.write(str(test_count) + "\n" + test_data)
