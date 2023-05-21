from tqdm import tqdm
import pickle
import copy
import numpy as np
import random
from enum import Enum
SUBTYPE = Enum('SUBTYPE', 'SPO POS OO OI IO II')
sub_type_to_string = {SUBTYPE.SPO: "spo", SUBTYPE.POS: "pos", SUBTYPE.OO: "oo", SUBTYPE.OI: "oi", SUBTYPE.IO: "io", SUBTYPE.II: "ii"}

def read_triples(filename):
    triples = []
    with open (filename, "r") as fin:
        lines = fin.readlines()
    for line in lines[1:]:
        h = int(line.split()[0])
        t = int(line.split()[1])
        r = int(line.split()[2])
        triples.append((h,t,r))

    return triples


def make_adjacency_lists(triples):
    adj_list_out = []
    adj_list_in = []

    for triple in triples:
        h = triple[0]
        t = triple[1]
        r = triple[2]
        while len(adj_list_out) < max(h, t) + 1:
            adj_list_out.append([])
        while len(adj_list_in) < max(h, t) + 1:
            adj_list_in.append([])
        adj_list_out[h].append((t, r))
        adj_list_in[t].append((h, r))

    return adj_list_out, adj_list_in


def load_pickle(filename):
    with open(filename, 'rb') as fin:
        data = pickle.load(fin)
    return data

def sample_star_subgraph_entities(subgraphs, db):
    id_to_entity_data = load_pickle('/var/scratch/dvs254/kbs/' + db + '-id-to-entity.pkl')
    id_to_relation_data = load_pickle('/var/scratch/dvs254/kbs/' + db + '-id-to-relation.pkl')
    subgraph_index = random.randint(0, len(subgraphs) - 1)
    subgraph = subgraphs[subgraph_index]
    size = subgraph.data['size']
    rid = subgraph.data['rel']
    center_id = subgraph.data['ent']
    entity_ids = []
    while len(entity_ids) < 10:
        eid = random.randint(0, size - 1)
        if eid not in entity_ids:
            entity_ids.append(eid)
    print('\nSubgraph ' + str(subgraph_index) + ':')
    print('Size: ' + str(size))
    print('Center: ' + str(id_to_entity_data[center_id]))
    print('Relation: ' + str(id_to_relation_data[rid]) + '\n')
    print("Entities:")
    for eid in entity_ids:
        print('  ' + str(id_to_entity_data[subgraph.data['entities'][eid]]))
    print()
    return


def sample_diamond_subgraph_entities(subgraphs, db):
    id_to_entity_data = load_pickle('/var/scratch/dvs254/kbs/' + db + '-id-to-entity.pkl')
    id_to_relation_data = load_pickle('/var/scratch/dvs254/kbs/' + db + '-id-to-relation.pkl')
    subgraph_index = random.randint(0, len(subgraphs) - 1)
    subgraph = subgraphs[subgraph_index]
    while subgraph.data['subType'] == SUBTYPE.SPO or subgraph.data['subType'] == SUBTYPE.POS:
        subgraph_index = random.randint(0, len(subgraphs) - 1)
        subgraph = subgraphs[subgraph_index]
    size = subgraph.data['size']
    rid1 = subgraph.data['rel1']
    rid2 = subgraph.data['rel2']
    center_id1 = subgraph.data['ent1']
    center_id2 = subgraph.data['ent2']
    entity_ids = []
    while len(entity_ids) < 10:
        eid = random.randint(0, size - 1)
        if eid not in entity_ids:
            entity_ids.append(eid)
    print('\nSubgraph ' + str(subgraph_index) + ':')
    print('Size: ' + str(size))
    print('Centers:')
    print('  ' + str(id_to_entity_data[center_id1]))
    print('  ' + str(id_to_entity_data[center_id2]))
    print('Relations:')
    print('  ' + str(id_to_relation_data[rid1]))
    print('  ' + str(id_to_relation_data[rid2]) + '\n')
    print("Entities:")
    for eid in entity_ids:
        print('  ' + str(id_to_entity_data[subgraph.data['entities'][eid]]))
    return

class Subgraph():
    def __init__(self, sid, st, sent, srel, ssize, entities):
        self.data = {}
        self.data['subType']  = st
        self.data['subId']    = sid
        self.data['ent']      = sent
        self.data['rel']      = srel
        self.data['size']     = ssize
        self.data['entities'] = copy.deepcopy(entities)

    def __str__():
        return str(self.data)

class SubgraphDiamond():
    def __init__(self, sid, st, sent1, sent2, srel1, srel2, ssize, entities):
        self.data = {}
        self.data['subType']  = st
        self.data['subId']    = sid
        self.data['ent1']     = sent1
        self.data['ent2']     = sent2
        self.data['rel1']     = srel1
        self.data['rel2']     = srel2
        self.data['size']     = ssize
        self.data['entities'] = copy.deepcopy(entities)

    def __str__():
        return str(self.data)

class SubgraphFactory():
    def __init__(self, db, min_subgraph_size, triples, ent_embeddings):
        self.db = db
        self.min_subgraph_size = min_subgraph_size
        self.E = ent_embeddings
        self.triples = triples

        self.subgraphs = []
        self.avg_embeddings = []
        self.var_embeddings = []

    def add_subgraphs(self, st, sent, srel, ssize, entities):
        subentities = copy.deepcopy(entities)
        sub = Subgraph(len(self.subgraphs), st, sent, srel, ssize, subentities)
        self.subgraphs.append(sub)

    def add_diamond_subgraphs(self, st, sent1, sent2, srel1, srel2, ssize, entities):
        subentities = copy.deepcopy(entities)
        sub = SubgraphDiamond(len(self.subgraphs), st, sent1, sent2, srel1, srel2, ssize, entities)
        self.subgraphs.append(sub)

    def get_Nsubgraphs(self):
        return len(self.subgraphs)

    def save(self, outdir, emb_model_str, subgraph_type_str, protocol=pickle.HIGHEST_PROTOCOL):
        filename = outdir + self.db + "-" + emb_model_str + "-" + subgraph_type_str + "-subgraphs-tau-" + str(self.min_subgraph_size) + ".pkl"
        print("writing to...", filename)
        with open(filename, 'wb') as fout:
            pickle.dump(self.subgraphs, fout, protocol=protocol)

        filename = outdir + self.db + "-" + emb_model_str + "-" + subgraph_type_str + "-avgemb-tau-" + str(self.min_subgraph_size) + ".pkl"
        with open(filename, 'wb') as fout:
            pickle.dump(self.avg_embeddings, fout, protocol=protocol)

        filename = outdir + self.db + "-" + emb_model_str + "-" + subgraph_type_str + "-varemb-tau-" + str(self.min_subgraph_size) + ".pkl"
        with open(filename, 'wb') as fout:
            pickle.dump(self.var_embeddings, fout, protocol=protocol)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fin:
            subgraphs = pickle.load(fin)
        return subgraphs

    def calculate_var_embeddings(self, count, mean, entities):
        E = self.E
        columnsSquareDiff = np.zeros(len(E[0]), dtype = np.float64)
        for entity in entities:
            columnsSquareDiff += (E[entity] - mean) * (E[entity] - mean)
        if count > 2:
            columnsSquareDiff /= (count - 1)
        else:
            columnsSquareDiff = mean
        return columnsSquareDiff

    # sub_type can be SPO or POS
    # TODO: Use this method to find first the SPO or POS subgraphs
    # then use entities in SPO subgraphs to further make SPOSPO, OPSPO etc.
    def make_subgraphs_per_type(self, sub_type):

        E = self.E
        min_subgraph_size = self.min_subgraph_size
        if sub_type == SUBTYPE.SPO:
            sorted_triples = sorted(self.triples, key = lambda l : (l[2], l[0]))
        elif sub_type == SUBTYPE.POS:
            sorted_triples = sorted(self.triples, key = lambda l : (l[2], l[1]))

        similar_entities = []
        current = np.zeros(len(E[0]), dtype = np.float64)
        count = 0
        prevo = -1
        prevp = -1
        cntTriples = len(sorted_triples)

        for i, triple in enumerate(sorted_triples):
            sub = triple[0]
            obj = triple[1]
            rel = triple[2]
            ent = -1
            other_ent = -1
            ER = None
            if sub_type == SUBTYPE.POS:
                ent = obj
                other_ent = sub
            else:
                ent = sub
                other_ent = obj

            if ent != prevo or rel != prevp:
                if count > min_subgraph_size:
                    mean = current/count
                    self.avg_embeddings.append(mean)
                    self.var_embeddings.append(self.calculate_var_embeddings(count, mean, similar_entities))
                    self.add_subgraphs(sub_type, prevo, prevp, count, similar_entities)
                count = 0
                prevo = ent
                prevp = rel
                current.fill(0.0)
                similar_entities.clear()
            count += 1
            current += E[other_ent]
            similar_entities.append(other_ent)
        # After looping over all triples, add remaining entities to a subgraph
        if count > min_subgraph_size:
            mean = current / count
            self.avg_embeddings.append(mean)
            self.var_embeddings.append(self.calculate_var_embeddings(count, mean, similar_entities))
            self.add_subgraphs(subType, prevo, prevp, count, similar_entities)

        print ("# of subgraphs ({}) : {}".format(sub_type_to_string[sub_type], self.get_Nsubgraphs()))

    def make_diamond_subgraphs(self, sub_type, adj_list_in, adj_list_out):
        E = self.E
        diamond_tuple_dicts = []    # change name
        valid_diamond_tuples = []
        if sub_type == SUBTYPE.OI or sub_type == SUBTYPE.II:
            adj_list = adj_list_out
        else:
            adj_list = adj_list_in
        for subgraph in self.subgraphs:
            if subgraph.data['subType'] != SUBTYPE.SPO and subgraph.data['subType'] != SUBTYPE.POS:
                break
            ent1 = subgraph.data['ent']
            rel1 = subgraph.data['rel']
            if (sub_type == SUBTYPE.OO or sub_type == SUBTYPE.OI) and subgraph.data['subType'] != SUBTYPE.SPO:
                continue
            if (sub_type == SUBTYPE.II or sub_type == SUBTYPE.IO) and subgraph.data['subType'] != SUBTYPE.POS:
                continue
            for entity in subgraph.data['entities']:
                for adj_entity in adj_list[entity]: # change depending on type
                    ent2 = adj_entity[0]
                    rel2 = adj_entity[1]
                    if ent1 == ent2 or rel1 >= rel2:
                        continue
                    while len(diamond_tuple_dicts) < ent2 + 1:
                        diamond_tuple_dicts.append({})
                    diamond_tuple_dict = diamond_tuple_dicts[ent2]
                    if (ent1, rel1, rel2) not in diamond_tuple_dict:
                        diamond_tuple_dict[(ent1, rel1, rel2)] = []
                    diamond_tuple_dict[(ent1, rel1, rel2)].append(entity)
                    if len(diamond_tuple_dict[(ent1, rel1, rel2)]) == self.min_subgraph_size + 1:
                        valid_diamond_tuples.append((ent1, ent2, rel1, rel2))

        ent2 = 0
        for tuple_dict in tqdm(diamond_tuple_dicts):
            for diamond_tuple in tuple_dict:
                count = len(tuple_dict[diamond_tuple])
                if count > self.min_subgraph_size:  # or >=
                    current = np.zeros(len(E[0]), dtype = np.float64)
                    similar_entities = []
                    for entity in tuple_dict[diamond_tuple]:
                        current += E[entity]  # change name of current?
                        similar_entities.append(entity)
                    mean = current / count
                    self.avg_embeddings.append(mean)
                    self.var_embeddings.append(self.calculate_var_embeddings(count, mean, similar_entities))
                    ent1 = diamond_tuple[0]
                    rel1 = diamond_tuple[1]
                    rel2 = diamond_tuple[2]
                    self.add_diamond_subgraphs(sub_type, ent1, ent2, rel1, rel2, count, similar_entities)
            ent2 += 1

        print ("# of subgraphs ({}) : {}".format(sub_type_to_string[sub_type], self.get_Nsubgraphs()))

        self.make_subgraphs_per_type(SUBTYPE.SPO)
        self.make_subgraphs_per_type(SUBTYPE.POS)
        #sample_star_subgraph_entities(self.subgraphs, self.db)

        if sub_type == "diamond":
            adj_list_out, adj_list_in = make_adjacency_lists(self.triples)
            self.make_diamond_subgraphs(SUBTYPE.OO, adj_list_in, adj_list_out)
            self.make_diamond_subgraphs(SUBTYPE.OI, adj_list_in, adj_list_out)
            self.make_diamond_subgraphs(SUBTYPE.IO, adj_list_in, adj_list_out)
            self.make_diamond_subgraphs(SUBTYPE.II, adj_list_in, adj_list_out)
            for i in range(len(self.subgraphs)):
                if self.subgraphs[i].data['subType'] != SUBTYPE.SPO and self.subgraphs[i].data['subType'] != SUBTYPE.POS:
                    first_dia_graph_index = i
                    break
            self.subgraphs = self.subgraphs[first_dia_graph_index:]
            self.avg_embeddings = self.avg_embeddings[first_dia_graph_index:]
            self.var_embeddings = self.var_embeddings[first_dia_graph_index:]
            #for i in range(8):
                #sample_diamond_subgraph_entities(self.subgraphs, self.db)
