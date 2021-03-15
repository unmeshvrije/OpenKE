from .dataset import Dataset

class Dataset_FB15k237(Dataset):
    def _load_dataset(self, path_file):
        facts = []
        with open(path_file, 'rt') as fin:
            nfacts = int(fin.readline())
            for l in fin:
                tkns = l.split(' ')
                h = int(tkns[0])
                t = int(tkns[1])
                r = int(tkns[2])
                facts.append((h,t,r))
        return facts

    def __init__(self, include_valid=True):
        super(Dataset_FB15k237, self).__init__("fb15k237")
        path = 'benchmarks/fb15k237'
        training_data_path = path + '/train2id.txt'
        training_data = self._load_dataset(training_data_path)
        valid_data_path = path + '/valid2id.txt'
        valid_data = self._load_dataset(valid_data_path)
        test_data_path = path + '/test2id.txt'
        test_data = self._load_dataset(test_data_path)
        # add valid data to the set of training data
        if include_valid:
            for v in valid_data:
                training_data.append(v)
        self.known_answers_hr = {}
        self.known_answers_tr = {}
        self.neighbours = {}
        self.facts = set(training_data)
        for t in training_data:
            q_hr = (t[0], t[2])
            if q_hr in self.known_answers_hr:
                self.known_answers_hr[q_hr].append(t[1])
            else:
                self.known_answers_hr[q_hr] = [ t[1] ]
            q_tr = (t[1], t[2])
            if q_tr in self.known_answers_tr:
                self.known_answers_tr[q_tr].append(t[0])
            else:
                self.known_answers_tr[q_tr] = [t[0]]
            if t[0] not in self.neighbours:
                self.neighbours[t[0]] = set()
            self.neighbours[t[0]].add(t[1])
            if t[1] not in self.neighbours:
                self.neighbours[t[1]] = set()
            self.neighbours[t[1]].add(t[0])

        self.test_answers_hr = {}
        self.test_answers_tr = {}
        for t in test_data:
            q_hr = (t[0], t[2])
            if q_hr in self.test_answers_hr:
                self.test_answers_hr[q_hr].append(t[1])
            else:
                self.test_answers_hr[q_hr] = [t[1]]
            q_tr = (t[1], t[2])
            if q_tr in self.test_answers_tr:
                self.test_answers_tr[q_tr].append(t[0])
            else:
                self.test_answers_tr[q_tr] = [t[0]]

        entity_dict_path = path + '/entity2id.txt'
        with open(entity_dict_path, 'rt') as fin:
            self.n_entities = int(fin.readline())
        relation_dict_path = path + '/relation2id.txt'
        with open(relation_dict_path, 'rt') as fin:
            self.n_relations = int(fin.readline())

        # test_data_path = path + '/test2id.txt'
        # self.test_data = self._load_dataset(test_data_path)
        entity_dict_path = path + '/entity2id.txt'
        self.ent_id2txt = {}
        self.ent_txt2id = {}
        with open(entity_dict_path, 'rt') as fin:
            self.n_entities = int(fin.readline())
            for l in fin:
                tkns = l.split('\t')
                id = int(tkns[1])
                txt = tkns[0]
                self.ent_id2txt[id] = txt
                self.ent_txt2id[txt] = id

        relation_dict_path = path + '/relation2id.txt'
        self.rel_id2txt = {}
        self.rel_txt2id = {}
        with open(relation_dict_path, 'rt') as fin:
            self.n_relations = int(fin.readline())
            for l in fin:
                tkns = l.split('\t')
                id = int(tkns[1])
                txt = tkns[0]
                self.rel_id2txt[id] = txt
                self.rel_txt2id[txt] = id

    def get_hr_subgraphs(self):
        return self.known_answers_hr

    def get_tr_subgraphs(self):
        return self.known_answers_tr

    def get_known_answers_for_hr(self, h, r):
        q = (h, r)
        if q in self.known_answers_hr:
            return self.known_answers_hr[q]
        else:
            return []

    def get_known_answers_for_tr(self, t, r):
        q = (t, r)
        if q in self.known_answers_tr:
            return self.known_answers_tr[q]
        else:
            return []

    def get_test_answers_for_hr(self, h, r):
        q = (h, r)
        if q in self.test_answers_hr:
            return self.test_answers_hr[q]
        else:
            return []

    def get_test_answers_for_tr(self, t, r):
        q = (t, r)
        if q in self.test_answers_tr:
            return self.test_answers_tr[q]
        else:
            return []

    def exists_htr(self, h, t, r):
        return (h,t,r) in self.facts

    def get_neighbours(self, e):
        if e not in self.neighbours:
            return set()
        else:
            return self.neighbours[e]

    def get_facts(self):
        return self.facts

    def get_n_entities(self):
        return self.n_entities

    def get_n_relations(self):
        return self.n_relations

    def get_entity_text(self, id):
        return self.ent_id2txt[id]

    def get_relation_text(self, id):
        return self.rel_id2txt[id]

    def get_relation_id(self, txt):
        return self.rel_txt2id[txt]

    def get_entity_id(self, txt):
        return self.ent_txt2id[txt]