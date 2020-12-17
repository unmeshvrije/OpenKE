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

    def __init__(self):
        super(Dataset_FB15k237, self).__init__("fb15k237")
        path = 'benchmarks/fb15k237'
        training_data_path = path + '/train2id.txt'
        training_data = self._load_dataset(training_data_path)
        self.known_answers_hr = {}
        self.known_answers_tr = {}
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

        # Load data
        #valid_data_path = path + '/valid2id.txt'
        #test_data_path = path + '/test2id.txt'
        #valid_data = self._load_dataset(valid_data_path)
        #self.test_data = self._load_dataset(test_data_path)

    def get_known_answers_for_hr(self, h, r):
        q = (h, r)
        if q in self.known_answers_hr:
            return self.known_answers_hr[q]
        else:
            return []

    def get_known_answers_for_tl(self, t, r):
        q = (t, r)
        if q in self.known_answers_tr:
            return self.known_answers_tr[q]
        else:
            return []

    def exists_htr(self, h, t, r):
        return (h,t,r) in self.facts