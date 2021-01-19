from abc import abstractmethod

class Dataset:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_path(self):
        return "./benchmarks/" + self.name + '/'

    @abstractmethod
    def get_known_answers_for_hr(self, h, r):
        pass

    @abstractmethod
    def get_known_answers_for_tr(self, t, r):
        pass

    @abstractmethod
    def get_hr_subgraphs(self):
        pass

    @abstractmethod
    def get_tr_subgraphs(self):
        pass

    @abstractmethod
    def exists_htr(self, h, t, r):
        pass

    @abstractmethod
    def get_neighbours(self, e) -> set():
        pass

    @abstractmethod
    def get_facts(self):
        pass

    @abstractmethod
    def get_n_entities(self):
        pass

    @abstractmethod
    def get_n_relations(self):
        pass

    @abstractmethod
    def get_entity_text(self, id):
        pass

    @abstractmethod
    def get_relation_text(self, id):
        pass