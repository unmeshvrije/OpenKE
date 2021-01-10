from abc import abstractmethod

class Dataset:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def get_known_answers_for_hr(self, h, r):
        pass

    @abstractmethod
    def get_known_answers_for_tr(self, t, r):
        pass

    @abstractmethod
    def exists_htr(self, h, t, r):
        pass

    @abstractmethod
    def get_neighbours(self, e) -> set():
        pass