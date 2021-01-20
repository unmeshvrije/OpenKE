from abc import ABC, abstractmethod

class Classifier(ABC):

    def __init__(self, dataset, type_prediction : {'head', 'tail'}, results_dir):
        self.dataset = dataset
        self.type_prediction = type_prediction
        self.result_dir = results_dir

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def predict(self, query_with_answers, type_answers):
        """
        The goal of this method is to label some answers. Given a query and a potential answer (completion),
        this method will assigns to it either a 1, -1, or 0 to indicate that the answer is true, false, or unknown,
        respectively.

        :param answers: These are a set of queries with related answers. They are produced by create_answers.py
        :return: The answers in input are labeled with three values: 1 (true), -1 (false), 0 (abstain)
        """
        pass

    def start_predict(self):
        pass