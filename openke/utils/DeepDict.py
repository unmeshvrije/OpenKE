from collections import defaultdict

class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)
