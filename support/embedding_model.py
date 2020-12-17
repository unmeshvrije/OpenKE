from kge.model import KgeModel
from kge.util.io import load_checkpoint
import torch
from .utils import *

class Embedding_Model:
    def __init__(self, results_dir, typ : {'transe', 'complex', 'rotate'}, dataset):
        self.typ = typ
        self.dataset = dataset
        # Load the model
        db = dataset.get_name()
        path = results_dir + '/' + db + "/embeddings/" + get_filename_model(db, typ)
        checkpoint = load_checkpoint(path)
        self.model = KgeModel.create_from(checkpoint)
        self.E = self.model._entity_embedder._embeddings_all().data
        self.R = self.model._relation_embedder._embeddings_all().data
        self.n = len(self.E)
        self.r = len(self.R)

    def get_type(self):
        return self.typ

    def get_dataset_name(self):
        return self.dataset.get_name()

    def num_entities(self):
        return self.model.num_entities()

    def num_relations(self):
        return self.model.num_relations()

    def get_embedding_entity(self, entity_id):
        return self.E[entity_id].numpy()

    def get_embedding_relation(self, relation_id):
        return self.R[relation_id].numpy()

    def get_size_embedding_entity(self):
        return len(self.E[0])

    def get_size_embedding_relation(self):
        return len(self.R[0])