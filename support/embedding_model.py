import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import numpy as np
from support.dataset import Dataset
from .utils import *
from openke.module.model import RotatE
from openke.data import TrainDataLoader

from enum import Enum
SubgraphType = Enum('SubgraphType', 'SPO POS')
from collections import namedtuple
Subgraph = namedtuple("Subgraph", "type ent rel")

class Embedding_Model:
    def __init__(self, results_dir, typ : {'transe', 'complex', 'rotate'}, dataset):
        self.typ = typ
        self.dataset = dataset
        # Load the model
        db = dataset.get_name()
        if typ == 'rotate':
            suf = '.ckpt'
            path = results_dir + '/' + db + "/embeddings/" + get_filename_model(db, typ, suf)
            model = torch.load(path, map_location=torch.device('cpu'))
            self.E = model['ent_embeddings.weight']
            self.R = model['rel_embeddings.weight']
            db_path = dataset.get_path()
            self.train_dataloader = TrainDataLoader(
                in_path=db_path,
                nbatches=100,
                threads=8,
                sampling_mode="normal",
                bern_flag=1,
                filter_flag=1,
                neg_ent=25,
                neg_rel=0
            )
            N_DIM = 200
            self.model = RotatE(
                ent_tot=self.train_dataloader.get_ent_tot(),
                rel_tot=self.train_dataloader.get_rel_tot(),
                dim=N_DIM,
                margin=6.0,
                epsilon=2.0)
        else:
            suf = '.pt'
            path = results_dir + '/' + db + "/embeddings/" + get_filename_model(db, typ, suf)
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
        return self.n #self.model.num_entities()

    def num_relations(self):
        return self.r #self.model.num_relations()

    def get_embedding_entity(self, entity_id):
        return self.E[entity_id].numpy()

    def get_embedding_relation(self, relation_id):
        return self.R[relation_id].numpy()

    def get_size_embedding_entity(self):
        return len(self.E[0])

    def get_size_embedding_relation(self):
        return len(self.R[0])

    def _make_subgraph_embeddings_per_type(self, dataset : Dataset, sub_type, tau = 10):
        if sub_type == SubgraphType.SPO:
            subgraphs = dataset.get_hr_subgraphs()
        else:
            subgraphs = dataset.get_tr_subgraphs()
        for query, answers in subgraphs.items():
            if len(answers) >= tau:
                ent, rel = query
                current = np.zeros(self.get_size_embedding_entity(), dtype=np.float64)
                for answer in answers:
                    current += self.get_embedding_entity(answer)
                avg = current / len(answers)
                self.avg_subgraphs.append(avg)
                self.subgraphs.append(Subgraph(type=sub_type, ent=ent, rel=rel))

    def make_subgraph_embeddings(self, dataset : Dataset, tau = 10):
        self.avg_subgraphs = []
        self.subgraphs = []
        self._make_subgraph_embeddings_per_type(dataset, SubgraphType.SPO, tau)
        self._make_subgraph_embeddings_per_type(dataset, SubgraphType.POS, tau)
        self.avg_subgraphs = np.asarray(self.avg_subgraphs)

    def get_most_similar_subgraphs(self, sub_type, ent, rel, k=10):
        # Create a model with the subgraphs
        if self.typ != 'rotate':
            scorer = self.model.get_scorer()
            e = self.E[ent].view(1, -1)
            r = self.R[rel].view(1, -1)
            T = torch.Tensor(self.avg_subgraphs)
            if sub_type == SubgraphType.SPO:
                combine = 'sp_'
                scores = scorer.score_emb(e, r, T, combine)
            else:
                combine = '_po'
                scores = scorer.score_emb(T, r, e, combine)
            o = torch.argsort(scores, dim=-1, descending=True)
            best_subgraphs = o[0][:k]
        else:
            new_E = torch.Tensor(self.E[ent])[np.newaxis, :]
            new_R = torch.Tensor(self.R[rel])[np.newaxis, :]
            new_S = torch.Tensor(self.avg_subgraphs)
            if sub_type == SubgraphType.POS:
                type_prediction = 'head'
            else:
                type_prediction = 'tail'
            subgraph_scores = self.model._calc(new_S, new_E, new_R, type_prediction + '_batch')
            o = np.argsort(subgraph_scores)
            best_subgraphs = o[:k]

        out = []
        for b in best_subgraphs:
            out.append(self.subgraphs[b])
        return out