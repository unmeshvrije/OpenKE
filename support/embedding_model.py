import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import numpy as np
from support.dataset import Dataset
from .utils import *
from openke.module.model import RotatE, ComplEx, TransE
from openke.data import TrainDataLoader, TestDataLoader
import os

from enum import Enum
SubgraphType = Enum('SubgraphType', 'SPO POS')
from collections import namedtuple
Subgraph = namedtuple("Subgraph", "type ent rel")

class Embedding_Model:

    # Sometimes the embedding model (libkge) uses a different dictionary than the dataset. In this case, I change the embeddings so that the correct ones are used
    def _fix_dictionary(self):
        self.ent_map = None
        self.rel_map = None
        if self.use_libkge:
            self.E = self.model._entity_embedder._embeddings_all().data
            self.R = self.model._relation_embedder._embeddings_all().data
            if self.n != self.model.dataset.num_entities() or self.r != self.model.dataset.num_relations():
                # Build the mapping from our entities to KGE entities
                self.ent_map = np.zeros(self.n, dtype=np.int64)
                kge_map = {}
                for i in range(self.model.dataset.num_entities()):
                    txt = self.model.dataset.entity_strings(i)
                    kge_map[txt] = i
                for i in range(self.n):
                    # Get text:
                    txt = self.dataset.get_entity_text(i)
                    if txt in kge_map:
                        kge_id = kge_map[txt]
                        self.ent_map[i] = kge_id
                    else:
                        print("Not found!")
                #new_E = self.E[ent_map]

                self.rel_map = np.zeros(self.r, dtype=np.int64)
                kge_map = {}
                for i in range(self.model.dataset.num_relations()):
                    txt = self.model.dataset.relation_strings(i)
                    kge_map[txt] = i
                for i in range(self.r):
                    # Get text:
                    txt = self.dataset.get_relation_text(i)
                    assert (txt in kge_map)
                    kge_id = kge_map[txt]
                    self.rel_map[i] = kge_id
                #new_R = self.R[rel_map]
                #self.E = new_E
                #self.R = new_R
        else:
            self.E = self.torch_model['ent_embeddings.weight']
            self.R = self.torch_model['rel_embeddings.weight']


    def __init__(self, results_dir, typ : {'transe', 'complex', 'rotate'}, dataset):
        self.typ = typ
        self.dataset = dataset
        # Load the model
        db = dataset.get_name()
        self.n = dataset.get_n_entities()
        self.r = dataset.get_n_relations()
        self.use_libkge = False

        suf = '.pt'
        path = results_dir + '/' + db + "/embeddings/" + get_filename_model(db, typ, suf)
        if os.path.exists(path):
            checkpoint = load_checkpoint(path)
            self.model = KgeModel.create_from(checkpoint)
            #self.E = self.model._entity_embedder._embeddings_all().data
            #self.R = self.model._relation_embedder._embeddings_all().data
            self.dim_e = self.model.get_s_embedder().dim
            assert(self.model.get_s_embedder().dim == self.model.get_o_embedder().dim)
            self.dim_r = self.model.get_p_embedder().dim
            self.use_libkge = True
        else:
            suf = '.ckpt'
            path = results_dir + '/' + db + "/embeddings/" + get_filename_model(db, typ, suf)
            if os.path.exists(path):
                self.torch_model = torch.load(path, map_location=torch.device('cpu'))
                #self.E = self.model['ent_embeddings.weight']
                #self.R = self.model['rel_embeddings.weight']
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
                self.test_dataloader = TestDataLoader(in_path=db_path)
                assert(self.n == self.train_dataloader.get_ent_tot())
                assert(self.r == self.train_dataloader.get_rel_tot())

                if typ == 'rotate':
                    self.model = RotatE(
                        ent_tot=self.train_dataloader.get_ent_tot(),
                        rel_tot=self.train_dataloader.get_rel_tot(),
                        dim=200,
                        margin=6.0,
                        epsilon=2.0)
                elif typ == 'complex':
                    self.model = ComplEx(
                        ent_tot=self.train_dataloader.get_ent_tot(),
                        rel_tot=self.train_dataloader.get_rel_tot(),
                        dim=256
                    )
                else: #transe
                    self.model = TransE(
                        ent_tot=self.train_dataloader.get_ent_tot(),
                        rel_tot=self.train_dataloader.get_rel_tot(),
                        dim=200,
                        p_norm=1,
                        norm_flag=True
                    )
                self.dim_e = len(self.torch_model['ent_embeddings.weight'][0])
                self.dim_r = len(self.torch_model['rel_embeddings.weight'][0])
            else:
                self.model = None
        assert(self.model is not None)
        self._fix_dictionary() # This method will create the E and R data structures

    def get_type(self):
        return self.typ

    def get_dataset_name(self):
        return self.dataset.get_name()

    def num_entities(self):
        return self.n

    def num_relations(self):
        return self.r

    def get_embedding_entity(self, entity_id):
        if self.ent_map is not None:
            entity_id = self.ent_map[entity_id]
        return self.E[entity_id].numpy()

    def get_embedding_relation(self, relation_id):
        if self.rel_map is not None:
            relation_id = self.rel_map[relation_id]
        return self.R[relation_id].numpy()

    def get_size_embedding_entity(self):
        return self.dim_e

    def get_size_embedding_relation(self):
        return self.dim_r

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
        if self.use_libkge:
            scorer = self.model.get_scorer()
            e = torch.from_numpy(self.get_embedding_entity(ent)).view(1, -1)
            r = torch.from_numpy(self.get_embedding_relation(rel)).view(1, -1)
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
            new_E = torch.Tensor(self.get_embedding_entity(ent))[np.newaxis, :]
            new_R = torch.Tensor(self.get_embedding_relation(rel))[np.newaxis, :]
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

    def score_sp(self, ent, rel):
        if self.use_libkge:
            if self.ent_map is not None:
                ent = self.ent_map[ent]
                ent = torch.Tensor(ent)
            if self.rel_map is not None:
                rel = self.rel_map[rel]
                rel = torch.Tensor(rel)
            scores = self.model.score_sp(ent, rel)
            if self.ent_map is not None:
                scores = torch.index_select(scores, 1, torch.from_numpy(self.ent_map))
            return scores
        else:
            scores = np.zeros((len(rel), self.n), dtype=float)
            for idx, r in enumerate(tqdm(rel)):
                e = ent[idx]
                T = self.E
                R = torch.from_numpy(self.get_embedding_relation(r))[np.newaxis, :]
                H = torch.from_numpy(self.get_embedding_entity(e))[np.newaxis, :]
                mode = 'tail_batch'
                s = self.model._calc(H, T, R, mode)
                s = -s
                scores[idx] = s.numpy()
            return torch.from_numpy(scores)

    def score_po(self, rel, ent):
        if self.use_libkge:
            if self.ent_map is not None:
                ent = self.ent_map[ent]
                ent = torch.Tensor(ent)
            if self.rel_map is not None:
                rel = self.rel_map[rel]
                rel = torch.Tensor(rel)
            scores = self.model.score_po(rel, ent)
            if self.ent_map is not None:
                scores = torch.index_select(scores, 1, torch.from_numpy(self.ent_map))
            return scores
        else:
            scores = np.zeros((len(rel), self.n), dtype=float)
            for idx, r in enumerate(tqdm(rel)):
                e = ent[idx]
                H = self.E
                R = torch.from_numpy(self.get_embedding_relation(r))[np.newaxis, :]
                T = torch.from_numpy(self.get_embedding_entity(e))[np.newaxis, :]
                mode = 'head_batch'
                s = self.model._calc(H, T, R, mode)
                s = -s
                scores[idx] = s.numpy()
            return torch.from_numpy(scores)
