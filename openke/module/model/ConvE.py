import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import math

class ConvE(Model):

    def __init__(self, ent_tot, rel_tot, dim = 100, input_dropout = 0.2, hidden_dropout = 0.3, feature_dropout = 0.2):
        super(ConvE, self).__init__(ent_tot, rel_tot)
        
        self.dim = dim
        hidden_dim = 32
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hid_drop = torch.nn.Dropout(hidden_dropout)
        self.feat_drop = torch.nn.Dropout2d(feature_dropout)
        self.conv = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.dim)

        self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.ent_tot)))
        self.dim_w = math.ceil(self.dim * .2)
        self.dim_h = self.dim // self.dim_w
        self.fc = torch.nn.Linear(self.dim_w * 2 * self.dim_h * hidden_dim, self.dim)

    def _calc(self, h, t, r, mode):
        r_reshaped = r.view(-1, 1, self.dim_w, self.dim_h)
        if mode == "tail_batch":
            h_reshaped = h.view(-1, 1, self.dim_w, self.dim_h)
            stacked_inputs = torch.cat([h_reshaped, r_reshaped], 2)
        else:
            t_reshaped = t.view(-1, 1, self.dim_w, self.dim_h)
            stacked_inputs = torch.cat([t_reshaped, r_reshaped], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.feat_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.mm(x, self.emb_ent.weight.transpose(1,0))
        x += self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x

    def _vector_op(self, vector, r, mode):
        implemented = False  # TODO: implement

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r, mode)
        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()