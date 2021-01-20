import supervised_classifier
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from support.embedding_model import Embedding_Model

class Conv_Dataset(Dataset):
    def __init__(self, dataset):
        dim1 = len(dataset)
        dim2 = len(dataset[0]['X'])
        dim3 = len(dataset[0]['X'][0])
        self.X = np.zeros(shape=(dim1, dim2, dim3), dtype=np.float32)
        self.Y = np.zeros(shape=(dim1, dim2), dtype=np.float32)
        for idx, a in enumerate(dataset):
            self.X[idx] = a['X']
            self.Y[idx] = a['Y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

class Conv_Model(nn.Module):
    def __init__(self, topk, kernel_size1, kernel_size2):
        super(Conv_Model, self).__init__()
        self.topk = topk
        self.conv1 = torch.nn.Conv2d(1, topk, kernel_size=kernel_size1)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size2, stride=2, padding=0)
        self.fc1 = None
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        out = self.conv1(x)
        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        if self.fc1 is None:
            self.fc1 = torch.nn.Linear(out.shape[1], self.topk)
        out = self.fc1(out)
        out = self.sig(out)
        return out

class Classifier_Conv(supervised_classifier.Supervised_Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 results_dir,
                 embedding_model : Embedding_Model,
                 topk,
                 hyper_params = None,
                 model_path = None):
        if hyper_params is None:
            hyper_params = { "kernel_size1" : 4, "kernel_size2" : 2, "topk" : topk }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(Classifier_Conv, self).__init__(dataset, type_prediction, results_dir,
                                             embedding_model, hyper_params, model_path)
        if model_path is not None:
            self.get_model().eval()

    def init_model(self, embedding_model, hyper_params):
        kernel_size1 = hyper_params['kernel_size1']
        kernel_size2 = hyper_params['kernel_size2']
        topk = hyper_params['topk']
        self.n_features = embedding_model.get_size_embedding_entity() * 2 + embedding_model.get_size_embedding_relation()
        self.set_model(Conv_Model(topk, kernel_size1, kernel_size2).to(self.device))

    def get_name(self):
        return "ConvNN"

    def create_training_data(self, queries_with_annotated_answers):
        out = []
        for query in tqdm(queries_with_annotated_answers):
            assert(query['query']['type'] == 1 or self.type_prediction == 'head')
            assert (query['query']['type'] == 0 or self.type_prediction == 'tail')
            ent = query['query']['ent']
            rel = query['query']['rel']
            emb_e = self.embedding_model.get_embedding_entity(ent)
            emb_r = self.embedding_model.get_embedding_relation(rel)
            answers = query['annotated_answers']
            X = np.zeros(shape=(len(answers), self.n_features), dtype=np.float)
            Y = np.zeros(shape=(len(answers)), dtype=np.int)
            for i, answer in enumerate(answers):
                a = answer['entity_id']
                emb_a = self.embedding_model.get_embedding_entity(a)
                #X
                if self.type_prediction == 'head':
                    X[i] = np.concatenate([emb_a, emb_r, emb_e])
                else:
                    X[i] = np.concatenate([emb_e, emb_r, emb_a])
                #Y
                if answer['checked']:
                    Y[i] = 1
                else:
                    Y[i] = 0
            data_entry = {}
            data_entry['X'] = X
            data_entry['Y'] = Y
            out.append(data_entry)
        return out

    def train(self, training_data, valid_data, model_path, batch_size=100, epochs=10):
        # Load input data
        self.get_model().train()
        training_data_set = Conv_Dataset(training_data)
        train_data_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.get_model().parameters())
        best_acc = 0
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Start epoch {}".format(epoch))
            running_loss = 0.0
            for i, data in enumerate(train_data_loader, 0):
                inputs, labels = data
                inputs.to(self.device)
                labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.get_model()(inputs)
                outputs_reshaped = outputs.reshape(outputs.shape[0], outputs.shape[1])
                loss = criterion(outputs_reshaped, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            if valid_data is not None:
                test_acc = self.validate(valid_data)
                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
                if is_best and model_path is not None:
                    self.save_model(model_path, epoch)
        # Save model
        if valid_data is None and model_path is not None:
            self.save_model(model_path, epoch)

    def predict(self, query_with_answers, type_answers):
        ent = query_with_answers['ent']
        rel = query_with_answers['rel']
        typ = query_with_answers['type']
        assert (typ == 1 or self.type_prediction == 'head')
        assert (typ == 0 or self.type_prediction == 'tail')
        emb_e = self.embedding_model.get_embedding_entity(ent)
        emb_r = self.embedding_model.get_embedding_relation(rel)
        n_features = len(emb_e) * 2 + len(emb_r)
        answers = query_with_answers[type_answers]
        X = np.zeros(shape=(len(answers), n_features), dtype=np.float)
        annotated_answers = []
        for i, a in enumerate(answers):
            answer = a['entity_id']
            emb_a = self.embedding_model.get_embedding_entity(answer)
            # X
            if self.type_prediction == 'head':
                X[i] = np.concatenate([emb_a, emb_r, emb_e])
            else:
                X[i] = np.concatenate([emb_e, emb_r, emb_a])
        # Do the prediction
        XS = torch.Tensor(X)
        XS = XS.view(1, X.shape[0], X.shape[1])
        out = self.get_model()(XS)
        out = out.view(X.shape[0])
        for i, answer in enumerate(answers):
            score = out[i].item()
            checked = score > 0.5
            annotated_answers.append({'entity_id': answer['entity_id'], 'checked': checked, 'score': score})
        return annotated_answers