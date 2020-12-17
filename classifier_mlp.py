import supervised_classifier
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout

from support.embedding_model import Embedding_Model

class MLP_Dataset(Dataset):
    def __init__(self, dataset):
        dim1 = len(dataset)
        dim2 = len(dataset[0]['X'])
        self.X = np.zeros(shape=(dim1, dim2), dtype=np.float32)
        self.Y = np.zeros(shape=(dim1, 1), dtype=np.float32)
        for idx, a in enumerate(dataset):
            self.X[idx] = a['X']
            self.Y[idx][0] = a['Y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

class MLP_model(nn.Module):
    def __init__(self, n_features, n_hidden_units, dropout):
        super(MLP_model, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden_units),
            nn.Dropout(0.2),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.Dropout(0.2),
            nn.Linear(n_hidden_units, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.mlp(x)
        return out

class Classifier_MLP(supervised_classifier.Supervised_Classifier):
    def __init__(self,
                 dataset,
                 type_prediction : {'head', 'tail'},
                 results_dir,
                 embedding_model : Embedding_Model,
                 n_units = 100,
                 dropout = 0.2):
        super(Classifier_MLP, self).__init__(dataset, type_prediction, results_dir, embedding_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_features = embedding_model.get_size_embedding_entity() * 2 + embedding_model.get_size_embedding_relation()
        self.net = MLP_model(n_features, n_units, dropout).to(self.device)

    def get_name(self):
        return "MLP"

    def create_training_data(self, queries_with_annotated_answers):
        out = []
        for query in tqdm(queries_with_annotated_answers):
            assert(query['query']['type'] == 1 or self.type_prediction == 'head')
            assert (query['query']['type'] == 0 or self.type_prediction == 'tail')
            ent = query['query']['ent']
            rel = query['query']['rel']
            emb_e = self.embedding_model.get_embedding_entity(ent)
            emb_r = self.embedding_model.get_embedding_relation(rel)
            for answer in query['annotated_answers']:
                a = answer['entity_id']
                emb_a = self.embedding_model.get_embedding_entity(a)
                data_entry = {}
                #X
                if self.type_prediction == 'head':
                    X = np.concatenate([emb_a, emb_r, emb_e])
                else:
                    X = np.concatenate([emb_e, emb_r, emb_a])
                #Y
                if answer['checked']:
                    Y = 1
                else:
                    Y = 0
                data_entry['X'] = X
                data_entry['Y'] = Y
                out.append(data_entry)
        return out

#    def train(self, training_data, model_path, batch_size=100, epochs=10):
#        n_units = 1000
#        dropout = 0.2
#        n_epochs = 10
#
#        dim1 = len(training_data)
#        dim2 = len(training_data[0]['X'])
#        x_train = np.zeros(shape=(dim1, dim2), dtype=np.float32)
#        y_train = np.zeros(shape=(dim1, 1), dtype=np.float32)
#        for idx, a in enumerate(training_data):
#            x_train[idx] = a['X']
#            y_train[idx][0] = a['Y']
#
#        model = Sequential()
#        model.add(Dense(n_units, input_shape=(batch_size, dim2)))
#        model.add(Dropout(dropout))
#        model.add(Dense(n_units))
#        model.add(Dropout(dropout))
#        model.add(Dense(1, activation='sigmoid'))
#        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#        model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2)

    def train(self, training_data, model_path, batch_size=100, epochs=10):
        # Load input data
        training_data_set = MLP_Dataset(training_data)
        train_data_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters())
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Start epoch {}".format(epoch))
            running_loss = 0.0
            for i, data in enumerate(train_data_loader, 0):
                inputs, labels = data
                inputs.to(self.device)
                labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        # Save model
        torch.save(self.net, model_path)

    def predict(self, answers):
        pass