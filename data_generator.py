import numpy as np
import keras
import glob
import pickle

class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    # TODO
    def __init__(self, list_IDs, input_folder, type_pred, db="fb15k237", emb_model="transe", topk=10, batch_size=10, dim_x=(1000,10,605), dim_y=(1000,10, 1), n_channels=1, n_classes=2, shuffle=True):
       #  'Initialization'
       self.dim_x = dim_x # dim is actually (samples, topk, features)
       self.dim_y = dim_y # dim is actually (samples, topk, 1)
       self.batch_size = batch_size
       self.list_IDs = list_IDs
       self.n_channels = n_channels
       self.n_classes = n_classes
       self.shuffle = shuffle
       self.on_epoch_end()
       self.folder = input_folder
       self.type_pred = type_pred
       self.db = db
       self.emb_model = emb_model
       self.topk = topk

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        print("*"*80, X.shape)
        X = np.reshape(X, (self.batch_size * self.dim_x[0], self.dim_x[1], self.dim_x[2]))
        y = np.reshape(y, (self.batch_size * self.dim_y[0], self.dim_y[1], self.dim_y[2]))
        return X, y

    def on_epoch_end(self):
    #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim_x, self.n_channels))
        X = [] #np.empty((self.batch_size, *self.dim_x, self.n_channels))
        y = [] #np.empty((self.batch_size, *self.dim_y, self.n_channels), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # fb15k237-transe-training-topk-50-tail-batch-131.pkl
            batch_file = self.folder + self.db + "-" + self.emb_model + "-training-topk-" + str(self.topk) + "-" + self.type_pred + "-batch-"+str(ID) + ".pkl"
            with open(batch_file, 'rb') as fin:
                training_data = pickle.load(fin)

            Xi = training_data['x_' + self.type_pred]
            N = len(Xi)
            N_FEATURES = len(Xi[0])
            if N != self.dim_x[0] * self.dim_x[1]:
                print(batch_file + "   XXXXXX ")
            yi = np.array(training_data['y_' + self.type_pred], dtype = np.int32)

            #TODO: this reshaping should not be necessary. Confirm it.
            Xi = np.reshape(Xi, (N//self.topk, self.topk, N_FEATURES))
            yi = np.reshape(yi, (N//self.topk, self.topk, 1))
            X.append(Xi)
            y.append(yi)

        return np.array(X), np.array(y)#keras.utils.to_categorical(y, num_classes=self.n_classes))
