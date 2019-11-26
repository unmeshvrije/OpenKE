import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

percent_test = 10
TOPK = 10
DIM = 200
N_QUERIES = 500

# each query has topK answers (sequence of 50 answers)
# each answer has 200 features

# 1 for score, 1 for rank, 3 for (h,r,t) ids and 1 for answer entity
N_FEATURES = DIM + 6

#N_TOTAL = 100#len()
#N_TEST = int(N_TOTAL * percent_test / 100)
#N_TRAIN = int(N_TOTAL - N_TEST)

#print(N_TRAIN)
#print(N_FEATURES)

x_train = np.random.random((N_QUERIES, TOPK, N_FEATURES))
#data = [[i for i in range(100)]]
#data = np.array(x_train, dtype=float)

y_train = ((np.random.random(N_QUERIES) * 42) % 2).astype(int)
#target = np.array(y_train, dtype=int)

#data = data.reshape((1, N_TRAIN, N_FEATURES))
#target = target.reshape((1, N_TRAIN, 1))


x_test = np.random.random((20, TOPK, N_FEATURES))
#x_test=[i for i in range(100,200)]
#x_test=np.array(x_test).reshape((1, N_TEST, N_FEATURES));

y_test = ((np.random.random(20) * 42) % 2).astype(int)
#y_test=np.array(y_test).reshape(1, N_TEST, 1)


model = Sequential()
model.add(LSTM(1, input_shape=(TOPK, N_FEATURES),return_sequences=False))
#model.add(Dense((N_FEATURES,N_TRAIN)))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2,validation_data=(x_test, y_test))


predict = model.predict(data)
