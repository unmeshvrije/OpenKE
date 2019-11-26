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

x_train = np.random.random((N_QUERIES, TOPK, N_FEATURES))
y_train = ((np.random.random((N_QUERIES, TOPK, 1)) * 42) % 2).astype(int)

x_test = np.random.random((20, TOPK, N_FEATURES))
y_test = ((np.random.random((20, TOPK, 1)) * 42) % 2).astype(int)

model = Sequential()
model.add(LSTM(1, input_shape=(TOPK, N_FEATURES),return_sequences=True))

model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2,validation_data=(x_test, y_test))
predict = model.predict(x_test)
