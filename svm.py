import numpy as np

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM

from sklearn.svm import SVC

percent_test = 10
TOPK = 10
DIM = 200

# 1 for score, 1 for rank, 3 for (h,r,t) ids and 1 for answer entity
N_FEATURES = DIM + 6
N_TOTAL = 100#len()
N_TEST = int(N_TOTAL * percent_test / 100)
N_TRAIN = int(N_TOTAL - N_TEST)

print(N_TRAIN)
print(N_FEATURES)
x_train_shape = (N_TRAIN, N_FEATURES)
y_train_shape = (N_TRAIN, 1)

x_train = np.random.random((N_TRAIN, N_FEATURES))
#data = [[i for i in range(100)]]
#data = np.array(x_train, dtype=float)

y_train = ((np.random.random(N_TRAIN) * 42) % 2).astype(int)
#target = np.array(y_train, dtype=int)

#data = data.reshape((1, N_TRAIN, N_FEATURES))
#target = target.reshape((1, N_TRAIN, 1))


x_test = np.random.random((N_TEST, N_FEATURES))
#x_test=[i for i in range(100,200)]
#x_test=np.array(x_test).reshape((1, N_TEST, N_FEATURES));

y_test = ((np.random.random(N_TEST) * 42) % 2).astype(int)
#y_test=np.array(y_test).reshape(1, N_TEST, 1)

model = SVC()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))
#model = Sequential()
#model.add(LSTM(N_TRAIN, input_shape=(N_FEATURES,),return_sequences=False))
#model.add(Dense((N_FEATURES,N_TRAIN)))
#model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
#model.fit(data, target, nb_epoch=100, batch_size=1, verbose=2,validation_data=(x_test, y_test))

a,b = (model.predict(x_test) == y_test).sum() , len(y_test)
print(a, b)

