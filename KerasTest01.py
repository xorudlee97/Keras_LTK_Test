import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# csv 파일 받기
file_dir = os.path.dirname(os.path.realpath(__file__))
csv_kospi = pd.read_csv(file_dir+"\\kospi200test.csv", encoding='cp949')
bat_size = 1

print(csv_kospi)

# 데이터 정렬
csv_kospi_list = np.transpose(csv_kospi[:-1])
csv_kospi_list = csv_kospi_list[1:-1]
csv_kospi_list = np.transpose(csv_kospi_list)

csv_kospi_X = csv_kospi_list.values[:,0:3]
csv_kospi_Y = csv_kospi_list.values[:,3]

Kospi_X_train, Kospi_X_test, Kospi_Y_train, Kospi_Y_test = train_test_split(
    csv_kospi_X, csv_kospi_Y, test_size = 0.5,shuffle = False
)

train_X_Arr1_Shape = Kospi_X_train.shape[0]
train_X_Arr2_Sahpe = Kospi_X_train.shape[1]
test_X_Arr1_Shape = Kospi_X_test.shape[0]
test_X_Arr2_Sahpe = Kospi_X_test.shape[1]

Kospi_X_train = np.reshape(Kospi_X_train,(train_X_Arr1_Shape, train_X_Arr2_Sahpe, 1))
Kospi_X_test = np.reshape(Kospi_X_test,(test_X_Arr1_Shape, test_X_Arr2_Sahpe, 1))
print(Kospi_X_train.shape)
print(Kospi_Y_train.shape)
print(Kospi_X_test.shape)
print(Kospi_Y_test.shape)


# 데이터 분할

# 모델 작업 시작
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(LSTM(128, batch_input_shape=(bat_size,Kospi_X_train.shape[1],1), stateful=True))
model.add(Dense(64, activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

num_epochs = 10
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stopping = EarlyStopping(monitor='mean_squared_error', patience=30, mode='auto')
for epoch_idx in range(num_epochs):
    print('epochs:' + str(epoch_idx))
    model.fit(Kospi_X_train, Kospi_Y_train, epochs=100, batch_size=bat_size, 
              verbose=1, shuffle=False,
              validation_data=(Kospi_X_test, Kospi_Y_train),
              callbacks=[early_stopping])#, tb_hist])
    model.reset_states()
mse, _ = model.evaluate(Kospi_X_train, Kospi_Y_train, batch_size = bat_size)
print("mse :", mse)
model.reset_states()

Kospi_Y_predict = model.predict(Kospi_X_test, batch_size=1)
print(Kospi_Y_predict.need())

# 2024.11
