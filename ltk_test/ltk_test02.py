from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop

import ltk_test01 as ltk

# CIFAR_10은 3채널로 구성된 32*32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수의 정의
BATCH_SIZE = 128
NB_EPOCH = 100
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2

# 데이터 셋 불러 오기
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# X_train = X_train[0:300]
# Y_train = Y_train[0:300]
X_list_train = []
Y_list_train = []
split_line = 300

for i in range(10):
    X_train_temp = X_train[i*split_line:(i+1)*split_line]
    Y_train_temp = Y_train[i*split_line:(i+1)*split_line]
    # 범주형으로 변환
    X_train_temp = X_train_temp.astype('float32') / 255
    Y_train_temp = np_utils.to_categorical(Y_train_temp, NB_CLASSES)
    X_list_train.append(X_train_temp)
    Y_list_train.append(Y_train_temp)
    X_train_temp = []
    Y_train_temp = []

# 범주형으로 변환
X_test = X_test.astype('float32') / 255
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.preprocessing.image import ImageDataGenerator

ltk_keras = ltk
model = KerasClassifier(build_fn= ltk_keras.Build_Model, verbose=1)
hyperparameters = ltk_keras.create_hyperparameters()
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=-1, cv=3, verbose=1)
Max_acc = 0

data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True
)
for i in range(10):
    search.fit(X_list_train[i], Y_list_train[i])
    print(search.best_params_)
    model = ltk_keras.Build_Model(search.best_params_["keep_prob"], search.best_params_['optimizer'], search.best_params_['kernel_regularizer_num'])
    history = model.fit_generator(
        data_generator.flow(X_list_train[i], Y_list_train[i], batch_size=search.best_params_["batch_size"]),
        steps_per_epoch= IMG_ROWS * IMG_COLS * IMG_CHANNELS,
        epochs = 10,
        validation_data = (X_test, Y_test),
        verbose=1
    )
    loss, acc = model.evaluate(X_test, Y_test)
    print("정답률: ", acc)
    if Max_acc < acc:
        Max_acc = acc
print("최종정답률: ", acc)
