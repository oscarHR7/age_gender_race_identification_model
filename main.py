import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

DATA_DIR = 'D:\TKFace'
TRAIN_TEST_SPLIT = 0.7
IMAGE_HEIGHT, IMAGE_WIDTH = 198, 198
IMAGE_CHANNELS = 3
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = {g: i for i, g in ID_GENDER_MAP.items()}
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = {r: i for i, r in ID_RACE_MAP.items()}


def parse_filepath(filepath):#通过解析数据集照片命名获得相对应的所需信息
    try:
        path, filename = os.path.split(filepath)#提取出路径和照片的名称
        filename, ext = os.path.splitext(filename)#分离照片名中的扩展名和原始照片名
        age, gender, race, _ = filename.split("_")
        return int(age), ID_GENDER_MAP[int(gender)], ID_RACE_MAP[int(race)]
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None, None


files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
attributes = list(map(parse_filepath, files))#解析所有照片命名所对应的所需信息

df_origin = pd.DataFrame(attributes)
df_origin['file'] = files
df_origin.columns = ['age', 'gender', 'race', 'file']#添加列标签
df_origin = df_origin.dropna()#有缺失数据的一行将被滤除
'''箱线图观察数据
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))#开辟一行两列的图片，通过ax1和ax2可以对图片进行相关定制
sns.boxplot(data=df_origin, x='gender', y='age', ax=ax1)
sns.boxplot(data=df_origin, x='race', y='age', ax=ax2)
plt.show()
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_origin, x='gender', y='age', hue='race')
plt.show()
df_origin.groupby(by=['race', 'gender'])['age'].count().plot(kind='bar')#柱状图
df_origin['age'].hist()柱状图
df_origin['age'].describe()分析年龄段的数据信息
'''

df = df_origin.copy()
df = df[(df['age'] > 10) & (df['age'] < 65)]#df[]这种情况一次只能选取行或者列，即一次选取中，只能为行或者列设置筛选条件（只能为一个维度设置筛选条件）。


df_random = np.random.permutation(len(df))#对原有数据集合打乱
train_up_to = int(len(df)*TRAIN_TEST_SPLIT)#训练集占原数据集的百分比
#制作训练集和测试集 切片只切的是序号而没有其他信息
train_idx = df_random[:train_up_to]
test_idx = df_random[train_up_to:]
#将 train_idx 进一步拆分为训练和验证集
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
#将gender和race换成对应的id表示
df['gender_id'] = df['gender'].map(lambda gender: GENDER_ID_MAP[gender])
df['race_id'] = df['race'].map(lambda race: RACE_ID_MAP[race])

max_age = df['age'].max()

#生成器的制作
def get_data_generator(df, indices, for_training, batch_size=16):
    images, ages, races, genders = [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]#对应行号的所有信息
            file, age, race, gender = r['file'], r['age'], r['race_id'], r['gender_id']
            im = Image.open(file)
            im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            im = np.array(im)/255.0#归一化
            images.append(im)
            ages.append(age / max_age)#归一化
            races.append(to_categorical(race, len(RACE_ID_MAP)))
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                images, ages, races, genders = [], [], [], []
        if not for_training:
            break


#model的训练
def conv_block(input_data, filters=32, bn=True, pool=True, kernel_size=3, activation='relu'):
    return_x = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)(input_data)
    if bn:
        return_x = BatchNormalization()(return_x)
    if pool:
        return_x = MaxPool2D()(return_x)
    return return_x


input_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
x = conv_block(input_layer, filters=32, bn=False, pool=False)
x = conv_block(x, filters=32*2)
x = conv_block(x, filters=32*3)
x = conv_block(x, filters=32*4)
x = conv_block(x, filters=32*5)
x = conv_block(x, filters=32*6)
bottleneck = GlobalMaxPool2D()(x)

age_x = Dense(128, activation='relu')(bottleneck)
age_output = Dense(1, activation='sigmoid', name='age_output')(age_x)

race_x = Dense(128, activation='relu')(bottleneck)
race_output = Dense(len(RACE_ID_MAP), activation='softmax', name='race_output')(race_x)

gender_x = Dense(128, activation='relu')(bottleneck)
gender_output = Dense(len(GENDER_ID_MAP), activation='softmax', name='gender_output')(gender_x)

model = Model(inputs=input_layer, outputs=[age_output, race_output, gender_output])
model.compile(optimizer='rmsprop',
              loss={
                  'age_output': 'mse',
                  'race_output': 'categorical_crossentropy',
                  'gender_output': 'categorical_crossentropy'},
              loss_weights={
                  'age_output': 2.,
                  'race_output': 1.5,
                  'gender_output': 1.},
              metrics={
                  'age_output': 'mae',
                  'race_output': 'accuracy',
                  'gender_output': 'accuracy'})

from tensorflow.keras.callbacks import ModelCheckpoint
batch_size = 64
valid_batch_size = 64
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)
'''
#callbacks = [
#    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
#]
'''

history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=10,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)


def plot_train_history(history):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].plot(history.history['race_output_accuracy'], label='Race Train accuracy')
    axes[0].plot(history.history['val_race_output_accuracy'], label='Race Val accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(history.history['gender_output_accuracy'], label='Gender Train accuracy')
    axes[1].plot(history.history['val_gender_output_accuracy'], label='Gener Val accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

    axes[2].plot(history.history['age_output_mae'], label='Age Train MAE')
    axes[2].plot(history.history['val_age_output_mae'], label='Age Val MAE')
    axes[2].set_xlabel('Epochs')
    axes[2].legend()

    axes[3].plot(history.history['loss'], label='Training loss')
    axes[3].plot(history.history['val_loss'], label='Validation loss')
    axes[3].set_xlabel('Epochs')
    axes[3].legend()
#观察是否存在过拟合
plot_train_history(history)
plt.show()
#测试预测集
test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
print(dict(zip(model.metrics_names, model.evaluate(test_gen, steps=len(test_idx)//128))))

