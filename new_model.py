import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, concatenate, GlobalAvgPool2D, Input, Dense

def fire_model(filters,prev):
    new_filter = filters
    filters = filters // 4
    print(new_filter)
    conv1 = Conv2D(filters, (1,1), padding='same',activation='relu')(prev)
    conv2 = Conv2D(new_filter, (1,1), padding='same',activation='relu')(conv1)
    conv3 = Conv2D(new_filter, (3,3), padding='same',activation='relu')(conv1)
    concat = concatenate([conv2, conv3], axis=0)
    return concat

def main():
    ip = Input(shape=(32,32,1))
    conv1 = Conv2D(64, (3,3), strides=(1,1), padding='same',activation='relu')(ip)
    maxpool1 = MaxPool2D(pool_size=(2,2), strides=(2,2))(conv1)
    fire2 = fire_model(128, maxpool1)
    fire3 = fire_model(128, fire2)
    maxpool2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(fire3)
    fire4 = fire_model(256, maxpool2)
    fire5 = fire_model(256, fire4)
    maxpool3 = MaxPool2D(pool_size=(2,2), strides=(2,2))(fire5)
    fire6 = fire_model(384, maxpool3)
    fire7 = fire_model(384, fire6)
    fire8 = fire_model(512, fire7)
    fire9 = fire_model(512, fire8)
    last = Dense(36, activation='softmax')(fire9)

    model = Model(ip, last)

if __name__ == '__main__':
    main()