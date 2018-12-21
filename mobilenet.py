from keras.applications.mobilenet import MobileNet
import tensorflow as tf

model = MobileNet(weights='imagenet', include_top=False, input_shape=(32,32,1))
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())