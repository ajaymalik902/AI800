import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import Input, Conv2D, Lambda, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras import Model
import random
from matplotlib import pyplot as plt

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
from PIL import Image

import os

width = 640
height = 480
chnls = 3

train_d = "train_data\\"
test_d = "test_data\\"

# train_d = "train_resize_512x512\\"
# test_d = "test_resize_512x512\\"

# train_d = "train_resize_1024x768\\"
# test_d = "test_resize_1024x768\\"

# train_d = "train_resize_128x128\\"
# test_d = "test_resize_128x128\\"
files = []
for file in os.listdir(train_d):
    if os.path.isfile(os.path.join(train_d, file)) and os.path.exists(os.path.join(train_d, file.replace(".jpg", "_masks\\"))):
        files.append(os.path.join(train_d, file))

files = files[:5]

X_train = np.zeros((len(files), width, height, chnls), dtype=np.uint8)
Y_train = np.zeros((len(files), width, height, 1), dtype=bool)

if not os.path.exists("train_resize_128x128/"):
    os.mkdir("train_resize_128x128/")

if not os.path.exists("test_resize_128x128/"):
    os.mkdir("test_resize_128x128/")

# test_resized = "test_resize_512x512/"
# train_resized = "train_resize_512x512/"

test_resized = "test_resize_128x128/"
train_resized = "train_resize_128x128/"

# test_resized = "test_resize_1024x768/"
# train_resized = "train_resize_1024x768/"

print("resizing training images and masks")
for n, id_ in tqdm(enumerate(files), total=len(files)):
    path = id_
    img = imread(path)[:,:,:chnls]
    img = resize(img, (width, height), mode='constant', preserve_range=True, anti_aliasing=False)
    # img.save(os.path.join(train_resized, path.split("\\")[1]))

    mask_adr = os.path.join(train_resized, path.split("\\")[1].replace(".jpg", "_masks\\"))
    if not os.path.exists(mask_adr):
        os.mkdir(mask_adr)
    X_train[n] = img
    msk_pth = path.replace('.jpg', '_masks\\')
    mask = np.zeros((width, height, 1), dtype=bool)
    import pdb;pdb.set_trace()
    for mask_file in next(os.walk(msk_pth))[2]:
        mask_ = Image.open(msk_pth+mask_file)
        imshow(np.array(mask_))
        plt.show()
        mask_ = resize(mask_, (width, height), mode='constant', preserve_range=True, anti_aliasing=False)
        # mask_.save(os.path.join(mask_adr, mask_file))
        mask_ = np.expand_dims(mask_, axis=-1)
        print(mask_)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask


image_x = random.randint(0, len(files)-1)
Image.fromarray(X_train[image_x]).show()
# Image._show(X_train[image_x])
Image.fromarray(np.squeeze(Y_train[image_x])).show()
# imshow(np.squeeze(Y_train[image_x]))

test_files = []
for file in os.listdir(test_d):
    if os.path.isfile(os.path.join(test_d, file)) and os.path.exists(os.path.join(test_d, file.replace(".jpg", "_masks\\"))):
        test_files.append(os.path.join(test_d, file))

test_files=test_files[:5]

X_test = np.zeros((len(files), width, height, chnls), dtype=np.uint8)
Y_test = np.zeros((len(files), width, height, 1), dtype=bool)

print("resizing testing images and masks")
for n, id_ in tqdm(enumerate(test_files), total=len(test_files)):
    path = id_
    img = imread(path)[:,:,:chnls]
    img = resize(img, (width, height), mode='constant', preserve_range=True, anti_aliasing=False)
    # img.save(os.path.join(test_resized, path.split("\\")[1]))

    mask_adr = os.path.join(test_resized, path.split("\\")[1].replace(".jpg", "_masks\\"))
    if not os.path.exists(mask_adr):
        os.mkdir(mask_adr)
    X_test[n] = img
    msk_pth = path.replace('.jpg', '_masks\\')
    mask = np.zeros((width, height, 1), dtype=bool)
    for mask_file in next(os.walk(msk_pth))[2]:
        mask_ = imread(msk_pth + mask_file)[:, :, :chnls]
        mask_ = resize(mask_, (width, height), mode='constant', preserve_range=True, anti_aliasing=False)
        # mask_.save(os.path.join(mask_adr, mask_file))
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_)
    Y_test[n] = mask

image_x = random.randint(0, len(test_files)-1)
Image.fromarray(X_test[image_x]).show()
# Image._show(X_train[image_x])
Image.fromarray(np.squeeze(Y_test[image_x])).show()
# imshow(np.squeeze(Y_train[image_x]))

inputs = Input((height,  width, chnls))

lmda = Lambda(lambda x: x/255)(inputs)

con_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(lmda)
con_1 = Dropout(0.1)(con_1)
con_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_1)
pool_1 = MaxPooling2D((2, 2))(con_1)

con_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_1)
con_2 = Dropout(0.1)(con_2)
con_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_2)
pool_2 = MaxPooling2D((2, 2))(con_2)

con_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_2)
con_3 = Dropout(0.2)(con_3)
con_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_3)
pool_3 = MaxPooling2D((2, 2))(con_3)

con_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_3)
con_4 = Dropout(0.2)(con_4)
con_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_4)
pool_4 = MaxPooling2D(pool_size=(2, 2))(con_4)

con_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_4)
con_5 = Dropout(0.3)(con_5)
con_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_5)

# Expansive path
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(con_5)
u6 = concatenate([u6, con_4])
con_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
con_6 = Dropout(0.2)(con_6)
con_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(con_6)
u7 = concatenate([u7, con_3])
con_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
con_7 = Dropout(0.2)(con_7)
con_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(con_7)
u8 = concatenate([u8, con_2])
con_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
con_8 = Dropout(0.1)(con_8)
con_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(con_8)
u9 = concatenate([u9, con_1], axis=3)
con_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
con_9 = Dropout(0.1)(con_9)
con_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(con_9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(con_9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

################################
# Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_para.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
model.save('saved_models/image_seg_unet.h5')

####################################

idx = random.randint(0, len(X_train)-1)

preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print(preds_test_t)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

