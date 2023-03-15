import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

width = 128
height = 128
chnls = 3

files = []
for file in os.listdir("raw_images"):
    if os.path.isfile(os.path.join("raw_images", file)):
        files.append(os.path.join("raw_images", file))

X_train = np.zeros((len(files), height, width, chnls), dtype=np.uint8)
Y_train = np.zeros((len(files), height, width, 1), dtype=np.bool)

for n, id_ in tqdm(enumerate(files), total=len(files)):
    path=id_
    img=imread(path)[:,:,:chnls]
    img=resize(img,(height,width), mode='constant', preserve_range=True)
    X_train[n] = img
    msk_pth = path.split('\\')[1].replace('.jpg', '_masks\\')
    mask=np.zeros((height,width,1),dtype=np.bool)
    for mask_file in next(os.walk(msk_pth))[2]:
        mask_ = imread(msk_pth+mask_file)
        mask_ = np.expand_dims(resize(mask_,(height,width),mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask
