import mxnet as mx
import pickle
from mxnet import ndarray as nd

def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3

    imgs_1 = []
    imgs_2 = []
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = img.asnumpy()
        if i % 2 == 0:
            imgs_1.append(img)
        else:
            imgs_2.append(img)
    return (imgs_1, imgs_2, issame_list)