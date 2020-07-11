import argparse
import datetime
import os
import sys
sys.path.append('..')
import time
import tensorflow as tf
import tqdm

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# tf.enable_eager_execution(config=config)

from backbones.resnet_v1 import ResNet_v1_50
from models.models import MyModel
from predict import get_embeddings
import numpy as np

import anyconfig
import munch

def parse_args(argv):
    parser = argparse.ArgumentParser(description='valid model')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args



def preprocess(image_path, config):
    image_raw = tf.io.read_file(image_path)
    # image = tf.image.decode_image(image_raw)
    image = tf.image.decode_png(image_raw)
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (config['image_size'], config['image_size']))

    # image = tf.image.resize(image, (224, 224))
    # image = tf.image.random_crop(image, size=[112, 112, 3])
    # image = tf.image.random_flip_left_right(image)

    image = image[None, ...]
    return image


def get_embeddings(model, image):
    # image = image[None, ...]
    prelogits, _, _ = model(image, training=False)
    embeddings = tf.nn.l2_normalize(prelogits, axis=-1)

    return embeddings

def cal_metric(sim, label, thresh):
    tp = tn = fp = fn = 0
    predict = tf.greater_equal(sim, thresh)
    for i in range(len(predict)):
        if predict[i] and label[i]:
            tp += 1
        elif predict[i] and not label[i]:
            fp += 1
        elif not predict[i] and label[i]:
            fn += 1
        else:
            tn += 1
    # print(f'tp:{tp} + tn:{tn}/predict:{len(predict)} ')
    # print(f'fp:{fp} + fn:{fn}')
    acc = (tp + tn) / len(predict)
    p = 0 if tp + fp == 0 else tp / (tp + fp)
    r = 0 if tp + fn == 0 else tp / (tp + fn)
    fpr = 0 if fp + tn == 0 else fp / (fp + tn)
    return acc, p, r, fpr

def cal_metric_fpr(sim, label, below_fpr=0.001):
    acc = p = r = thresh = 0
    for t in np.linspace(-1, 1, 100):
        thresh = t
        acc, p, r, fpr = cal_metric(sim, label, thresh)
        if fpr <= below_fpr:
            break

    return acc, p, r, thresh

def validate(model, valid_dataset, thresh, below_fpr= 0.001):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    imgs_1, imgs_2, issame_list = valid_dataset
    print("Loading data complete!")
    i = 0
    length = len(issame_list)
    sims = []
    for img1, img2, is_same in zip(imgs_1, imgs_2, issame_list): 
        img1 = tf.cast(img1, tf.float32)
        img1 = img1 / 255
        img1 = img1[None, ...]
        img2 = tf.cast(img2, tf.float32)
        img2 = img2 / 255
        img2 = img2[None, ...]
        embedding_1 = get_embeddings(model, img1)
        embedding_2 = get_embeddings(model, img2)
        sim = tf.matmul(embedding_1, tf.transpose(embedding_2))
        sims.append(sim)
        
    # labels = tf.constant(issame_list)
    acc, p, r, fpr = cal_metric(sims, issame_list, thresh)
    acc_fpr, p_fpr, r_fpr, thresh_fpr = cal_metric_fpr(sims, issame_list, below_fpr)

    return acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr

# acc = validate()
