import sys
import os
import tensorflow as tf
import numpy as np
import random

seed = 42
if '--seed' in sys.argv:
    seed = int(sys.argv[sys.argv.index('--seed')+1])

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed+1)
random.seed(seed+2)

# use only as much gpu memory as necessary
from keras import backend as K
tf.set_random_seed(seed+3)
tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tf_config))


import argparse
import re
import math
import glob
import pandas as pd
import skimage.io
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.color
import scipy.ndimage.morphology
from scipy.misc import imresize
from tqdm import tqdm
from multiprocessing.pool import Pool
import dataset
import rle
import tta
import visualisation
import sklearn.cluster

import keras
import keras.applications
from keras.layers import MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import concatenate, add
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import config
import utils
import matplotlib.pyplot as plt
from dataset import CROP_SIZE
import metrics
import scipy.signal

import PIL.Image


dice_weight = 10.0
bce_weight = 4.0
bce_center_weight = 20.0*20.0
vector_loss_weight = 1e-9 / 2 / config.IMG_SCALE / config.IMG_SCALE
vector_loss_weight_scaled = 1e-6/2
watershed_vector_loss_weight = 2e-4
watershed_energy_loss_weight = [6.0, 4.0, 4.0]
area_weight = 1.0e-3


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return 1.0-jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)


def build_model_vector_unet4_watershed(input_shape=(CROP_SIZE, CROP_SIZE, 3), filters=64,
                                       output_mask_only=False, output_area=True, input_channels_used=1):
    inputs = keras.layers.Input(shape=input_shape)

    if input_shape[:-1] == input_channels_used:
        x = inputs
    else:
        x = keras.layers.Lambda(lambda x: x[:, :, :, :input_channels_used], output_shape=input_shape[:-1]+(input_channels_used,))(inputs)

    def add_levels(input_tensor, sizes):
        filters = sizes[0]

        down = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        down = Conv2D(filters, (3, 3), activation='relu', padding='same')(down)

        if len(sizes) == 1:
            return down

        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down)

        subnet = add_levels(down_pool, sizes[1:])

        up = UpSampling2D((2, 2))(subnet)
        up = concatenate([down, up], axis=3)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        # up = BatchNormalization()(up)
        up = Activation('relu')(up)
        return up

    unet = add_levels(input_tensor=x, sizes=[
        filters*2, filters*2, filters*4, filters*8, filters*8, filters*8])

    if not output_mask_only:
        conv_vector = Conv2D(2, (1, 1))(unet)
        conv_watershed_vector = Conv2D(2, (1, 1))(unet)
        conv_area = Conv2D(1, (1, 1))(unet)
        conv_watershed = concatenate([unet, conv_vector, conv_watershed_vector, conv_area])

        sub_unet = add_levels(input_tensor=conv_watershed, sizes=[128, 256, 256, 512])

        conv_mask = Conv2D(1, (1, 1), activation='sigmoid')(sub_unet)
        conv_mask_center = Conv2D(1, (1, 1), activation='sigmoid')(sub_unet)
        conv_watershed_energy = Conv2D(dataset.WATERSHED_ENERGY_LEVELS, (1, 1), activation='sigmoid')(sub_unet)

        outputs = [conv_mask, conv_vector, conv_mask_center, conv_watershed_vector, conv_watershed_energy]
        if output_area:
            outputs += [conv_area]
        output = concatenate(outputs)
    else:
        conv_mask = Conv2D(1, (1, 1), activation='sigmoid')(unet)
        output = conv_mask

    model = Model(inputs, output)
    # model.summary()
    return model


def build_model_vector_unet7_watershed(input_shape=(CROP_SIZE, CROP_SIZE, 3), filters=64,
                                       output_mask_only=False, output_area=True, input_channels_used=1):
    inputs = keras.layers.Input(shape=input_shape)

    if input_shape[:-1] == input_channels_used:
        x = inputs
    else:
        x = keras.layers.Lambda(lambda x: x[:, :, :, :input_channels_used], output_shape=input_shape[:-1]+(input_channels_used,))(inputs)

    def add_levels(input_tensor, sizes):
        filters = sizes[0]

        down = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        down = Conv2D(filters, (3, 3), activation='relu', padding='same')(down)

        if len(sizes) == 1:
            return down

        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down)

        subnet = add_levels(down_pool, sizes[1:])

        up = upsampling2x_res2(subnet)
        up = concatenate([down, up], axis=3)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        # up = BatchNormalization()(up)
        up = Activation('relu')(up)
        return up

    unet = add_levels(input_tensor=x, sizes=[
        filters*2, filters*2, filters*4, filters*8, filters*8, filters*8])

    if not output_mask_only:
        conv_vector = Conv2D(2, (1, 1))(unet)
        conv_watershed_vector = Conv2D(2, (1, 1))(unet)
        conv_area = Conv2D(1, (1, 1))(unet)
        conv_watershed = concatenate([unet, conv_vector, conv_watershed_vector, conv_area])

        sub_unet = add_levels(input_tensor=conv_watershed, sizes=[128, 256, 256, 512])

        conv_mask = Conv2D(1, (1, 1), activation='sigmoid')(sub_unet)
        conv_mask_center = Conv2D(1, (1, 1), activation='sigmoid')(sub_unet)
        conv_watershed_energy = Conv2D(dataset.WATERSHED_ENERGY_LEVELS, (1, 1), activation='sigmoid')(sub_unet)

        outputs = [conv_mask, conv_vector, conv_mask_center, conv_watershed_vector, conv_watershed_energy]
        if output_area:
            outputs += [conv_area]
        output = concatenate(outputs)
    else:
        conv_mask = Conv2D(1, (1, 1), activation='sigmoid')(unet)
        output = conv_mask

    model = Model(inputs, output)
    # model.summary()
    return model


class Upsample2xBilinear(keras.layers.Layer):
    def call(self, x, **kwargs):
        height_factor = 2
        width_factor = 2

        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = tf.image.resize_bilinear(x, new_shape)
        x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x

    def compute_output_shape(self, input_shape):
        height = 2 * input_shape[1] if input_shape[1] is not None else None
        width = 2 * input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])


def upsampling2x_res2(input_tensor):
    linear = Upsample2xBilinear()(input_tensor)
    # print('filters:', K.int_shape(input_tensor))
    residual = Conv2DTranspose(filters=K.int_shape(input_tensor)[-1] // 2,
                               kernel_size=(3, 3),
                               strides=2,
                               activation='relu',
                               kernel_initializer='zeros',
                               padding='same')(input_tensor)
    residual = Conv2D(filters=K.int_shape(input_tensor)[-1], kernel_size=(1, 1))(residual)
    return add([linear, residual])


def combined_loss(y_true, y_pred):
    mask_true = y_true[:, :, :, 0]
    mask_pred = y_pred[:, :, :, 0]

    dice = dice_coef_loss(mask_true, mask_pred)
    bce = binary_crossentropy(mask_true, mask_pred)

    bce_center = binary_crossentropy(y_true[:, :, :, 3], y_pred[:, :, :, 3])

    mask_true_f = K.flatten(mask_true)

    vector_true_1_f = K.flatten(y_true[:, :, :, 1])
    vector_pred_1_f = K.flatten(y_pred[:, :, :, 1])

    vector_true_2_f = K.flatten(y_true[:, :, :, 2])
    vector_pred_2_f = K.flatten(y_pred[:, :, :, 2])

    vector_loss = K.sum(
        mask_true_f*(K.square(vector_true_1_f-vector_pred_1_f) + K.square(vector_true_2_f-vector_pred_2_f))
    )

    return dice * dice_weight + bce * bce_weight + vector_loss * vector_loss_weight + bce_center * bce_center_weight


def combined_loss_watershed(y_true, y_pred):
    # y:
    # 0: mask
    # 1,2: vector to center
    # 3: center 3x3 patch
    # 4,5: vector from border
    # 6-8: watershed energy levels
    # 9: area or mask

    mask_true = y_true[:, :, :, 0]
    mask_pred = y_pred[:, :, :, 0]

    dice = dice_coef_loss(mask_true, mask_pred)
    bce = binary_crossentropy(mask_true, mask_pred)
    bce_center = binary_crossentropy(y_true[:, :, :, 3], y_pred[:, :, :, 3])

    return dice * dice_weight +\
           bce * bce_weight +\
           bce_center * bce_center_weight + \
           vector_m(y_true, y_pred) + \
           watershed_vector_m(y_true, y_pred) + \
           watershed_energy_m(y_true, y_pred)


def simple_loss_watershed(y_true, y_pred):
    # y:
    # 0: mask
    # 1,2: vector to center
    # 3: center 3x3 patch
    # 4,5: vector from border
    # 6-8: watershed energy levels
    # 9: area or mask

    mask_true = y_true[:, :, :, 0]
    mask_pred = y_pred[:, :, :, 0]

    dice = dice_coef_loss(mask_true, mask_pred)
    bce = binary_crossentropy(mask_true, mask_pred)
    bce_center = binary_crossentropy(y_true[:, :, :, 3], y_pred[:, :, :, 3])

    return dice * dice_weight +\
           bce * bce_weight +\
           bce_center * bce_center_weight + \
           watershed_energy_m(y_true, y_pred)


def bce_m(y_true, y_pred):
    mask_true = y_true[:, :, :, 0]
    mask_pred = y_pred[:, :, :, 0]

    return bce_weight * binary_crossentropy(mask_true, mask_pred)


def bce_center_m(y_true, y_pred):
    mask_true = y_true[:, :, :, 3]
    mask_pred = y_pred[:, :, :, 3]

    return bce_center_weight * binary_crossentropy(mask_true, mask_pred)


def dice_m(y_true, y_pred):
    mask_true = y_true[:, :, :, 0]
    mask_pred = y_pred[:, :, :, 0]
    return dice_weight * dice_coef_loss(mask_true, mask_pred)


def area_m(y_true, y_pred):
    return K.mean(K.abs(K.sqrt(y_true[:, :, :, -1]) - K.sqrt(y_pred[:, :, :, -1]))) * area_weight


def vector_m(y_true, y_pred):
    mask_true = y_true[:, :, :, 0]
    mask_true_f = K.flatten(mask_true)
    area_true_f = K.flatten(y_true[:, :, :, -1])

    vector_true_1_f = K.flatten(y_true[:, :, :, 1])
    vector_pred_1_f = K.flatten(y_pred[:, :, :, 1])

    vector_true_2_f = K.flatten(y_true[:, :, :, 2])
    vector_pred_2_f = K.flatten(y_pred[:, :, :, 2])

    vector_loss = K.sum(
        mask_true_f / (area_true_f + 0.1) *
        (K.square(vector_true_1_f - vector_pred_1_f) + K.square(vector_true_2_f - vector_pred_2_f))
    )

    return vector_loss * vector_loss_weight_scaled


def watershed_vector_m(y_true, y_pred):
    mask_true = y_true[:, :, :, 0]
    mask_true_f = K.flatten(mask_true)

    vector_true_1_f = K.flatten(y_true[:, :, :, 4])
    vector_pred_1_f = K.flatten(y_pred[:, :, :, 4])

    vector_true_2_f = K.flatten(y_true[:, :, :, 5])
    vector_pred_2_f = K.flatten(y_pred[:, :, :, 5])

    vector_loss = K.sum(
        mask_true_f * (K.square(vector_true_1_f - vector_pred_1_f) + K.square(vector_true_2_f - vector_pred_2_f))
    )

    return vector_loss * watershed_vector_loss_weight


def watershed_energy_m(y_true, y_pred):
    watershed_energy_loss = sum(
        [
            watershed_energy_loss_weight[i]*binary_crossentropy(y_true[:, :, :, i+6], y_pred[:, :, :, i+6])
            for i in range(dataset.WATERSHED_ENERGY_LEVELS)
        ]
    )
    return watershed_energy_loss


class ModelInfo:
    def __init__(self,
                 factory,
                 preprocess_input,
                 args,
                 dataset_args,
                 default_batch_size,
                 loss=combined_loss_watershed,
                 train_frozen_epochs=0,
                 train_lr_multiplier=1.0):
        self.factory = factory
        self.loss = loss
        self.preprocess_input = preprocess_input
        self.args = args
        self.dataset_args = dataset_args
        self.default_batch_size = default_batch_size
        self.train_frozen_epochs = train_frozen_epochs
        self.train_lr_multiplier = train_lr_multiplier


MODELS = {
    'unet4_64_wshed': ModelInfo(
        factory=build_model_vector_unet4_watershed,
        preprocess_input=dataset.preprocess_func_mono3,
        args=dict(filters=64, input_channels_used=1),
        dataset_args=dict(output_watershed=True),
        default_batch_size=4,
        train_frozen_epochs=0
    ),
    'unet7_64_wshed': ModelInfo(
        factory=build_model_vector_unet7_watershed,
        preprocess_input=dataset.preprocess_func_mono3,
        args=dict(filters=64, input_channels_used=1),
        dataset_args=dict(output_watershed=True),
        default_batch_size=4,
        train_frozen_epochs=0
    ),
}


MODELS['unet4_64_wshed_s1'] = MODELS['unet4_64_wshed']
MODELS['unet4_64_wshed_s2'] = MODELS['unet4_64_wshed']
MODELS['unet4_64_wshed_s3'] = MODELS['unet4_64_wshed']
MODELS['unet4_64_wshed_s4'] = MODELS['unet4_64_wshed']

MODELS['unet4_64_wshed_h1'] = MODELS['unet4_64_wshed_hed']
MODELS['unet4_64_wshed_h2'] = MODELS['unet4_64_wshed_hed']
MODELS['unet4_64_wshed_h3'] = MODELS['unet4_64_wshed_hed']
MODELS['unet4_64_wshed_h4'] = MODELS['unet4_64_wshed_hed']


def train(model_name, fold, run, weights):
    model_info = MODELS[model_name]
    model = model_info.factory(**model_info.args)
    use_watershed = model_info.dataset_args['output_watershed']

    data = dataset.UVectorNetDataset(fold=fold,
                                     seed=seed,
                                     batch_size=model_info.default_batch_size,
                                     preprocess_func=model_info.preprocess_input,
                                     **model_info.dataset_args)

    loss = model_info.loss
    metrics = [bce_m, bce_center_m, dice_m, vector_m]
    if use_watershed:
        metrics += [watershed_vector_m, watershed_energy_m]

    model.compile(Adam(lr=1e-4, clipvalue=0.1), loss=loss, metrics=metrics)

    checkpoints_dir = f'../output/checkpoints/{model_name}_{run}{fold}'
    tensorboard_dir = f'../output/tensorboard/wshed/{model_name}_{run}{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print('\n', model_name, '\n')

    initial_epoch = 0
    if len(weights):
        initial_epoch = int(re.search('checkpoint-(...)-', weights).group(1))
        model.load_weights(weights)

    validation_data = data.load_validation()

    if model_info.train_frozen_epochs > 0:
        if len(weights) == 0:
            model.fit_generator(
                data.generate_train(),
                validation_data=validation_data,
                steps_per_epoch=512*8//data.batch_size,
                epochs=model_info.train_frozen_epochs,
                verbose=1,
                initial_epoch=0
            )

        for layer in model.layers:
            layer.trainable = True

        model.compile(Adam(lr=1e-4, clipvalue=0.1), loss=loss,
                      metrics=metrics)

    checkpoint_template = "/checkpoint-{epoch:03d}-{loss:.3f}-{val_loss:.3f}.hdf5"

    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + checkpoint_template, verbose=1, period=8, save_weights_only=True)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=True)

    def cheduler(epoch):
        if epoch < 32:
            return 1e-4
        if epoch < 64:
            return 5e-5
        if epoch < 96:
            return 2e-5
        if epoch < 122:
            return 1e-5
        if epoch < 140:
            return 5e-6
        return 2e-6

    model.fit_generator(
        data.generate_train(),
        steps_per_epoch=512*8//data.batch_size,
        epochs=180,
        verbose=1,
        validation_data=validation_data,
        initial_epoch=initial_epoch,
        callbacks=[
            checkpoint_periodical,
            LearningRateScheduler(cheduler),
            tensorboard
        ]
    )


def predictions_to_center_of_mass(y):
    rows, cols = y.shape[:2]
    res = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            center_row = int(np.clip(row+y[row, col, 1], 0, rows - 1))
            center_col = int(np.clip(col+y[row, col, 2], 0, cols - 1))
            res[center_row, center_col] += y[row, col, 0]
    return res


def predictions_to_shift_from_border(y, distance):
    rows, cols = y.shape[:2]
    res = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            center_row = np.clip(row+y[row, col, 4]*distance, 0, rows - 1.001)
            center_col = np.clip(col+y[row, col, 5]*distance, 0, cols - 1.001)

            row_fractonal, center_row = math.modf(center_row)
            col_fractonal, center_col = math.modf(center_col)
            center_row = int(center_row)
            center_col = int(center_col)

            val = y[row, col, 0]

            res[center_row, center_col] += val * (1.0 - row_fractonal) * (1.0 - col_fractonal)
            res[center_row, center_col + 1] += val * (1.0 - row_fractonal) * col_fractonal
            res[center_row + 1, center_col] += val * row_fractonal * (1.0 - col_fractonal)
            res[center_row + 1, center_col +1] += val * row_fractonal * col_fractonal
    return np.clip(res, 0.0, 1.0)


def round_kernel(r):
    kernel = np.ones((r * 2 + 1, r * 2 + 1))
    for row in range(-r, r + 1):
        for col in range(-r, r + 1):
            if row ** 2 + col ** 2 > r ** 2:
                kernel[col + r, row + r] = 0
    return kernel


def find_cluster_centers(prediction, extra_clusters=False, return_peak_sum=False):
    center_pred = prediction[:, :, 3].copy()
    pad_width = 16
    center_pred = np.pad(center_pred, pad_width=pad_width, mode='constant')

    round_masks = {r:round_kernel(r) for r in range(3, 14)}

    peaks = skimage.feature.peak_local_max(center_pred, min_distance=3, threshold_abs=0.04, indices=True, num_peaks=4000)

    value_with_peak = [(center_pred[peak[0], peak[1]], peak[0], peak[1]) for peak in peaks]
    value_with_peak = sorted(value_with_peak, reverse=True)

    extra_peaks = [(7, 4.75)] if extra_clusters else []
    peak_sums = []
    res = []
    for peak_value, row, col in value_with_peak:
        cur_peak_sums = []
        matched = False
        matched_r = 0
        for r, threshold in [(3, 8.0), (4, 7.5), (5, 6.5), (7, 6.5), (9, 6.5), (11, 6.0), (13, 5.2)] + extra_peaks:
            kernel = round_masks[r]
            # print(np.sum(center_pred[row - r:row + r + 1, col - r:col + r + 1] * kernel), r)
            peak_sum = np.sum(center_pred[row-r:row+r+1, col-r:col+r+1] * kernel)
            cur_peak_sums.append(peak_sum)

            if not matched:
                matched_r = r

            if peak_sum > threshold:
                matched = True
                if not return_peak_sum:
                    break

        if matched:
            res.append([row-pad_width, col-pad_width])
            peak_sums.append(cur_peak_sums)
        # clear found cluster center
        center_pred[row-matched_r:row+matched_r+1, col-matched_r:col+matched_r+1][round_masks[matched_r] > 0] = 0.0

    if return_peak_sum:
        return np.array(res), peak_sums
    else:
        return np.array(res)


def find_masks(prediction, prob_treshold=0.48):
    utils.print_stats('find masks', prediction[:, :, 0])
    cluster_centers = find_cluster_centers(prediction)
    nb_masks = cluster_centers.shape[0]
    rows, cols = prediction.shape[:2]
    masks = np.zeros((nb_masks, rows, cols))

    mask_05 = prediction[:, :, 0] > 0.5
    mask_01 = prediction[:, :, 0] > 0.25
    mask_05_dilated = skimage.morphology.dilation(mask_05, skimage.morphology.disk(4))

    if len(cluster_centers) > 0:
        for row in range(rows):
            for col in range(cols):
                if mask_05_dilated[row, col] and mask_01[row, col]:
                # if prediction[row, col, 0] > prob_treshold:
                    # predicted center of mask
                    pred_mask_center = np.array([
                        row + prediction[row, col, 1],
                        col + prediction[row, col, 2]
                        ])
                    distance_to_masks = np.linalg.norm(cluster_centers - pred_mask_center, axis=1)
                    masks[np.argmin(distance_to_masks), row, col] = 1.0
    # clean masks
    # for i in range(nb_masks):
    #     masks[i] = skimage.morphology.opening(masks[i], skimage.morphology.disk(1))
    return masks


def clip_mask(prediction):
    mask_05 = prediction[:, :, 0] > 0.5
    mask_01 = prediction[:, :, 0] > 0.25
    mask_05_dilated = skimage.morphology.dilation(mask_05, skimage.morphology.disk(4))
    mask = np.logical_and(mask_01, mask_05_dilated)
    return mask


def find_masks_watershed_filtered(predictions):
    mean_prediction = np.mean(predictions, axis=0)
    masks_combined = find_masks_watershed(mean_prediction)

    separate_masks = []
    for i in range(predictions.shape[0]):
        separate_masks.append(find_masks_watershed(predictions[i]))

    masks_median_iou = []
    center_area = []
    mask_mean = []
    for mask in masks_combined:
        tta_iou = []
        mask_center = scipy.ndimage.measurements.center_of_mass(mask)
        for separate_mask in separate_masks:
            iou = 0.0
            for tta_mask in separate_mask:
                if tta_mask[int(mask_center[0]), int(mask_center[1])]:
                    intersection = np.sum(tta_mask * mask)
                    union = np.sum(np.maximum(tta_mask, mask))
                    iou = intersection / union
                    break
            tta_iou.append(iou)
        masks_median_iou.append(np.median(tta_iou))
        center_area.append(np.sum(mean_prediction[:, :, dataset.Y_CENTER][mask > 0.5]))
        mask_mean.append(np.mean(mean_prediction[:, :, dataset.Y_MASK][mask > 0.5]))

    masks_median_iou = np.array(masks_median_iou)
    center_area = np.array(center_area)
    mask_mean = np.array(mask_mean)

    valid_samples = (masks_median_iou > config.THRESHOLD_IOU) * (center_area > config.THRESHOLD_CENTER_AREA) * (mask_mean > config.THRESHOLD_MASK_MEAN)
    invalid_samples = np.sum(valid_samples == False)
    if invalid_samples:
        total_samples = masks_combined.shape[0]
        dropped_from_iou = np.sum(masks_median_iou < config.THRESHOLD_IOU)
        dropped_from_center_area = np.sum(center_area < config.THRESHOLD_CENTER_AREA)
        dropped_from_mask_mean = np.sum(mask_mean < config.THRESHOLD_MASK_MEAN)
        print(f'dropped {invalid_samples} samples from {total_samples}, iou: {dropped_from_iou}, center area: {dropped_from_center_area}, mask mean: {dropped_from_mask_mean}')
        print('dropped iou', masks_median_iou[masks_median_iou < config.THRESHOLD_IOU])
        print('dropped center area', center_area[center_area < config.THRESHOLD_CENTER_AREA])
        print('dropped mask mean', mask_mean[mask_mean < config.THRESHOLD_MASK_MEAN])

    return masks_combined[valid_samples]


def find_masks_watershed(prediction):
    utils.print_stats('find masks', prediction[:, :, 0])
    mask = clip_mask(prediction)
    cluster_centers = find_cluster_centers(prediction)
    return find_masks_watershed_from_centers(prediction, mask, cluster_centers)


def split_center(prediction, instance_mask):
    center_masked = prediction[:, :, dataset.Y_CENTER].copy()
    center_masked[instance_mask == False] = 0
    max_value = np.max(center_masked)

    levels = 64

    points = []
    for pos in np.transpose(np.nonzero(center_masked > max_value/levels)):
        points += [pos] * int(center_masked[pos[0], pos[1]] * levels / max_value)

    kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(points)
    centers = kmeans.cluster_centers_
    pos1, pos2 = centers[0], centers[1]

    return pos1, pos2


def find_masks_watershed_from_centers(prediction, mask, cluster_centers):
    energy = prediction[:, :, dataset.Y_WATERSHED_ENERGY_1] + \
             prediction[:, :, dataset.Y_WATERSHED_ENERGY_2] + \
             prediction[:, :, dataset.Y_WATERSHED_ENERGY_3] + \
             prediction[:, :, dataset.Y_MASK] + \
             prediction[:, :, dataset.Y_CENTER]

    energy_orig = energy.copy()
    markers = np.zeros_like(energy_orig, dtype=np.int32)
    for i, cluster_center in enumerate(cluster_centers):
        markers[cluster_center[0], cluster_center[1]] += i+1

    # markers = scipy.ndimage.label(energy)[0]

    # find missed centers
    if config.CHECK_MISSING_CENTERS_FROM_WSHED:
        mask_strict = prediction[:, :, dataset.Y_WATERSHED_ENERGY_2] > 0.5

        labels = skimage.morphology.watershed(-energy_orig, markers, mask=mask_strict)
        watershed_level3 = prediction[:, :, dataset.Y_WATERSHED_ENERGY_2] > 0.55
        watershed_level3_missed = watershed_level3 * (labels == 0)
        watershed_level3_missed_labels, watershed_level3_missed_labels_num = skimage.measure.label(watershed_level3_missed,
                                                                                                   connectivity=1,
                                                                                                   return_num=True)
        added_centers = 0
        for label_id in range(watershed_level3_missed_labels_num):
            missed_mask = watershed_level3_missed_labels == label_id+1
            if np.sum(missed_mask) > 24:
                center_of_mass = scipy.ndimage.measurements.center_of_mass(missed_mask)
                added_centers += 1
                markers[int(center_of_mass[0]), int(center_of_mass[1])] += len(cluster_centers) + added_centers

        if added_centers:
            print(f'added extra {added_centers} labels')

    # now re-calculate labels from updated markers
    labels = skimage.morphology.watershed(-energy_orig, markers, mask=mask)

    # check if the total sum of center prediction exceeds expected*1.5, try to split the relevant mask
    nb_masks = np.max(labels)
    centers_added = 0
    for mask_id in range(nb_masks):
        cur_mask = labels == mask_id + 1
        center_sum = np.sum(prediction[cur_mask, dataset.Y_CENTER])
        if center_sum > 15.0:
            centers_added += 1
            pos1, pos2 = split_center(prediction, cur_mask)
            existing_marker = np.max(markers[cur_mask])
            markers[markers == existing_marker] = 0

            nb_masks += 1
            markers[int(pos1[0]), int(pos1[1])] = existing_marker
            markers[int(pos2[0]), int(pos2[1])] = nb_masks

    if centers_added:
        print(f'added {centers_added} centers')
        labels = skimage.morphology.watershed(-energy_orig, markers, mask=mask)

    # fig, axis = plt.subplots(2, 2)
    # axis[0, 0].imshow(energy)
    # axis[0, 1].imshow(energy_orig)
    # axis[1, 0].imshow(markers)
    # axis[1, 1].imshow(labels)
    # plt.figure()

    used_labels = np.unique(labels)[1:]  # skip bg

    nb_masks = len(used_labels)
    rows, cols = prediction.shape[:2]
    masks = np.zeros((nb_masks, rows, cols), dtype=np.bool)

    for i, l in enumerate(used_labels):
        masks[i] = labels == l
    return masks


def check_score(gt_masks, pred_masks):
    return metrics.score(gt_masks, pred_masks, verbose=1)


def check(model_name, fold, weights, sample_id):
    model_info = MODELS[model_name]
    use_watershed = model_info.dataset_args['output_watershed']
    data = dataset.UVectorNetDataset(fold=fold, batch_size=1,  **model_info.dataset_args)

    model = model_info.factory(**model_info.args)
    model.load_weights(weights)
    loss = combined_loss_watershed if use_watershed else combined_loss
    model.compile(Adam(lr=1e-4), loss=loss, metrics=[bce_m, dice_m, vector_m])

    samples = data.validation_sample_ids
    if len(sample_id):
        samples = sample_id.split(',') + samples

    for sample_id in samples:
        print(sample_id)
        img = data.images[sample_id]

        cfg = dataset.SampleCfg(
            sample_id=sample_id,
            src_center_x=img.shape[1]//2,
            src_center_y=img.shape[0]//2
        )

        X = data.generate_x(cfg)
        y = data.generate_y(cfg)

        utils.print_stats('preprocessed crop', X)

        pred = model.predict(np.array([X]))[0]

        max_vector = np.max(np.abs(y)) + 1e-3

        plt.subplot(3, 5, 1)
        plt.imshow((X+1)/2)

        plt.subplot(3, 5, 2)
        plt.imshow(y[:, :, 0])
        plt.subplot(3, 5, 3)
        plt.imshow(y[:, :, 1] / max_vector)
        plt.subplot(3, 5, 4)
        plt.imshow(y[:, :, 2] / max_vector)

        plt.subplot(3, 5, 5)
        # plt.imshow(y[:, :, 3])

        mask_sum = scipy.signal.convolve2d(pred[:, :, 3], round_kernel(7), mode='same')
        plt.imshow(mask_sum)

        plt.subplot(3, 5, 6)
        center_of_mass = predictions_to_center_of_mass(pred)
        # plt.imshow(center_of_mass/np.max(center_of_mass))
        plt.imshow(np.clip(center_of_mass, 0, 10.0)/10.0)

        plt.subplot(3, 5, 7)
        plt.imshow(pred[:, :, 0])
        plt.subplot(3, 5, 8)
        plt.imshow(pred[:, :, 1] / max_vector)
        plt.subplot(3, 5, 9)
        plt.imshow(pred[:, :, 2] / max_vector)

        plt.subplot(3, 5, 10)
        plt.imshow(pred[:, :, 3])

        if use_watershed:
            plt.subplot(3, 5, 11)
            plt.imshow(pred[:, :, 4])

            plt.subplot(3, 5, 12)
            plt.imshow(pred[:, :, 5])

            plt.subplot(3, 5, 13)
            plt.imshow(pred[:, :, 6])

            plt.subplot(3, 5, 14)
            plt.imshow(pred[:, :, 7])
            # plt.imshow(predictions_to_shift_from_border(pred, distance=2.0))

            plt.subplot(3, 5, 15)
            plt.imshow(pred[:, :, 8])
            # plt.imshow(predictions_to_shift_from_border(pred, distance=3.5))

        masks = find_masks(pred)
        plt.show()


def check_tta(model_name, fold, weights, sample_id):
    use_watershed = True
    data = dataset.UVectorNetDataset(fold=fold, batch_size=1, output_watershed=use_watershed)

    model_info = MODELS[model_name]

    if sample_id == '':
        sample_id = '02903040e19ddf92f452907644ad3822918f54af41dd85e5a3fe3e1b6d6f9339'

    if len(sample_id):
        samples = sample_id.split(',')

    for sample_id in samples:
        print(sample_id)
        img = data.images[sample_id]
        img = img * 0.9 / np.percentile(img, 99.75)

        tta_versions = []
        for transpose in [False, True]:
            for h_flip in [False, True]:
                for v_flip in [False, True]:
                    tta_versions.append(tta.TTA(h_flip=h_flip, v_flip=v_flip, transpose=transpose))

        cfg = dataset.SampleCfg(
            sample_id=sample_id,
            src_center_x=img.shape[1]//2,
            src_center_y=img.shape[0]//2
        )

        X_orig = data.generate_x(cfg, img)

        fig, axes = plt.subplots(3, 3)
        X = [cur_tta.process_image(X_orig) for cur_tta in tta_versions]
        axes[2, 2].imshow(X_orig/2+0.5)
        for i, cur_X in enumerate(X):
            axes[i%3, i//3].imshow(cur_X/2+0.5)
            axes[i % 3, i // 3].set_title(str(tta_versions[i]))

        # plt.show()

        try:
            pred = np.load(f'../cache/tta_check{sample_id}.npy')
        except FileNotFoundError:
            model = model_info.factory(**model_info.args)
            model.load_weights(weights)
            loss = combined_loss_watershed if use_watershed else combined_loss
            model.compile(Adam(lr=1e-4), loss=loss, metrics=[bce_m, dice_m, vector_m])
            pred = model.predict(np.array(X))
            np.save(f'../cache/tta_check{sample_id}.npy', pred)

        for prediction_channel in range(pred.shape[-1]):
            fig, axes = plt.subplots(3, 6)
            for i, cur_tta in enumerate(tta_versions):
                axes[i % 3, i // 3].imshow(pred[i, :, :, prediction_channel])
                axes[i % 3, i // 3].set_title(str(cur_tta))
                converted_pred = cur_tta.process_prediction(pred[i])
                axes[i % 3, i // 3 + 3].imshow(converted_pred[:, :, prediction_channel])
                axes[i % 3, i // 3 + 3].set_title(str(cur_tta))

            # plt.show()

        plt.show()


def check_full(model_name, fold, weights, sample_id, run, checkpoint):
    model_info = MODELS[model_name]
    data = dataset.UVectorNetDataset(fold=fold, preprocess_func=model_info.preprocess_input, **model_info.dataset_args)

    if checkpoint > 0:
        run = f'ch{checkpoint}'

    if len(weights):
        model_info = MODELS[model_name]
        model = model_info.factory(input_shape=(512, 512, 3), **model_info.args)
        model.load_weights(weights)
        model.compile(Adam(lr=1e-4), loss=combined_loss, metrics=[bce_m, dice_m, vector_m])

    samples = data.validation_sample_ids

    if run != '':
        df = pd.read_csv(f'../output/predict_train/{model_name}_{run}{fold}.csv')
        samples = np.array(df.sample_id)[np.argsort(df.score)]

    print(sample_id)
    if len(sample_id):
        samples = sample_id.split(',') + list(samples)

    for sample_id in samples:
        print(sample_id)
        img = data.images[sample_id]
        gt_masks = skimage.io.imread_collection("{}/{}/masks/*.png".format(config.TRAIN_DIR, sample_id)).concatenate()

        if len(weights):
            def preprocess_input(crop):
                res = data.preprocess(crop)
                utils.print_stats('preprocessed crop', res)
                return res

            pred = utils.combine_tiled_predictions(model,
                                                   img=img,
                                                   preprocess_input=preprocess_input,
                                                   crop_size=512,
                                                   channels=data.y_depth(),
                                                   overlap=128)
        else:
            output_dir = f'../output/predict_train/{model_name}_{run}{fold}'
            pred = np.load(f'{output_dir}/{sample_id}.npy')

        # masks = find_masks(pred)
        cluster_centers = find_cluster_centers(pred)
        masks = find_masks_watershed(pred)
        check_score(gt_masks=gt_masks, pred_masks=masks)

        img_light = np.power(img/255.0, 0.5)*255.0

        fig, ax = plt.subplots(3, 4)

        ax[0, 0].imshow(img)
        ax[0, 1].imshow(img_light/255.0)

        ax[0, 2].set_title('gt masks')
        ax[0, 2].imshow(visualisation.apply_masks(img_light, gt_masks/255.0))

        ax[0, 3].set_title('predicted masks')
        ax[0, 3].imshow(visualisation.apply_masks(img_light, masks))

        ax[1, 0].set_title('selected cluster centers')
        ax[1, 0].imshow(img)
        ax[1, 0].scatter(cluster_centers[:, 1], cluster_centers[:, 0], s=5, c='black')
        ax[1, 0].scatter(cluster_centers[:, 1], cluster_centers[:, 0], s=1, c='white')

        ax[1, 1].set_title('mask')
        ax[1, 1].imshow(pred[..., 0])

        ax[1, 2].set_title('mask clip')
        ax[1, 2].imshow(clip_mask(pred))

        ax[1, 3].set_title('predicted centers of nuclei')
        ax[1, 3].imshow(pred[:, :, dataset.Y_CENTER])

        # ax[2, 0].set_title('mask moved by to_center vector')
        # center_of_mass = predictions_to_center_of_mass(pred)
        # ax[2, 0].imshow(np.clip(center_of_mass, 0, 10.0) / 10.0)
        ax[2, 0].set_title('prepared input')

        utils.print_stats('img', img)
        preprocessed = data.preprocess(img)
        utils.print_stats('preprocessed', preprocessed)
        ax[2, 0].imshow(preprocessed[..., 0])

        ax[2, 1].imshow(scipy.signal.convolve2d(pred[:, :, dataset.Y_CENTER], round_kernel(3), mode='same'))
        ax[2, 1].set_title('centers sum r=3')

        ax[2, 2].imshow(scipy.signal.convolve2d(pred[:, :, dataset.Y_CENTER], round_kernel(5), mode='same'))
        ax[2, 2].set_title('centers sum r=5')

        ax[2, 3].imshow(pred[:, :, dataset.Y_WATERSHED_ENERGY_1])
        ax[2, 3].set_title('watershed level 1')

        plt.show()


def check_extra():
    img = PIL.Image.open('../input/extra_data_check/set24/0013/images/0013.png')
    img = np.array(img)[:, :, :3]

    pred = np.load('../output/predict_extra_data/0013.npy')
    print(pred.shape)

    # masks = find_masks(pred)
    cluster_centers = find_cluster_centers(pred)
    masks = find_masks_watershed(pred)

    img_light = np.power(img/255.0, 0.5)*255.0

    fig, ax = plt.subplots(3, 4)

    ax[0, 0].imshow(img)
    ax[0, 1].imshow(img_light/255.0)

    # ax[0, 2].set_title('gt masks')
    # ax[0, 2].imshow(visualisation.apply_masks(img_light, gt_masks/255.0))

    # ax[0, 3].set_title('predicted masks')
    # ax[0, 3].imshow(visualisation.apply_masks(img_light, masks.astype(np.float32)))

    ax[0, 3].set_title('predicted masks')
    ax[0, 3].imshow(np.sum(masks, axis=-1))

    ax[1, 0].set_title('selected cluster centers')
    ax[1, 0].imshow(img)
    ax[1, 0].scatter(cluster_centers[:, 1], cluster_centers[:, 0], s=5, c='black')
    ax[1, 0].scatter(cluster_centers[:, 1], cluster_centers[:, 0], s=1, c='white')

    ax[1, 1].set_title('mask')
    ax[1, 1].imshow(pred[..., 0])

    ax[1, 2].set_title('mask clip')
    ax[1, 2].imshow(clip_mask(pred))

    ax[1, 3].set_title('predicted centers of nuclei')
    ax[1, 3].imshow(pred[:, :, dataset.Y_CENTER])

    # ax[2, 0].set_title('mask moved by to_center vector')
    # center_of_mass = predictions_to_center_of_mass(pred)
    # ax[2, 0].imshow(np.clip(center_of_mass, 0, 10.0) / 10.0)
    ax[2, 0].set_title('prepared input')

    # utils.print_stats('img', img)
    # preprocessed = data.preprocess(img)
    # utils.print_stats('preprocessed', preprocessed)
    # ax[2, 0].imshow(preprocessed[..., 0])

    ax[2, 1].imshow(scipy.signal.convolve2d(pred[:, :, dataset.Y_CENTER], round_kernel(3), mode='same'))
    ax[2, 1].set_title('centers sum r=3')

    ax[2, 2].imshow(scipy.signal.convolve2d(pred[:, :, dataset.Y_CENTER], round_kernel(5), mode='same'))
    ax[2, 2].set_title('centers sum r=5')

    ax[2, 3].imshow(pred[:, :, dataset.Y_WATERSHED_ENERGY_1])
    ax[2, 3].set_title('watershed level 1')

    ax[2, 0].imshow(pred[:, :, dataset.Y_WATERSHED_ENERGY_2])
    ax[2, 0].set_title('watershed level 2')

    plt.show()


def weights_for_model_checkpoint(model_name, fold, checkpoint):
    return glob.glob(f'../output/checkpoints/{model_name}_{fold}/checkpoint-{checkpoint:03}*')[0]


def predict_train(model_name, fold, weights, run, use_tta, checkpoint, scale_img=1.0, only_samples=None):
    model_info = MODELS[model_name]
    data = dataset.UVectorNetDataset(fold=fold,
                                     preprocess_func=model_info.preprocess_input,
                                     ** model_info.dataset_args)

    crop_size = 768
    overlap = 128

    if checkpoint > 0:
        weights = weights_for_model_checkpoint(model_name, fold, checkpoint)
        print('load '+weights)
        run = f'ch{checkpoint}'

    model_info = MODELS[model_name]
    model = model_info.factory(input_shape=(crop_size, crop_size, 3), **model_info.args)
    model.load_weights(weights)
    model.compile(Adam(lr=1e-4), loss=combined_loss, metrics=[bce_m, dice_m, vector_m])

    samples = data.validation_sample_ids

    output_dir = f'../output/predict_train/{model_name}_{run}{fold}'
    os.makedirs(output_dir, exist_ok=True)

    # output_dir_scaled = f'../output/predict_train/{model_name}_{run}_scaled_{fold}'
    # os.makedirs(output_dir_scaled, exist_ok=True)

    for sample_id in tqdm(samples):
        # print(sample_id)
        img = data.images[sample_id]
        # utils.print_stats('img', img)

        # img = img * 0.9 / np.max(img)
        img = img * 0.9 / np.percentile(img, 99.75)
        # utils.print_stats('img', img)

        def preprocess_input(crop):
            # utils.print_stats('input crop', crop)
            crop = data.preprocess(crop*255)
            # utils.print_stats('preprocessed crop', crop)
            return crop
            # crop = crop.astype(np.float32) - data.img_median[sample_id]
            # return crop * data.img_value_scale[sample_id]

        # use_tta = True

        if use_tta:
            predictions = []
            for h_flip in [False, True]:
                for v_flip in [False, True]:
                    for transpose in [False, True]:
                        cur_tta = tta.TTA(h_flip=h_flip, v_flip=v_flip, transpose=transpose)
                        img2 = cur_tta.process_image(img)

                        pred = utils.combine_tiled_predictions(model,
                                                               img=img2,
                                                               # preprocess_input=model_info.preprocess_input,
                                                               preprocess_input=preprocess_input,
                                                               crop_size=crop_size,
                                                               channels=data.y_depth(),
                                                               overlap=overlap)
                        pred = cur_tta.process_prediction(pred)
                        predictions.append(pred)
            # pred0 = predictions[0]
            np.save(f'{output_dir}/tta_{sample_id}.npy', np.array(predictions))
            pred = np.mean(np.array(predictions), axis=0)
            # pred[:, :, 1:] = pred0[:, :, 1:]
        else:
            pred = utils.combine_tiled_predictions(model,
                                                   img=img,
                                                   # preprocess_input=model_info.preprocess_input,
                                                   preprocess_input=preprocess_input,
                                                   crop_size=crop_size,
                                                   channels=data.y_depth(),
                                                   overlap=overlap)
        np.save(f'{output_dir}/{sample_id}.npy', pred)
        np.save(f'{output_dir}/mask_{sample_id}.npy', pred[..., 0].astype(np.float32))


def calc_score(output_dirs, sample_id, filter_masks):
    print(sample_id)
    gt_masks = skimage.io.imread_collection("{}/{}/masks/*.png".format(config.TRAIN_DIR, sample_id)).concatenate()

    if filter_masks:
        preds = np.concatenate([np.load(f'{output_dir}/tta_{sample_id}.npy') for output_dir in output_dirs], axis=0)
        pred_masks = find_masks_watershed_filtered(preds)
    else:
        preds = [np.load(f'{output_dir}/{sample_id}.npy') for output_dir in output_dirs]
        pred = np.mean(preds, axis=0)
        pred_masks = find_masks_watershed(pred)

    return metrics.score(masks=gt_masks, pred=pred_masks, verbose=1)


def check_score_train(model_names, fold, runs, checkpoint, filter_masks):
    data = dataset.UVectorNetDataset(fold=fold, preprocess_func=None)

    if checkpoint > 0:
        runs = f'ch{checkpoint}'

    output_dirs = [f'../output/predict_train/{model_name}_{run}{fold}'
                   for model_name, run in zip(model_names.split(','), runs.split(','))]
    samples = data.validation_sample_ids
    pool = Pool(16)
    scores = pool.starmap(calc_score, [(output_dirs, sample_id, filter_masks) for sample_id in samples])

    # scores = []
    # for sample_id in tqdm(samples):
    #     score = calc_score(output_dirs, sample_id, filter_masks)
    #     scores.append(score)
    print(np.mean(scores))
    df = pd.DataFrame(dict(sample_id=samples, score=scores))
    df.to_csv(f'../output/predict_train/{model_names}_{runs}{fold}.csv', index=False)

    # plt.hist(scores, bins=16)
    # plt.show()


def check_test(model_name, weights):
    model_info = MODELS[model_name]
    model = model_info.factory(input_shape=(1024, 1024, 3), **model_info.args)
    model.load_weights(weights)
    model.compile(Adam(lr=1e-4), loss=combined_loss, metrics=[bce_m, dice_m, vector_m])

    sample_submission = pd.read_csv('../input/stage1_sample_submission.csv')
    samples = list(sample_submission.ImageId)

    for sample_id in samples:
        print(sample_id)
        with PIL.Image.open(f'../input/stage1_test/{sample_id}/images/{sample_id}.png') as img:
            orig_img = np.array(img)[:, :, :3]
            for i, img in enumerate([orig_img, imresize(orig_img, 2.0), np.power(orig_img/255.0, 0.5)*255.0]):
                pred = utils.combine_tiled_predictions(model,
                                                       img=img,
                                                       preprocess_input=model_info.preprocess_input,
                                                       crop_size=1024,
                                                       channels=4,
                                                       overlap=256)
                masks = find_masks(pred)

                utils.print_stats('img', img)

                plt.subplot(2, 4, 1)
                plt.imshow(img/255.0)

                plt.subplot(2, 4, 3)
                plt.imshow(metrics.combine_masks(masks))

                plt.subplot(2, 4, 4)
                plt.imshow(np.power(img/255.0, 0.5))
                plt.imshow(metrics.combine_masks(masks), alpha=0.25)

                plt.subplot(2, 4, 5)
                plt.imshow(pred[:, :, 0])

                plt.subplot(2, 4, 6)
                plt.imshow(pred[:, :, 3])

                if i < 2:
                    plt.figure()

            plt.show()


def save_prediction_quant(fn, predictions):
    fields_to_keep = [dataset.Y_MASK,
                      dataset.Y_CENTER,
                      dataset.Y_WATERSHED_ENERGY_1, dataset.Y_WATERSHED_ENERGY_2, dataset.Y_WATERSHED_ENERGY_3]
    res_shape = list(predictions.shape)
    res_shape[-1] = len(fields_to_keep)

    pred_int = np.zeros(res_shape, dtype=np.uint16)
    for i, field in enumerate(fields_to_keep):
        pred_int[..., i] = np.clip(predictions[..., field]*65535, 0, 65535).astype(np.uint16)

    # pred_int = np.zeros(res_shape, dtype=np.float32)
    # for i, field in enumerate(fields_to_keep):
    #     pred_int[..., i] = predictions[..., field]

    np.save(fn, pred_int)


def load_prediction_quant(fn):
    fields_to_keep = [dataset.Y_MASK,
                      dataset.Y_CENTER,
                      dataset.Y_WATERSHED_ENERGY_1, dataset.Y_WATERSHED_ENERGY_2, dataset.Y_WATERSHED_ENERGY_3]

    data_int = np.load(fn).astype(np.float32)
    res_shape = list(data_int.shape)
    res_shape[-1] = dataset.Y_MASK_AREA+1

    res = np.zeros(res_shape, dtype=np.float32)

    for i, field in enumerate(fields_to_keep):
        res[..., field] = data_int[..., i] / 65535.0

    return res


def prepare_submission(model_name, weights, submission_name, use_tta):
    MAX_CROPS_SIZE = 1024

    model_info = MODELS[model_name]
    model = model_info.factory(input_shape=(None, None, 3), **model_info.args)
    model.load_weights(weights)
    model.compile(Adam(lr=1e-4), loss=combined_loss, metrics=[bce_m, dice_m, vector_m])

    output_dir = f'../output/predict_test/{submission_name}'
    os.makedirs(output_dir, exist_ok=True)

    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION)
    samples = list(sample_submission.ImageId)

    submission_file = open(f'../submissions/{submission_name}.csv', 'w+')
    submission_file.write('ImageId,EncodedPixels\n')

    data = dataset.UVectorNetDataset(fold=-1, preprocess_func=model_info.preprocess_input, **model_info.dataset_args)

    for sample_id in tqdm(samples):
        print(sample_id)
        with PIL.Image.open(f'{config.SUBMISSION_DIR}/{sample_id}/images/{sample_id}.png') as img:
            orig_img = np.array(img)
            if len(orig_img.shape) == 2:
                orig_img = np.stack([orig_img, orig_img, orig_img], axis=2)
                print(orig_img.shape)

            orig_img = orig_img[:, :, :3]
            if config.IMG_SCALE > 1:
                scaled_img = imresize(orig_img, float(config.IMG_SCALE), interp='lanczos')
            else:
                scaled_img = orig_img
            scaled_img = scaled_img * 0.9 / np.percentile(scaled_img, 99.75)

            # utils.print_stats('img', img)

            def preprocess_input(crop):
                # utils.print_stats('input crop', crop)
                crop = data.preprocess(crop * 255)
                # utils.print_stats('preprocessed crop', crop)
                return crop
            overlap = 256
            crop_size = overlap + overlap // 2 + max(scaled_img.shape[0], scaled_img.shape[1])
            pad = 128
            crop_size = (crop_size + pad - 1) // pad * pad
            crop_size = min(MAX_CROPS_SIZE, crop_size)
            print('crop size', crop_size)

            try:
                pred = np.load(f'{output_dir}/{sample_id}.npy')
                print('skip '+sample_id)
            except FileNotFoundError:
                if use_tta:
                    predictions = []
                    for h_flip in [False, True]:
                        for v_flip in [False, True]:
                            for transpose in [False, True]:
                                cur_tta = tta.TTA(h_flip=h_flip, v_flip=v_flip, transpose=transpose)
                                img2 = cur_tta.process_image(scaled_img)
                                pred = utils.combine_tiled_predictions(model,
                                                                       img=img2,
                                                                       preprocess_input=preprocess_input,
                                                                       crop_size=crop_size,
                                                                       channels=data.y_depth(),
                                                                       overlap=overlap)
                                pred = cur_tta.process_prediction(pred)
                                predictions.append(pred)
                    # pred0 = predictions[0]
                    # np.save(f'{output_dir}/tta_{sample_id}.npy', np.array(predictions))
                    save_prediction_quant(f'{output_dir}/tta_{sample_id}_q.npy', np.array(predictions))
                    pred = np.mean(np.array(predictions), axis=0)
                else:
                    pred = utils.combine_tiled_predictions(model,
                                                           img=scaled_img,
                                                           preprocess_input=preprocess_input,
                                                           crop_size=crop_size,
                                                           channels=data.y_depth(),
                                                           overlap=overlap)
                np.save(f'{output_dir}/{sample_id}.npy', pred)
                # np.save(f'{output_dir}/mask_{sample_id}.npy', pred[..., 0].astype(np.float32))
            nb_masks = 0
            masks = find_masks_watershed(pred)
            existing_masks = np.zeros(orig_img.shape[:2], dtype=np.uint8)
            for mask in masks:
                if config.IMG_SCALE > 1:
                    mask = imresize(mask, orig_img.shape[:2], interp='nearest') / 255
                if np.sum(mask) > 24:  # skip very small dots
                    # mask = scipy.ndimage.morphology.binary_fill_holes(mask)
                    # mask[existing_masks] = 0
                    # existing_masks[mask] = 1
                    enc = rle.rle_encode(mask)
                    submission_file.write('{},{}\n'.format(sample_id, ' '.join(np.array(enc).astype(str))))
                    nb_masks += 1
            if nb_masks == 0:
                submission_file.write('{},1 1\n'.format(sample_id))


def combine_submissions_sample(sample_id, input_dirs, output_dir, filter_masks):
    print(sample_id)

    try:
        with open(f'{output_dir}/{sample_id}.txt', 'r') as sub_file:
            print('skip', sample_id)
            return sub_file.read()
    except FileNotFoundError:
        pass

    try:
        predictions = [load_prediction_quant(f'{d}/tta_{sample_id}_q.npy') for d in input_dirs]
    except FileNotFoundError:
        predictions = [np.load(f'{d}/tta_{sample_id}.npy') for d in input_dirs]

    predictions = np.concatenate(predictions, axis=0)
    mean_predictions = np.mean(predictions, axis=0)
    np.save(f'{output_dir}/{sample_id}.npy', mean_predictions)

    if filter_masks:
        masks = find_masks_watershed_filtered(predictions)
    else:
        masks = find_masks_watershed(mean_predictions)
    res = ''

    nb_masks = 0
    for mask in masks:
        if np.sum(mask) > 24:  # skip very small dots
            enc = rle.rle_encode(mask)
            res += '{},{}\n'.format(sample_id, ' '.join(np.array(enc).astype(str)))
            nb_masks += 1
    if nb_masks == 0:
        res = '{},1 1\n'.format(sample_id)

    with open(f'{output_dir}/{sample_id}.txt', 'w+') as sub_file:
        sub_file.write(res)

    return res


def combine_submissions(submission_name, submissions_to_combine, filter_masks):
    input_dirs = [f'../output/predict_test/{submission_to_combine}'
                  for submission_to_combine in submissions_to_combine.split(',')]

    output_dir = f'../output/predict_test/{submission_name}'
    os.makedirs(output_dir, exist_ok=True)

    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION)
    samples = list(sample_submission.ImageId)

    sample_with_sizes = []
    for sample_id in samples:
        statinfo = os.stat(f'{input_dirs[0]}/tta_{sample_id}_q.npy')
        sample_with_sizes.append((statinfo.st_size, sample_id))

    sample_with_sizes = sorted(sample_with_sizes, reverse=True)
    process_sequentially = 32  # process sequentially the largest files to avoid running out of RAM
    large_samples = sample_with_sizes[:process_sequentially]
    small_samples = sample_with_sizes[process_sequentially:]
    random.shuffle(small_samples)

    for _, sample_id in large_samples:
        combine_submissions_sample(sample_id, input_dirs, output_dir, filter_masks)

    submission_file = open(f'../submissions/{submission_name}.csv', 'w+')
    submission_file.write('ImageId,EncodedPixels\n')

    pool = Pool(8)
    results = pool.starmap(combine_submissions_sample,
                           [(sample_id, input_dirs, output_dir, filter_masks) for _, sample_id in small_samples])

    for sample_id in samples:
        with open(f'{output_dir}/{sample_id}.txt', 'r') as sub_file:
            submission_file.write(sub_file.read())


def check_combine_submissions(submissions_to_combine, filter_masks, single_sample_id):
    input_dirs = [f'../output/predict_test/{submission_to_combine}'
                  for submission_to_combine in submissions_to_combine.split(',')]

    if single_sample_id == '':
        sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION)
        samples = list(sample_submission.ImageId)
    else:
        samples = [single_sample_id]

    for sample_id in tqdm(samples):
        print(sample_id)
        predictions = [np.load(f'{d}/tta_{sample_id}.npy') for d in input_dirs]
        predictions = np.concatenate(predictions, axis=0)
        mean_predictions = np.mean(predictions, axis=0)

        if filter_masks:
            masks = find_masks_watershed_filtered(predictions)
        else:
            masks = find_masks_watershed(mean_predictions)



def predict_extra_data(model_name, weights, data_dir, run, use_tta):
    model_info = MODELS[model_name]
    model = model_info.factory(input_shape=(1024, 1024, 3), **model_info.args)
    model.load_weights(weights)
    model.compile(Adam(lr=1e-4), loss=combined_loss, metrics=[bce_m, dice_m, vector_m])

    data = dataset.UVectorNetDataset(fold=-1,
                                     preprocess_func=model_info.preprocess_input,
                                     **model_info.dataset_args)

    output_dir = f'../output/predict_extra_data/{run}'
    os.makedirs(output_dir, exist_ok=True)

    for sample_id in tqdm(sorted(os.listdir(data_dir))):
        print(sample_id)

        # if os.path.exists(f'{output_dir}/{sample_id}.npy'):
        #     continue
        # try:
        #     pred = np.load(f'{output_dir}/{sample_id}.npy')
        # except FileNotFoundError:
        if 1:
            with PIL.Image.open(f'{data_dir}/{sample_id}/images/{sample_id}.png') as img:
                # orig_img = np.array(img)[:, :, :3]
                orig_img = np.array(img)
                utils.print_stats('orig img', orig_img)
                if len(orig_img.shape) == 2:
                    orig_img = np.repeat(orig_img[:, :, np.newaxis], 3, axis=-1)
                else:
                    orig_img = orig_img[:, :, :3]

                if config.IMG_SCALE > 1:
                    scaled_img = imresize(orig_img, float(config.IMG_SCALE), interp='lanczos')
                else:
                    scaled_img = orig_img
                scaled_img = scaled_img * 0.9 / np.percentile(scaled_img, 99.75)

                # utils.print_stats('img', img)

                def preprocess_input(crop):
                    # utils.print_stats('input crop', crop)
                    crop = data.preprocess(crop * 255)
                    # utils.print_stats('preprocessed crop', crop)
                    return crop

                if use_tta:
                    predictions = []
                    for h_flip in [False, True]:
                        for v_flip in [False, True]:
                            for transpose in [False, True]:
                                cur_tta = tta.TTA(h_flip=h_flip, v_flip=v_flip, transpose=transpose)
                                img2 = cur_tta.process_image(scaled_img)
                                pred = utils.combine_tiled_predictions(model,
                                                                       img=img2,
                                                                       preprocess_input=preprocess_input,
                                                                       crop_size=1024,
                                                                       channels=data.y_depth(),
                                                                       overlap=256)
                                pred = cur_tta.process_prediction(pred)
                                predictions.append(pred)
                    # pred0 = predictions[0]
                    pred = np.mean(np.array(predictions), axis=0)
                    masks = find_masks_watershed_filtered(np.array(predictions))
                else:
                    pred = utils.combine_tiled_predictions(model,
                                                           img=scaled_img,
                                                           preprocess_input=preprocess_input,
                                                           crop_size=1024,
                                                           channels=data.y_depth(),
                                                           overlap=256)
                    masks = find_masks_watershed(pred)
                np.save(f'{output_dir}/{sample_id}.npy', pred)

        mask_idx = 0
        os.makedirs(f'{data_dir}/{sample_id}/masks/', exist_ok=True)
        for mask in masks:
            if config.IMG_SCALE > 1:
                mask = imresize(mask, orig_img.shape[:2], interp='nearest') / 255
            if np.sum(mask) > 16:  # skip very small dots
                mask_idx += 1
                skimage.io.imsave(f'{data_dir}/{sample_id}/masks/{mask_idx:03}.png', mask.astype(np.uint8))


def predict_extra_data_from_runs(runs, data_dir):
    for sample_id in tqdm(sorted(os.listdir(data_dir))):
        print(sample_id)

        predictions = []
        for run in runs.split(','):
            predictions.append(np.load(f'../output/predict_extra_data/{run}/{sample_id}.npy'))

        pred = np.mean(np.array(predictions), axis=0)
        masks = find_masks_watershed(pred)
        mask_idx = 0
        os.makedirs(f'{data_dir}/{sample_id}/masks/', exist_ok=True)
        for mask in masks:
            if np.sum(mask) > 16:  # skip very small dots
                mask_idx += 1
                skimage.io.imsave(f'{data_dir}/{sample_id}/masks/{mask_idx:03}.png', mask.astype(np.uint8))


def check_submission(submission_name):
    df = pd.read_csv(f'../submissions/{submission_name}.csv')
    try:
        df_dropped = pd.read_csv(f'../submissions/{submission_name}_dropped.csv')
    except FileNotFoundError:
        df_dropped = None
    sample_ids = sorted(list(df['ImageId'].unique()))
    output_dir = f'../output/predict_test/{submission_name}'

    # sample_ids = ['0ed3555a4bd48046d3b63d8baf03a5aa97e523aa483aaa07459e7afa39fb96c6']
    # sample_ids = ['505bc0a3928d8aef5ce441c5a611fdd32e1e8eccdc15cc3a52b88030acb50f81']

    for sample_id in sample_ids:
        print(sample_id)
        pred = np.load(f'{output_dir}/{sample_id}.npy')
        with PIL.Image.open(f'{config.SUBMISSION_DIR}/{sample_id}/images/{sample_id}.png') as img:
            orig_img = np.array(img)[:, :, :3]
            if len(orig_img.shape) > 2:
                orig_img = orig_img[:, :, :3]
            light_img = np.power(orig_img / 255.0, 0.5)

            masks_dropped = []
            if df_dropped is not None:
                for mask_data in df_dropped.loc[df_dropped.ImageId == sample_id].EncodedPixels:
                    masks_dropped.append(rle.rle_decode(str(mask_data), mask_shape=orig_img.shape[:2], mask_dtype=np.float32))

            # if len(masks_dropped) == 0:
            #     continue

            fig, ax = plt.subplots(3, 3)

            ax[0, 0].imshow(orig_img)
            ax[0, 1].imshow(light_img)

            masks = []
            for mask_data in df.loc[df.ImageId == sample_id].EncodedPixels:
                masks.append(rle.rle_decode(str(mask_data), mask_shape=orig_img.shape[:2], mask_dtype=np.float32))

            # plt.imshow(metrics.combine_masks(np.array(masks)), alpha=0.5)
            ax[0, 2].imshow(visualisation.apply_masks(light_img*255, np.array(masks)))

            cluster_centers = find_cluster_centers(pred)

            ax[1, 0].imshow(orig_img)
            ax[1, 0].scatter(cluster_centers[:, 1], cluster_centers[:, 0], s=3)

            ax[1, 1].set_title('mask')
            ax[1, 1].imshow(pred[..., 0])

            ax[1, 2].set_title('predicted centers of nuclei')
            ax[1, 2].imshow(pred[:, :, 3])

            # ax[2, 0].set_title('mask moved by to_center vector')
            # center_of_mass = predictions_to_center_of_mass(pred)
            # ax[2, 0].imshow(np.clip(center_of_mass, 0, 10.0) / 10.0)
            ax[2, 0].set_title('dropped masks {}'.format(len(masks_dropped)))
            ax[2, 0].imshow(visualisation.apply_masks(orig_img, np.array(masks_dropped)))

            ax[2, 1].imshow(scipy.signal.convolve2d(pred[:, :, 3], round_kernel(3), mode='same'))
            ax[2, 1].set_title('centers sum r=3')

            ax[2, 2].imshow(scipy.signal.convolve2d(pred[:, :, 3], round_kernel(5), mode='same'))
            ax[2, 2].set_title('centers sum r=5')

            plt.show()


def prepare_combined_submission(submission_name, model_name, checkpoints, filter_masks):
    submission_names = []
    for checkpoint in checkpoints.split(','):
        K.clear_session()
        ch = int(checkpoint)
        weights = glob.glob(f'../output/checkpoints/{model_name}_-1/checkpoint-{ch:03}*')[0]
        sub_name = f'{submission_name}_{model_name}_ch{ch}'
        prepare_submission(model_name=model_name, weights=weights, submission_name=sub_name, use_tta=True)
        submission_names.append(sub_name)

    combine_submissions(f'{submission_name}_{model_name}_{checkpoints}',
                        submissions_to_combine=','.join(submission_names),
                        filter_masks=filter_masks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='unet')
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--sample_id', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--submission_name', type=str, default='')
    parser.add_argument('--submissions_to_combine', type=str, default='')
    parser.add_argument('--use-tta', action='store_true')
    parser.add_argument('--filter-masks', action='store_true')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--checkpoints', type=str, default='128,136,144,152,160')
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    action = args.action
    model = args.model
    weights = args.weights
    fold = args.fold
    use_tta = args.use_tta

    if action == 'train':
        train(model_name=model, fold=fold, run=args.run, weights=weights)
    elif action == 'check':
        check(model_name=model, fold=fold, weights=weights, sample_id=args.sample_id)
    elif action == 'check_tta':
        check_tta(model_name=model, fold=fold, weights=weights, sample_id=args.sample_id)
    elif action == 'check_full':
        check_full(model_name=model, fold=fold, weights=weights, sample_id=args.sample_id, run=args.run, checkpoint=args.checkpoint)
    elif action == 'check_extra':
        check_extra()
    elif action == 'check_test':
        check_test(model_name=model, weights=weights)
    elif action == 'predict_train':
        predict_train(model_name=model, fold=fold, run=args.run, weights=weights, use_tta=use_tta, checkpoint=args.checkpoint)
    elif action == 'check_score_train':
        check_score_train(model_names=model, fold=fold, runs=args.run, checkpoint=args.checkpoint, filter_masks=args.filter_masks)
    elif action == 'predict_check_train':
        predict_train(model_name=model, fold=fold, run=args.run, weights=weights, use_tta=use_tta, checkpoint=args.checkpoint)
        check_score_train(model_names=model, fold=fold, runs=args.run, checkpoint=args.checkpoint, filter_masks=args.filter_masks)
    elif action == 'prepare_submission':
        prepare_submission(model_name=model, weights=weights, submission_name=args.submission_name, use_tta=use_tta)
    elif action == 'combine_submissions':
        combine_submissions(submission_name=args.submission_name, submissions_to_combine=args.submissions_to_combine,
                            filter_masks=args.filter_masks)
    elif action == 'check_combine_submissions':
        check_combine_submissions(submissions_to_combine=args.submissions_to_combine,
                                  filter_masks=args.filter_masks, single_sample_id=args.sample_id)
    elif action == 'check_submission':
        check_submission(submission_name=args.submission_name)
    elif action == 'predict_extra_data':
        predict_extra_data(model_name=model, weights=weights, run=args.run, data_dir=args.data_dir, use_tta=use_tta)
    elif action == 'predict_extra_data_from_runs':
        predict_extra_data_from_runs(runs=args.run, data_dir=args.data_dir)
    elif action == 'prepare_combined_submission':
        prepare_combined_submission(submission_name=args.submission_name, model_name=model, checkpoints=args.checkpoints,
                                    filter_masks=args.filter_masks)
