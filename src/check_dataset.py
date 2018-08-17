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


import dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import utils
import config

import imgaug as ia
from imgaug import augmenters as iaa

test_sample_ids = [
    '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9',
    '01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf',
    '08275a5b1c2dfcd739e8c4888a5ee2d29f83eccfa75185404ced1dc0866ea992'
]


def check_sample_transform(sample_id):
    img = Image.open(f'{config.TRAIN_DIR}/{sample_id}/images/{sample_id}.png')
    img = np.array(img)[:, :, :3]

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)

    sample_cfg = dataset.SampleCfg(
        sample_id=sample_id,
        src_center_x=128,
        src_center_y=128,
        shear=15
    )

    crop = skimage.transform.warp(img, sample_cfg.transform(),
                                  mode='constant', order=0, output_shape=(256, 256))

    utils.print_stats('crop', crop)
    plt.imshow(crop)

    plt.show()


def generate_train_benchmark(use_watershed):
    data = dataset.UVectorNetDataset(fold=1, batch_size=8, output_watershed=use_watershed)
    count = 0
    with utils.timeit_context('generate 100 batches'):
        for X, y in data.generate_train():
            count += 1
            if count >= 10:
                break


def check_generate_train():
    data = dataset.UVectorNetDataset(fold=-1, batch_size=1)
    # data.train_sample_ids = data.train_sample_ids[:2]
    while True:
        for X, y in data.generate_train():
            for i in range(data.batch_size):
                plt.subplot(1, 5, 1)
                utils.print_stats('X', X[i])
                plt.imshow(np.clip(X[i]/2+0.5, 0, 1))

                plt.subplot(1, 5, 2)
                plt.imshow(y[i, :, :, 0])
                plt.subplot(1, 5, 3)
                max_vector = np.max(np.abs(y[i])) + 1e-3
                plt.imshow(y[i, :, :, 1] / max_vector)
                plt.subplot(1, 5, 4)
                plt.imshow(y[i, :, :, 2] / max_vector)

                plt.subplot(1, 5, 5)
                plt.imshow(y[i, :, :, 3])

                plt.show()


def check_color_conv():
    data = dataset.UVectorNetDataset(fold=-1, batch_size=1, output_watershed=True, use_colors=False)

    from skimage.color import rgb2hed
    import skimage.exposure
    import PIL.Image

    for sample_id, img in data.images.items():
        print(sample_id)
        fn = '/home/dmytro/ml/kaggle/2018_data_science_bowl/input/stage1_test/0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466/images/0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466.png'
        img = PIL.Image.open(fn)
        img = np.array(img)[:, :, :3]

        plt.subplot(1, 3, 1)
        utils.print_stats('img', img)
        plt.imshow(img)

        plt.subplot(1, 3, 2)
        preprocessed = data.preprocess(img)[:, :, 0]
        utils.print_stats('preprocessed', preprocessed)
        plt.imshow(preprocessed)

        plt.subplot(1, 3, 3)
        img_hed = rgb2hed(img/255.0*2-1)[:, :, 0]
        utils.print_stats('img_hed', img_hed)
        # hed_scaled = skimage.exposure.rescale_intensity(img_hed[:, :, 0]*-1, out_range=(0.0, 1.0))
        # utils.print_stats('img_hed_scaled', img_hed)
        plt.imshow(img_hed * -1)

        plt.show()

def check_mask_max_level():
    data = dataset.UVectorNetDataset(fold=-1, batch_size=1, output_watershed=True,
                                     preprocess_func=dataset.preprocess_func_h_and_mono)
    for sample_id in data.train_sample_ids:
        for mask, offset in data.masks[sample_id]:
            if np.max(mask) != 255:
                print(sample_id)
                break


def check_generate_train_watershed():
    data = dataset.UVectorNetDataset(fold=-1, batch_size=2, output_watershed=True,
                                     preprocess_func=dataset.preprocess_func_h_and_mono)
    data.generate_train_configurations(nb_config=8)

#     data.train_sample_ids = '''0ddd8deaf1696db68b00c600601c6a74a0502caaf274222c8367bdc31458ae7e
# 16c3d5935ba94b720becc24b7a05741c26149e221e3401924080f41e2f891368
# 245b995878370ef4ea977568b2b67f93d4ecaa9308761b9d3e148e0803780183
# 2f929b067a59f88530b6bfa6f6889bc3a38adf88d594895973d1c8b2549fd93d
# 30f65741053db713b3f328d31d3234b6fedbe31df65c1a8ea29be28146cab789
# 35ca5f142a7d7a3e4b59f1a767a31f87cb00d66348226bc64094ee3d1e46531c
# 37ed50eea5a1e0bade3e6753793b6caeb061cd4c2f365658c257f69cab1f6288
# 693bc64581275f04fc456da74f031d583733360a1f6032fa38b3fbf592ff4352
# 77ceeb87f560775ac150b8b9b09684ed3e806d0af6f26cce8f10c5fc280f5df2
# 947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050
# a22b7882fa85b9f0fcef659a7b82bfcddf01710f9a7617a9e036e84ac6901841
# a246bcaa64af48ee5ca181cd594c0fc43466e7614406eb8bc01199a16ebc95d0
# c7d546766518703580f63d5d8f11d54971044753f53c0b257d19c2f99d4bfdd0
# dae976f161fe42dc58dee87d4bf2eb9f65736597cab0114138641b2a39a5c42b
# edd36ed822e7ed760ff73e0524df22aa5bf5c565efcdc6c39603239c0896e7a8
# ef6634efb46567d87b811be786b18c4cd0e2cda23d79b65d6afe0d259ef3ade6
# f26f4c2c70c38fe12e00d5a814d5116691f2ca548908126923fd76ddd665ed24
# f6863b83d75e5927b30e2e326405b588293283c25aaef2251b30c343296b9cb1
# fd8065bcb1afdbed19e028465d5d00cd2ecadc4558de05c6fa28bea3c817aa22'''.split()

    # data.train_sample_ids = data.train_sample_ids[:2]
    while True:
        for X, y in data.generate_train():
            for i in range(data.batch_size):
                sample_cfg = data.last_requests[i]
                if np.max(data.masks[sample_cfg.sample_id][0][0]) == 255:
                    print('skip', sample_cfg.sample_id)
                    continue

                plt.subplot(4, 4, 1)
                utils.print_stats('X', X[i, :, :, 0])
                plt.imshow(X[i, :, :, 0])
                utils.print_stats(f'X', X[i])

                plt.subplot(4, 4, 2)
                utils.print_stats('X', X[i, :, :, 1])
                plt.imshow(X[i, :, :, 1])

                plt.subplot(4, 4, 3)
                img = data.images[sample_cfg.sample_id]
                crop = skimage.transform.warp(img, sample_cfg.transform(),
                                      mode='constant',
                                      cval=data.img_gray_min[sample_cfg.sample_id],
                                      order=1,
                                      output_shape=(256, 256))
                utils.print_stats('img', crop)
                plt.imshow(crop)
                utils.print_stats('mask', data.masks[sample_cfg.sample_id][0][0])

                for j in range(10):
                    utils.print_stats(f'y {j}', y[i, :, :, j])
                    plt.subplot(4, 4, j+4)
                    plt.imshow(y[i, :, :, j])
                plt.show()
                plt.cla()


def check_load_validation():
    data = dataset.UVectorNetDataset(fold=1, batch_size=4)
    X, y = data.load_validation()

    print(X.shape, y.shape)

    for i in range(X.shape[0]):
        plt.subplot(1, 5, 1)
        plt.imshow(np.clip(X[i]/2+0.5, 0, 1))

        plt.subplot(1, 5, 2)
        plt.imshow(y[i, :, :, 0])
        plt.subplot(1, 5, 3)
        max_vector = np.max(np.abs(y[i])) + 1e-3
        plt.imshow(y[i, :, :, 1] / max_vector)
        plt.subplot(1, 5, 4)
        plt.imshow(y[i, :, :, 2] / max_vector)

        plt.subplot(1, 5, 5)
        plt.imshow(y[i, :, :, 3])

        plt.show()


def check_aug(sample_id=test_sample_ids[0]):
    data = dataset.UVectorNetDataset(fold=1, batch_size=4)
    sample_cfg = dataset.SampleCfg(
        sample_id=sample_id,
        src_center_x=128,
        src_center_y=128,
    )

    plt.subplot(1, 3, 1)

    X = data.generate_x(sample_cfg)
    plt.imshow(X)

    #
    # sample_cfg.aug = iaa.Invert(1.0)
    # X = data.generate_x(sample_cfg)
    # plt.imshow(X)
    # plt.show()
    #
    # sample_cfg.aug = iaa.Grayscale(alpha=(0.0, 1.0))
    # X = data.generate_x(sample_cfg)
    # plt.imshow(X)
    # plt.show()

    # sample_cfg.aug = iaa.WithColorspace(
    #     to_colorspace="HSV",
    #     from_colorspace="RGB",
    #     children=[
    #         iaa.WithChannels(0, iaa.Add((0, 255))),
    #         iaa.WithChannels(1, iaa.Add((-25, 25))),
    #         iaa.WithChannels(2, iaa.Multiply((0.5, 2.0)))
    #     ]
    # )
    #
    # sample_cfg.aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
    #
    # for i in range(16):
    #     X = data.generate_x(sample_cfg)
    #     plt.imshow(X)
    #     plt.show()

    plt.subplot(1, 3, 2)

    import extra_augmentations
    # sample_cfg.aug = extra_augmentations.Gamma(-1)
    sample_cfg.aug = iaa.AdditiveGaussianNoise(scale=0.025*255)
    X = data.generate_x(sample_cfg)
    plt.imshow(X)

    plt.subplot(1, 3, 3)

    sample_cfg.aug = extra_augmentations.Gamma(1)
    X = data.generate_x(sample_cfg)
    plt.imshow(X)

    plt.show()


def check_size_distribution():
    data = dataset.UVectorNetDataset(fold=-1, batch_size=4, preprocess_func=lambda x: x)
    area_distribution = []
    size_distribution = []

    for sample_id, mask_samples in data.masks.items():
        for mask, crop_offset in mask_samples:
            if np.max(mask) != 255:
                print(sample_id)
            area = np.sum(mask / 255.0)
            area_distribution.append(area)
            size_distribution.append(area**0.5)

    print(np.median(size_distribution), np.mean(size_distribution))
    plt.hist(size_distribution, bins=1024)
    plt.show()


def check_watershed_direction_and_energy():
    data = dataset.UVectorNetDataset(fold=-1, batch_size=4, output_watershed=True)
    for mask_samples in data.masks.values():
        crops = []
        for mask, crop_offset in mask_samples:
            cfg = dataset.SampleCfg(
                sample_id=0,
                src_center_x=128,
                src_center_y=128,
                scale_x=1,
                scale_y=1
            )
            tform = cfg.transform()
            crop = utils.transform_crop(mask, crop_offset, tform, output_shape=(dataset.CROP_SIZE, dataset.CROP_SIZE))
            crops.append(crop)

        wshed = dataset.watershed_direction_and_energy(crops, output_mask_area=data.output_mask_area)

        plt.subplot(3, 3, 1)
        plt.title('mask')
        plt.imshow(np.sum(crops, axis=0))

        for i in range(wshed.shape[2]):
            plt.subplot(3, 3, i+2)
            plt.title('mask')
            utils.print_stats(f'res {i}', wshed[:, :, i])
            plt.imshow(wshed[:, :, i])
        plt.show()




# check_sample_transform(sample_id=test_sample_id)
# check_generate_train()
# generate_train_benchmark(use_watershed=False)
# generate_train_benchmark(use_watershed=True)
# check_load_validation()
# check_watershed_direction_and_energy()

check_generate_train_watershed()
# check_mask_max_level()

# check_aug(sample_id=test_sample_ids[0])
# check_aug(sample_id=test_sample_ids[1])
# check_aug(sample_id=test_sample_ids[2])

# check_size_distribution()

# check_color_conv()