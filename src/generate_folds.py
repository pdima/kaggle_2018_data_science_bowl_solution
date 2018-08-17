import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

import config

CROP_SIZE = 256
NB_FOLDS = 4


def load_img(sample_id):
    img = Image.open(f'{config.TRAIN_DIR}{sample_id}/images/{sample_id}.png')
    img = np.array(img)[:, :, :3]
    return img


def build_model():
    base_model = VGG16(include_top=False, input_shape=(CROP_SIZE, CROP_SIZE, 3))
    base_model.summary()
    x = base_model.get_layer('block4_pool').output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.inputs, outputs=x)
    model.summary()

    return model


def center_crop(img):
    center_row = img.shape[0] // 2
    center_col = img.shape[0] // 2

    return img[center_row - 128: center_row + 128, center_col - 128:center_col + 128]


def generate_features(sample_ids):
    images = []
    crops = []

    for sample_id in tqdm(sample_ids):
        img = load_img(sample_id)
        images.append(img)
        crops.append(center_crop(img))
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(crop)
        # plt.show()

    model = build_model()
    model.compile('SGD', 'mae')

    X = preprocess_input_vgg16(np.array(crops, dtype=np.float32))
    print(X.shape)
    features = model.predict(X, batch_size=16, verbose=1)

    print(features.shape)
    return features


def check_clusters():
    sample_ids = sorted(os.listdir(config.TRAIN_DIR))
    features = generate_features(sample_ids)
    est = KMeans(n_clusters=NB_FOLDS, max_iter=2000, n_jobs=1, verbose=0, random_state=42)
    clusters = est.fit_predict(features)

    res = []

    for fold in range(NB_FOLDS):
        fold_images = []
        for i in range(32):
            sample_id = np.array(sample_ids)[clusters == fold][i]
            fold_images.append(center_crop(load_img(sample_id)))
        fold_images = np.column_stack(fold_images)
        res.append(fold_images)
    res = np.row_stack(res)

    plt.imshow(res)
    plt.show()


def check_folds():
    folds = pd.read_csv('../output/folds.csv')

    res = []

    for fold in range(NB_FOLDS):
        fold_images = []
        samples = list(folds[folds.fold == fold].sample_id)
        for i in range(128):
            sample_id = samples[i]
            fold_images.append(center_crop(load_img(sample_id)))
        fold_images = np.column_stack(fold_images)
        res.append(fold_images)
    res = np.row_stack(res)

    plt.imshow(res)
    plt.show()


def generate_folds():
    sample_ids = sorted(os.listdir(config.TRAIN_DIR))
    features = generate_features(sample_ids)

    nb_iters = 3

    fold_ids = [[] for _ in range(NB_FOLDS)]
    undistributed_ids = np.array(sample_ids)
    undistributed_features = features

    for iter in range(nb_iters):
        est = KMeans(n_clusters=NB_FOLDS+1, max_iter=2000, n_jobs=1, verbose=0, random_state=42)
        clusters = est.fit_predict(undistributed_features)
        cluster_sizes = [np.sum(clusters == fold) for fold in range(NB_FOLDS+1)]
        cluster_size_order = list(np.argsort(cluster_sizes))

        big_cluster_idx = cluster_size_order[-1]

        for cluster_idx in reversed(cluster_size_order[:-1]):
            smallest_fold_id = np.argmin([len(fold) for fold in fold_ids])
            fold_ids[smallest_fold_id] += list(undistributed_ids[clusters == cluster_idx])

        undistributed_ids = undistributed_ids[clusters == big_cluster_idx]
        undistributed_features = undistributed_features[clusters == big_cluster_idx]

    smallest_fold_id = np.argmin([len(fold) for fold in fold_ids])
    fold_ids[smallest_fold_id] += list(undistributed_ids)

    print([len(fold) for fold in fold_ids])

    df = pd.DataFrame(dict(sample_id=sum(fold_ids, []), fold=sum([[i]*len(fold_ids[i]) for i in range(NB_FOLDS)], [])))
    df.to_csv('../output/folds.csv', index=False, columns=['sample_id', 'fold'])


# check_clusters()
# generate_folds()
check_folds()
