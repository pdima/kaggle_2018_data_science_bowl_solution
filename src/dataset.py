from collections import namedtuple
import numpy as np
import pandas as pd
import math
import random
import os
import pickle
import utils
from PIL import Image
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import skimage.filters
import skimage.transform
import skimage.color
import scipy.ndimage.measurements
import scipy.misc
import numba
import enum
import gc

import imgaug as ia
from imgaug import augmenters as iaa
import extra_augmentations
import config

CROP_SIZE = 256 * config.IMG_SCALE

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])

WATERSHED_ENERGY_LEVELS = 3
WATERSHED_DISTANCES = [1.5, 3.0, 5.0]

Y_MASK = 0
Y_OFFSET_TO_CENTER_ROW = 1
Y_OFFSET_TO_CENTER_COL = 2
Y_CENTER = 3
Y_VECTOR_FROM_BRODER_ROW = 4
Y_VECTOR_FROM_BRODER_COL = 5
Y_WATERSHED_ENERGY_1 = 6
Y_WATERSHED_ENERGY_2 = 7
Y_WATERSHED_ENERGY_3 = 8
Y_MASK_AREA = 9


class ImageType(enum.Enum):
    Monochrome = 1
    Color = 2


def show_images(images):
    nb_images = len(images)
    for i, img in enumerate(images):
        plt.subplot((nb_images+1)//2+1, 2, i+1)
        plt.imshow(img)
    plt.show()


@numba.jit
def watershed_direction_and_energy(crops, output_mask_area):
    if output_mask_area:
        res_channels = 2 + WATERSHED_ENERGY_LEVELS + 1
    else:
        res_channels = 2 + WATERSHED_ENERGY_LEVELS

    res = np.zeros((CROP_SIZE, CROP_SIZE, res_channels))

    for crop in crops:
        rows_non_zero = crop.sum(axis=1).nonzero()[0]
        cols_non_zero = crop.sum(axis=0).nonzero()[0]

        if np.sum(rows_non_zero) == 0:
            continue

        row_non_zero_from = rows_non_zero[0]
        row_non_zero_to = rows_non_zero[-1]

        col_non_zero_from = cols_non_zero[0]+1
        col_non_zero_to = cols_non_zero[-1]+1

        if row_non_zero_from > 0:
            row_non_zero_from -= 1
        if col_non_zero_from > 0:
            col_non_zero_from -= 1
        if row_non_zero_to < CROP_SIZE:
            row_non_zero_to += 1
        if col_non_zero_to < CROP_SIZE:
            col_non_zero_to += 1

        crop_non_zero = crop[row_non_zero_from:row_non_zero_to, col_non_zero_from:col_non_zero_to]
        rows, cols = crop_non_zero.shape

        res_crop = np.zeros((rows, cols, res_channels))

        crop_smooth = scipy.ndimage.morphology.binary_dilation(crop_non_zero)
        crop_smooth = scipy.ndimage.morphology.binary_erosion(crop_smooth)

        edt, inds = scipy.ndimage.morphology.distance_transform_edt(crop_smooth,
                                                                    return_distances=True,
                                                                    return_indices=True)
        border_vector = np.array([
            np.expand_dims(np.arange(0, rows), axis=1) - inds[0],
            np.expand_dims(np.arange(0, cols), axis=0) - inds[1]])

        border_vector_norm = border_vector / (np.linalg.norm(border_vector, axis=0, keepdims=True) + 1e-5)
        # border_vector_norm = scipy.misc.imresize(border_vector_norm, 50, interp='bilinear')

        res_crop[:, :, 0] = border_vector_norm[0]
        res_crop[:, :, 1] = border_vector_norm[1]

        for i in range(WATERSHED_ENERGY_LEVELS):
            res_crop[:, :, 2+i] = (edt >= WATERSHED_DISTANCES[i])

        if output_mask_area:
            crop_area = np.sum(crop_non_zero)
            res_crop[:, :, -1] = crop_area

        # mask = crop > 0
        res[crop > 0] = res_crop[crop_non_zero > 0]

        # show_images([crop, mask, ] + [res[:, :, i] for i in range(6)])
    # print(res.shape)
    return res


class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self,
                 sample_id,
                 src_center_x,
                 src_center_y,
                 scale_x=1.0,
                 scale_y=1.0,
                 angle=0.0,
                 shear=0.0,
                 hflip=False,
                 vflip=False,
                 aug: iaa.Augmenter=None):
        self.sample_id = sample_id
        self.src_center_x = src_center_x
        self.src_center_y = src_center_y
        self.angle = angle
        self.shear = shear
        self.scale_y = scale_y
        self.scale_x = scale_x
        self.vflip = vflip
        self.hflip = hflip
        self.aug = aug

    def __str__(self):
        return str(self.__dict__)

    def transform(self):
        scale_x = self.scale_x
        if self.hflip:
            scale_x *= -1
        scale_y = self.scale_y
        if self.vflip:
            scale_y *= -1

        tform = skimage.transform.AffineTransform(translation=(self.src_center_x, self.src_center_y))
        tform = skimage.transform.AffineTransform(scale=(1.0/self.scale_x, 1.0/self.scale_y)) + tform
        tform = skimage.transform.AffineTransform(rotation=self.angle * math.pi / 180,
                                shear=self.shear * math.pi / 180) + tform
        tform = skimage.transform.AffineTransform(translation=(-CROP_SIZE/2, -CROP_SIZE/2)) + tform

        return tform


class Dataset:
    def __init__(self, fold,
                 seed=42,
                 normalize_size=False,
                 expected_mask_size=config.EXPECTED_MASK_SIZE,
                 use_extra_data=True):
        self.fold = fold
        folds = pd.read_csv('../output/folds.csv')
        self.normalize_size = normalize_size
        self.expected_mask_size = expected_mask_size
        self.use_extra_data = use_extra_data
        self.seed = seed

        ia.seed(seed)

        self.train_sample_ids = list(sorted(folds[folds.fold != fold].sample_id))

        if fold == -1:
            # use some for validation even with the full dataset
            self.validation_sample_ids = list(sorted(folds[folds.fold == 2].sample_id))
        else:
            self.validation_sample_ids = list(sorted(folds[folds.fold == fold].sample_id))

        self.nb_train = len(self.train_sample_ids)
        self.nb_test = len(self.validation_sample_ids)

        print(f'Fold: {fold}, train samples: {self.nb_train} validation samples: {self.nb_test}')

        self.median_mask_size = {}
        with utils.timeit_context('load'):
            try:
                self.images, self.masks, self.extra_sample_ids = pickle.load(open('../cache/masks.pkl', 'rb'))
            except FileNotFoundError:
                self.images, self.masks, self.extra_sample_ids = self.load()
                pickle.dump((self.images, self.masks, self.extra_sample_ids), open('../cache/masks.pkl', 'wb'))

            for sample_id, mask_samples in self.masks.items():
                size_values = []
                for mask, crop_offset in mask_samples:
                    area = np.sum(mask) / 255.0
                    size_values.append(area ** 0.5)
                self.median_mask_size[sample_id] = np.median(size_values)

        print('extra samples: ', len(self.extra_sample_ids))
        if use_extra_data:
            self.train_sample_ids += self.extra_sample_ids

    def load(self):
        images = {}
        masks = {}
        extra_sample_ids = []

        print('load dataset...')
        for sample_id in self.train_sample_ids + self.validation_sample_ids:
            sample_dir = config.TRAIN_DIR+sample_id
            with Image.open(f'{sample_dir}/images/{sample_id}.png') as img:
                images[sample_id] = np.array(img)[:, :, :3]  # skip alpha

            sample_masks = []

            masks_dir = f'{sample_dir}/masks'
            for mask_fn in os.listdir(masks_dir):
                with Image.open(f'{masks_dir}/{mask_fn}') as mask:
                    m = np.array(mask) > 0
                    sample_masks.append(utils.nonzero_crop(m))
            masks[sample_id] = sample_masks

        print('load extra samples...')
        extra_datasets = ['set1', 'set12', 'set5', 'set9', 'set18', 'set19', 'set21', 'set22', 'set23',
                          'train_synth_easy0.1_seed12', #'train_synth_hard0.2_seed14',  # 'set14',
                          'set_bbbc022_w2', 'set_bbbc022_w3', 'set_bbbc022_w4', 'set_bbbc022_w5', 'stage1_train']
        for extra_set in extra_datasets:
            extra_set_dir = os.path.join(config.EXTRA_DATA_DIR, extra_set)
            for base_sample_id in sorted(os.listdir(extra_set_dir)):
                sample_id = 'extra_'+extra_set+'_'+base_sample_id
                sample_dir = os.path.join(extra_set_dir, base_sample_id)

                with Image.open(f'{sample_dir}/images/{base_sample_id}.png') as img:
                    img_arr = np.array(img)
                    print(img_arr.shape, extra_set, base_sample_id)
                    if len(img_arr.shape) == 2:
                        img_arr2 = np.zeros(img_arr.shape+(3,), dtype=img_arr.dtype)
                        img_arr2[:, :, 0] = img_arr
                        img_arr2[:, :, 1] = img_arr
                        img_arr2[:, :, 2] = img_arr
                        img_arr = img_arr2
                    else:
                        img_arr = img_arr[:, :, :3]  # skip alpha
                gc.collect()

                sample_masks = []

                masks_dir = f'{sample_dir}/masks'
                for mask_fn in sorted(os.listdir(masks_dir)):
                    with Image.open(f'{masks_dir}/{mask_fn}') as mask:
                        try:
                            m = np.array(mask) > 0
                            sample_masks.append(utils.nonzero_crop(m))
                        except IndexError:
                            print(f'skip mask {masks_dir}/{mask_fn} for sample {sample_id}')

                if len(sample_masks):
                    extra_sample_ids.append(sample_id)
                    masks[sample_id] = sample_masks
                    images[sample_id] = img_arr


        return images, masks, extra_sample_ids


def preprocess_func_mono3(crop):
    gray = skimage.color.rgb2gray(crop / 255.0)
    gray = gray * 2 - 1.0

    X = np.zeros_like(crop, dtype=np.float32)
    X[..., 0] = gray
    X[..., 1] = gray
    X[..., 2] = gray

    return X


def preprocess_func_h3(crop):
    if len(crop.shape) == 3:
        h = skimage.color.rgb2hed(crop / 255.0 * 2 - 1)[:, :, 0]
    elif len(crop.shape) == 4:
        h = np.array([skimage.color.rgb2hed(c / 255.0 * 2 - 1)[:, :, 0] for c in crop])
    else:
        assert False, ('invalid crop shape size:', crop.shape)

    X = np.zeros_like(crop, dtype=np.float32)
    X[..., 0] = h
    X[..., 1] = h
    X[..., 2] = h

    return X





def preprocess_func_h_and_mono(crop):
    if len(crop.shape) == 3:
        h = skimage.color.rgb2hed(crop / 255.0 * 2 - 1)[:, :, 0]
    elif len(crop.shape) == 4:
        h = np.array([skimage.color.rgb2hed(c / 255.0 * 2 - 1)[:, :, 0] for c in crop])
    else:
        assert False, ('invalid crop shape size:', crop.shape)
    gray = skimage.color.rgb2gray(crop / 255.0)
    gray = gray * 2 - 1.0

    X = np.zeros_like(crop, dtype=np.float32)
    X[..., 0] = gray
    X[..., 1] = h * -1
    X[..., 2] = 0

    return X


class UVectorNetDataset(Dataset):
    def __init__(self, fold,
                 preprocess_func,
                 seed=1,
                 batch_size=8,
                 output_watershed=True,
                 output_mask_area=True,
                 output_mask_only=False,
                 use_extra_data=True):
        super().__init__(fold, use_extra_data=use_extra_data, seed=seed)
        self.batch_size = batch_size
        self.preprocess_func = preprocess_func
        self.output_watershed = output_watershed
        self.output_mask_area = output_mask_area
        self.output_mask_only = output_mask_only

        print(f'output_watershed {output_watershed} output_mask_area {output_mask_area} output_mask_only '
              f'{output_mask_only}')

        self.pool = ThreadPool(8)

        self.last_requests = []  # for debugging

        self.img_median = {}
        self.img_gray_mean = {}
        self.img_gray_max = {}
        self.img_gray_min = {}
        self.img_gray_std = {}
        self.img_value_scale = {}
        # self.img_std = {}
        for sample_id, img in self.images.items():
            img_median = np.median(img, axis=(0, 1))
            self.img_median[sample_id] = img_median
            # self.img_std[sample_id] = np.max(np.std(img, axis=(0, 1)))
            max_value = np.max(np.abs(img - img_median))
            self.img_value_scale[sample_id] = 1.0/max_value

            gray = skimage.color.rgb2gray(img)
            self.img_gray_mean[sample_id] = gray.mean()
            self.img_gray_std[sample_id] = gray.std()
            self.img_gray_max[sample_id] = np.max(gray)
            self.img_gray_min[sample_id] = np.min(gray)

    def preprocess(self, crop):
        return self.preprocess_func(crop)

        # if not self.use_colors:
        #     if self.convert_to_hed:
        #         if len(crop.shape) == 3:
        #             gray = skimage.color.rgb2hed(crop / 255.0 * 2 - 1)[:, :, 0]
        #         elif len(crop.shape) == 4:
        #             gray = np.array([skimage.color.rgb2hed(c / 255.0 * 2 - 1)[:, :, 0] for c in crop])
        #     else:
        #         gray = skimage.color.rgb2gray(crop/255.0)
        #         gray = gray * 2 - 1.0
        #
        #     X = np.zeros_like(crop, dtype=np.float32)
        #     X[..., 0] = gray
        #     X[..., 1] = gray
        #     X[..., 2] = gray
        # else:
        #     X = crop/127.5 - 1.0
        #
        # return X

    def generate_x(self, sample_cfg: SampleCfg, img=None):
        if img is None:
            img = self.images[sample_cfg.sample_id]
        # utils.print_stats('img', img)
        # if np.max(img) < 128:
        img = img * 0.9 / np.percentile(img, 99.75)  # 0.9 to keep some space for augmentations
        # img = img * 0.9 / np.max(img)  # 0.9 to keep some space for augmentations

        # if self.img_gray_mean[sample_cfg.sample_id] > 0.5:
        #     img = 255 - img

        crop = skimage.transform.warp(img, sample_cfg.transform(),
                                      mode='constant',
                                      cval=self.img_gray_min[sample_cfg.sample_id],
                                      order=1,
                                      output_shape=(CROP_SIZE, CROP_SIZE))*255.0
        # utils.print_stats('img warp', img)
        if sample_cfg.aug:
            crop = sample_cfg.aug.augment_image(crop.astype(np.uint8)).astype(np.float32)

        return self.preprocess(crop)

    def y_depth(self):
        if self.output_mask_only:
            return 1

        res = 4
        if self.output_watershed:
            res += 2 + WATERSHED_ENERGY_LEVELS
            if self.output_mask_area:
                res += 1
        return res

    def generate_y(self, sample_cfg: SampleCfg):
        tform = sample_cfg.transform()
        res = np.zeros((CROP_SIZE, CROP_SIZE, self.y_depth()))

        # offset_field = np.zeros((CROP_SIZE, CROP_SIZE, 2))

        crops = []
        for mask, crop_offset in self.masks[sample_cfg.sample_id]:
            crop = utils.transform_crop(mask, crop_offset, tform, output_shape=(CROP_SIZE, CROP_SIZE))
            crops.append(crop)
            if np.max(crop) > 0.5:
                res[:, :, 0] = np.maximum(res[:, :, 0], crop)

                if not self.output_mask_only:
                    center_of_mass = scipy.ndimage.measurements.center_of_mass(crop)

                    current_offset_field = np.zeros((CROP_SIZE, CROP_SIZE, 2))
                    current_offset_field[:, :, 0] = np.expand_dims(center_of_mass[0] - np.arange(0, CROP_SIZE), axis=1)
                    current_offset_field[:, :, 1] = np.expand_dims(center_of_mass[1] - np.arange(0, CROP_SIZE), axis=0)

                    res[:, :, 1:3][crop > 0.5] = current_offset_field[crop > 0.5]

                    center_row = int(np.round(np.clip(center_of_mass[0], 1, CROP_SIZE - 2)))
                    center_col = int(np.round(np.clip(center_of_mass[1], 1, CROP_SIZE - 2)))

                    res[center_row-1:center_row+2, center_col-1:center_col+2, 3] = 1

        if self.output_watershed:
            res[:, :, 4:] = watershed_direction_and_energy(crops, output_mask_area=self.output_mask_area)

        return res

    def generate_train(self):
        sample_weights = []

        for sample_id in self.train_sample_ids:
            area = self.images[sample_id].shape[0] * self.images[sample_id].shape[1]
            # scale = np.clip(self.expected_mask_size/self.median_mask_size[sample_id], 0.5, 2.0)
            # use more of larger images but not proportionally to area, let's keep some variability as well
            weight = area**0.5
            if sample_id.startswith('extra_set14'):
                weight *= 0.1
            if 'synth' in sample_id:
                weight *= 0.2
            sample_weights.append(weight)

        sample_weights = np.array(sample_weights)
        sample_weights /= np.sum(sample_weights)

        while True:
            requests = []
            for i in range(self.batch_size):
                sample_id = np.random.choice(self.train_sample_ids, p=sample_weights)
                img = self.images[sample_id]

                scale = 2 ** np.random.uniform(-1.1, 1.1)
                scale *= config.IMG_SCALE

                scale_x = 2 ** np.random.uniform(-0.3, 0.3)
                scale_y = 2 ** np.random.uniform(-0.3, 0.3)

                cfg = SampleCfg(
                    sample_id=sample_id,
                    src_center_x=np.random.uniform(CROP_SIZE // 2 - 32, img.shape[1] - CROP_SIZE // 2 + 32),
                    src_center_y=np.random.uniform(CROP_SIZE // 2 - 32, img.shape[0] - CROP_SIZE // 2 + 32),
                    angle=np.random.uniform(-180, 180),
                    shear=np.random.uniform(-15, 15),
                    scale_x=scale * scale_x,
                    scale_y=scale * scale_y
                )

                cfg.aug = iaa.Sequential(
                    [
                        # iaa.Invert(p=0.2),
                        iaa.Sometimes(0.25, iaa.Grayscale(alpha=(0.0, 1.0))),
                        iaa.Sometimes(0.25, iaa.WithColorspace(
                                to_colorspace="HSV",
                                from_colorspace="RGB",
                                children=[
                                    # iaa.WithChannels(0, iaa.Add((0, 255))),
                                    iaa.WithChannels(0, iaa.Add((-10, 10))),
                                    iaa.WithChannels(1, iaa.Add((-25, 25))),
                                    iaa.WithChannels(2, iaa.Multiply((0.8, 1.1)))
                                ]
                            )
                        ),
                        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0.0, 3.0))),
                        iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.025*255))),
                        iaa.Sometimes(0.5, extra_augmentations.Gamma(-1.0, 1.0))
                    ]
                )

                requests.append(cfg)

            X = self.pool.map(self.generate_x, requests)
            y = self.pool.map(self.generate_y, requests)

            self.last_requests = requests

            yield np.array(X), np.array(y)

    def load_validation(self):
        requests = []

        for sample_id in self.validation_sample_ids:
            img = self.images[sample_id]

            x_steps = (img.shape[1] + CROP_SIZE//2) // CROP_SIZE
            y_steps = (img.shape[0] + CROP_SIZE//2) // CROP_SIZE

            for x_step in range(x_steps):
                for y_step in range(y_steps):
                    cfg = SampleCfg(
                        sample_id=sample_id,
                        src_center_x=CROP_SIZE // 2 + (img.shape[1] - CROP_SIZE) * x_step // x_steps,
                        src_center_y=CROP_SIZE // 2 + (img.shape[0] - CROP_SIZE) * y_step // y_steps,
                        scale_x=config.IMG_SCALE,
                        scale_y=config.IMG_SCALE
                    )
                    requests.append(cfg)

        X = self.pool.map(self.generate_x, requests)
        y = self.pool.map(self.generate_y, requests)

        y = np.array(y)
        return np.array(X), y


if __name__ == '__main__':
    data = UVectorNetDataset(fold=1)
    sample_id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
    sample_cfg = SampleCfg(
        sample_id=sample_id,
        src_center_x=128,
        src_center_y=128
    )
    data.generate_y(sample_cfg)
