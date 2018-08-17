import numpy as np
import scipy.io
import scipy.misc
import skimage.io
import skimage.exposure
import skimage.segmentation
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob
import pandas as pd
import tqdm

from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QPainterPath
from PyQt5.QtCore import Qt

from scipy import ndimage as ndi
from skimage.color import rgb2hed

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import utils


def prepare_set5():
    src_dir='../extra_data/set5/CRCHistoPhenotypes_2016_04_28/Detection/'
    dst_dir='../input/extra_data/set5/'

    for sample_id in os.listdir(src_dir):
        if not sample_id.startswith('img'):
            continue

        print(sample_id)
        img = skimage.io.imread(f'{src_dir}/{sample_id}/{sample_id}.bmp')
        points = scipy.io.loadmat(f'{src_dir}/{sample_id}/{sample_id}_detection.mat')['detection']

        dest_img_dir = f'{dst_dir}/{sample_id}/images/'
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(f'{dst_dir}/{sample_id}/masks/', exist_ok=True)

        skimage.io.imsave(f'{dest_img_dir}/{sample_id}.png', img)

        hint = Image.fromarray(img)
        draw = ImageDraw.Draw(hint)
        for point in points:
            draw.ellipse([tuple(point-2), tuple(point+2)], fill='white', outline='black')

        del draw
        normlized = skimage.exposure.equalize_adapthist(np.array(hint))
        skimage.io.imsave(f'{dest_img_dir}/hint.png', normlized)
        # hint.save(f'{dest_img_dir}/hint.png', "PNG")


def prepare_set1():
    src_dir = '../extra_data/nuclei/'
    dst_dir = '../input/extra_data/set1/'

    sample_num = 0
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith('original.tif'):
            continue
        sample_num += 1
        sample_id = f'{sample_num:03}'
        mask_fn = fn.replace('original.tif', 'mask.png')
        print(sample_id)

        dest_img_dir = f'{dst_dir}/{sample_id}/images/'
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(f'{dst_dir}/{sample_id}/masks/', exist_ok=True)

        img = skimage.io.imread(f'{src_dir}/{fn}')
        # img = scipy.misc.imresize(img, 0.5)
        skimage.io.imsave(f'{dest_img_dir}/{sample_id}.png', img)
        # skimage.io.imsave(f'{dest_img_dir}/hint.png', img)

        mask = skimage.io.imread(f'{src_dir}/{mask_fn}')

        # mask_small = scipy.misc.imresize(mask, 1.0/16)

        labels = ndi.label(mask)[0]

        nb_masks = np.max(labels)
        for i in range(nb_masks):
            single_mask = labels == i + 1
            # utils.print_stats('single mask', single_mask*1)
            skimage.io.imsave(f'{dst_dir}/{sample_id}/masks/{i+1:03}.png', single_mask*1)


def prepare_set12():
    src_dir = '../extra_data/set12/'
    dst_dir = '../input/extra_data/set12/'

    sample_num = 0
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith('tif'):
            continue

        sample_num += 1
        sample_id = f'set12_{sample_num:03}'

        dest_img_dir = f'{dst_dir}/{sample_id}/images/'
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(f'{dst_dir}/{sample_id}/masks/', exist_ok=True)

        print(fn)

        img = skimage.io.imread(f'{src_dir}/{fn}')
        # skimage.io.imsave(f'{dest_img_dir}/{sample_id}.png', img)

        img_hed = rgb2hed(img)
        skimage.io.imsave(f'{dest_img_dir}/hint.png',
                          skimage.exposure.rescale_intensity(img_hed[:, :, 0]*-1, out_range=(0, 1.0)))


        annotation_fn = fn.replace('tif', 'xml')
        annotation = ET.parse(f'{src_dir}/Annotations/{annotation_fn}')
        for mask_id, region in enumerate(annotation.getroot().iter('Region')):
            x_points = []
            y_points = []
            vertices = region.findall('Vertices')[0]
            for vertex in vertices.findall('Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                x_points.append(x)
                y_points.append(y)
            # plt.scatter(x_points, y_points, s=1)

            # mask_img = QImage(img.shape[1], img.shape[0], QImage.Format_Grayscale8)
            # mask_img.fill(Qt.black)
            # painter = QPainter()
            # painter.begin(mask_img)
            # path = QPainterPath()
            # path.moveTo(x_points[0], y_points[0])
            # for x, y in zip(x_points, y_points):
            #     path.lineTo(x, y)
            # painter.fillPath(path, Qt.white)
            # painter.end()
            #
            # mask_img.save(f'{dst_dir}/{sample_id}/masks/{mask_id+1:03}.png')

        # plt.figure()
    # plt.show()


def convert_masks():
    """
    Some masks are in 0..1 and some in 0.255 range, convert everything to 0..255
    """
    for fn in sorted(glob.glob('../input/extra_data/*/masks/*.png')):
        print(fn)
        img = skimage.io.imread(fn)
        # utils.print_stats('mask', img)
        img[img > 0] = 255
        skimage.io.imsave(fn, img)


def prepare_bbbc022():
    src_dir = '../extra_data/set14/'
    dst_dir = '../input/extra_data/set14'

    os.makedirs(dst_dir, exist_ok=True)
    check_first = open(src_dir+'files.txt', 'r').readlines()
    df = pd.read_csv(src_dir + 'BBBC022_v1_image.csv', error_bad_lines=False)
    zips = set()
    tifs = [fn.replace('.png\n', '.tif') for fn in check_first]

    src_file_names1 = []
    src_file_names2 = []

    check_df = df[df.Image_FileName_OrigHoechst.isin(tifs)]
    for index, row in check_df.iterrows():
        src = src_dir+'BBBC022_v1_images_' + str(row['Image_Metadata_PlateID']) + 'w1/' + row[
            'Image_FileName_OrigHoechst']
        if os.path.exists(src):
            src_file_names1.append(src)
    for index, row in df.iterrows():
        src = src_dir + 'BBBC022_v1_images_' + str(row['Image_Metadata_PlateID']) + 'w1/' + row[
            'Image_FileName_OrigHoechst']
        if os.path.exists(src):
            src_file_names2.append(src)

    np.random.shuffle(src_file_names2)

    src_file_names = src_file_names1 + list(src_file_names2)

    # zips = check_df.Image_Metadata_PlateID.unique()
    # print(zips)

    # for fn in check_first:
    #
    #     z = 'BBBC022_v1_images_' + str(row['Image_Metadata_PlateID']) + 'w1/'


    # files = data.Image_FileName_OrigHoechst

    nb_files = 0
    for src in tqdm.tqdm(src_file_names[:1024]):
        print(os.path.exists(src))
        if os.path.exists(src):
            nb_files += 1
            img = Image.open(src)
            img = np.array(img).astype(np.float)
            # utils.print_stats('img', img)
            img /= np.max(img)
            # utils.print_stats('img norm', img)
            os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
            os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
            skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)
            print(nb_files)

    print(len(src_file_names1))


def prepare_bbbc022_other():
    for w in [2, 3, 4, 5]:
        src_dir = f'../extra_data/set14/BBBC022_v1_images_20585w{w}'
        dst_dir = f'../extra_data/set_bbbc022_w{w}'

        os.makedirs(dst_dir, exist_ok=True)
        src_file_names = []

        for fn in os.listdir(src_dir):
            if fn.endswith('tif'):
                src = os.path.join(src_dir, fn)
                src_file_names.append(src)

        np.random.shuffle(src_file_names)

        nb_files = 0
        for src in tqdm.tqdm(src_file_names[:64]):
            # print(os.path.exists(src))
            if os.path.exists(src):
                nb_files += 1
                img = Image.open(src)
                img = np.array(img).astype(np.float)
                # utils.print_stats('img', img)
                img /= np.max(img)
                # utils.print_stats('img norm', img)
                os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
                os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
                skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)
                # print(nb_files)


def prepare_set6():
    src_dir = '../extra_data/set6/Dataset/EDF'
    dst_dir = '../extra_data/set6/set6'

    os.makedirs(dst_dir, exist_ok=True)
    src_file_names = []

    for fn in os.listdir(src_dir):
        if ('GT' not in fn) and fn.endswith('png'):
            src = os.path.join(src_dir, fn)
            src_file_names.append(src)

    nb_files = 0
    for src in tqdm.tqdm(src_file_names):
        if os.path.exists(src):
            nb_files += 1
            img = Image.open(src)
            img = np.array(img).astype(np.float)
            # utils.print_stats('img', img)
            img /= np.max(img)
            # utils.print_stats('img norm', img)
            os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
            os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
            skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)


def prepare_set7():
    src_dir = '../extra_data/set7/Breast Cancer Cells'
    dst_dir = '../extra_data/set7/set7'

    os.makedirs(dst_dir, exist_ok=True)
    src_file_names = []

    for fn in os.listdir(src_dir):
        if fn.endswith('tif'):
            src = os.path.join(src_dir, fn)
            src_file_names.append(src)

    nb_files = 0
    for src in tqdm.tqdm(src_file_names):
        if os.path.exists(src):
            nb_files += 1
            img = Image.open(src)
            img = np.array(img).astype(np.float)
            # utils.print_stats('img', img)
            img /= np.max(img)
            # utils.print_stats('img norm', img)
            os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
            os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
            skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)


def prepare_set9():
    src_dir = '../extra_data/set9'
    dst_dir = '../extra_data/set9'

    os.makedirs(dst_dir, exist_ok=True)

    nb_files = 0
    for src in tqdm.tqdm(glob.glob(f'{src_dir}/**/*.png', recursive=True)):
        nb_files += 1
        img = Image.open(src)
        img = np.array(img).astype(np.float)
        # utils.print_stats('img', img)
        img /= np.max(img)
        # utils.print_stats('img norm', img)
        os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
        os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)


def prepare_set18():
    src_dir = '../extra_data/set18/TNBC_NucleiSegmentation'
    dst_dir = '../input/extra_data/set18/'

    sample_num = 0
    for slide_num in range(1, 12):
        slides_dir = f'{src_dir}/Slide_{slide_num:02}/'
        gt_dir = f'{src_dir}/GT_{slide_num:02}/'
        for fn in os.listdir(slides_dir):
            if not fn.endswith('png'):
                continue
            sample_num += 1
            sample_id = f'{sample_num:03}'

            print(sample_id)

            dest_img_dir = f'{dst_dir}/{sample_id}/images/'
            os.makedirs(dest_img_dir, exist_ok=True)
            os.makedirs(f'{dst_dir}/{sample_id}/masks/', exist_ok=True)

            img = skimage.io.imread(f'{slides_dir}/{fn}')
            img_hed = rgb2hed(img[:, :, :3])

            skimage.io.imsave(f'{dest_img_dir}/{sample_id}.png', img)
            skimage.io.imsave(f'{dest_img_dir}/hint.png',
                              skimage.exposure.rescale_intensity(img_hed[:, :, 0] * -1, out_range=(0, 1.0)))

            mask = skimage.io.imread(f'{gt_dir}/{fn}')

            # mask_small = scipy.misc.imresize(mask, 1.0/16)

            labels = ndi.label(mask)[0]

            nb_masks = np.max(labels)
            for i in range(nb_masks):
                single_mask = labels == i + 1
                # utils.print_stats('single mask', single_mask*1)
                skimage.io.imsave(f'{dst_dir}/{sample_id}/masks/{i+1:03}.png', single_mask*1)


def prepare_set19():
    src_dir = '../extra_data/set19/PSB_2015_ImageSize_400/Original_Images'
    dst_dir = '../input/extra_data/set19_2'

    centers = {}
    for line in open('../extra_data/set19/PSB_2015_ImageSize_400/Nuclei_Detection/Experts.csv'):
        items = line.split(',')
        if items[0] == 'Image Name':
            continue
        image_name = items[0]
        x = [float(f) for f in items[1:][::2] if f not in ('', '\n')]
        y = [float(f) for f in items[2:][::2] if f not in ('', '\n')]
        centers[image_name.replace('png', 'tiff')] = (x, y)

    sample_num = 0
    for src in tqdm.tqdm(sorted(os.listdir(src_dir))):
        if src not in centers:
            print('skip', src)
            continue

        sample_num += 1
        sample_id = f'{sample_num:03}'

        # print(sample_id)

        dest_img_dir = f'{dst_dir}/{sample_id}/images/'
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(f'{dst_dir}/{sample_id}/masks/', exist_ok=True)

        img = skimage.io.imread(f'{src_dir}/{src}')

        # plt.imshow(img)
        # plt.scatter(centers[src][0], centers[src][1])
        # plt.show()

        # print(img.shape)
        img_hed = rgb2hed(img[:, :, :3])
        img_hed = skimage.exposure.rescale_intensity(img_hed[:, :, 0] * -1, out_range=np.uint8).astype(np.uint8)

        hint = Image.fromarray(img_hed)

        draw = ImageDraw.Draw(hint)
        for x, y in zip(*centers[src]):
            draw.ellipse([x-2, y-2, x+2, y+2], fill='white', outline='black')
        del draw

        skimage.io.imsave(f'{dest_img_dir}/{sample_id}.png', img)
        skimage.io.imsave(f'{dest_img_dir}/hint.png', np.array(hint))
        np.save(f'{dest_img_dir}/centers.npy', np.array(centers[src]))



def prepare_set21():
    src_dir = '../extra_data/set21'
    dst_dir = '../input/extra_data_check/set21'

    os.makedirs(dst_dir, exist_ok=True)

    nb_files = 0
    for src in tqdm.tqdm(sorted(glob.glob(f'{src_dir}/**/**/*.*', recursive=True))):
        if not src.endswith('jpg') and not src.endswith('png'):
            continue
        nb_files += 1
        img = Image.open(src)
        img = np.array(img).astype(np.float)
        # utils.print_stats('img', img)
        img /= np.max(img)
        # utils.print_stats('img norm', img)
        os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
        os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)



def prepare_set22():
    src_dir = '../extra_data/set22/wikimedia'
    dst_dir = '../input/extra_data_check/set22_2'

    os.makedirs(dst_dir, exist_ok=True)

    nb_files = 0
    for src in tqdm.tqdm(sorted(glob.glob(f'{src_dir}/*.*', recursive=True))):
        nb_files += 1
        img = Image.open(src)
        img = np.array(img).astype(np.float)
        # utils.print_stats('img', img)
        img /= np.max(img)
        # utils.print_stats('img norm', img)
        os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
        os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)

        img_hed = rgb2hed(img)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/hint.png',
                          skimage.exposure.rescale_intensity(img_hed[:, :, 0] * -1, out_range=(0, 1.0)))


def prepare_set23():
    src_dir = '../extra_data/set23/wikimedia2'
    dst_dir = '../input/extra_data_check/set23'

    os.makedirs(dst_dir, exist_ok=True)

    nb_files = 0
    for src in tqdm.tqdm(sorted(glob.glob(f'{src_dir}/*.*', recursive=True))):
        nb_files += 1
        img = Image.open(src)
        img = np.array(img).astype(np.float)
        # utils.print_stats('img', img)
        img /= np.max(img)
        # utils.print_stats('img norm', img)
        os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
        os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)

        img_hed = rgb2hed(img)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/hint.png',
                          skimage.exposure.rescale_intensity(img_hed[:, :, 0] * -1, out_range=(0, 1.0)))


def prepare_set24():
    src_dir = '../extra_data/stanford_and_rna/stanford_and_rna'
    dst_dir = '../input/extra_data_check/set24_h'

    os.makedirs(dst_dir, exist_ok=True)

    nb_files = 0
    for src in tqdm.tqdm(sorted(glob.glob(f'{src_dir}/*.*', recursive=True))):
        nb_files += 1
        img = Image.open(src)
        img = np.array(img).astype(np.float)
        # utils.print_stats('img', img)
        img /= np.max(img)
        # utils.print_stats('img norm', img)
        os.makedirs(f'{dst_dir}/{nb_files:04}/images', exist_ok=True)
        os.makedirs(f'{dst_dir}/{nb_files:04}/masks', exist_ok=True)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/{nb_files:04}.png', img)

        img_hed = rgb2hed(img)
        skimage.io.imsave(f'{dst_dir}/{nb_files:04}/images/hint.png',
                          skimage.exposure.rescale_intensity(img_hed[:, :, 0] * -1, out_range=(0, 1.0)))


def prepare_stage1_test_data():
    # src_dir = '../input/stage2_test_final/'
    src_dir = '../input/stage1_test/'

    import rle
    import PIL
    df = pd.read_csv(f'../input/stage1_solution.csv')
    for sample_id in os.listdir(src_dir):
        img = PIL.Image.open(f'{src_dir}/{sample_id}/images/{sample_id}.png')
        orig_img = np.array(img)
        masks = []
        os.makedirs(f'{src_dir}/{sample_id}/masks/', exist_ok=True)
        for mask_data in df.loc[df.ImageId == sample_id].EncodedPixels:
            masks.append(rle.rle_decode(str(mask_data), mask_shape=orig_img.shape[:2], mask_dtype=np.float32))
            mask_num = len(masks)
            skimage.io.imsave(f'{src_dir}/{sample_id}/masks/{mask_num:04}.png', masks[-1])


# prepare_set1()
# prepare_set12()
# convert_masks()
# prepare_bbbc022()
# prepare_bbbc022_other()

# prepare_set9()

# prepare_set18()

# prepare_set19()

# prepare_set23()

# prepare_set24()

prepare_stage1_test_data()
