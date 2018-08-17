INPUT_DIR = '../input/'
# TRAIN_DIR = INPUT_DIR + 'stage1_train/'
TRAIN_DIR = '../kaggle-dsbowl-2018-dataset-fixes/stage1_train/'
TEST_DIR = INPUT_DIR + 'stage2_test_final/'
EXTRA_DATA_DIR = INPUT_DIR + 'extra_data/'
CACHE_DIR = '../cache/'

SAMPLE_SUBMISSION = '../input/stage2_sample_submission_final.csv'
SUBMISSION_DIR = '../input/stage2_test_final'

EXPECTED_MASK_SIZE = 24

IMG_SCALE = 1
CHECK_MISSING_CENTERS_FROM_WSHED = True


# THRESHOLD_IOU = 0.6
# THRESHOLD_CENTER_AREA = 4.5
# THRESHOLD_MASK_MEAN = 0.55

#  sub2:
THRESHOLD_IOU = 0.58
THRESHOLD_CENTER_AREA = 4.4
THRESHOLD_MASK_MEAN = 0.55
