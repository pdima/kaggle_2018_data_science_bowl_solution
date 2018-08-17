import numpy as np
import dataset

class TTA:
    def __init__(self, h_flip, v_flip, transpose):
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.transpose = transpose

    def process_image(self, img):
        img2 = img.copy()
        if self.h_flip:
            img2 = np.fliplr(img2).copy()
        if self.v_flip:
            img2 = np.flipud(img2).copy()
        if self.transpose:
            img2 = np.transpose(img2, axes=(1, 0, 2)).copy()
        return img2

    def process_prediction(self, prediction):
        if self.transpose:
            prediction = np.transpose(prediction, axes=(1, 0, 2)).copy()
            prediction[:, :, [dataset.Y_OFFSET_TO_CENTER_ROW, dataset.Y_OFFSET_TO_CENTER_COL]] = \
                prediction[:, :, [dataset.Y_OFFSET_TO_CENTER_COL, dataset.Y_OFFSET_TO_CENTER_ROW]].copy()

            prediction[:, :, [dataset.Y_VECTOR_FROM_BRODER_ROW, dataset.Y_VECTOR_FROM_BRODER_COL]] = \
                prediction[:, :, [dataset.Y_VECTOR_FROM_BRODER_COL, dataset.Y_VECTOR_FROM_BRODER_ROW]].copy()
        if self.v_flip:
            prediction = np.flipud(prediction).copy()
            prediction[..., dataset.Y_OFFSET_TO_CENTER_ROW] *= -1
            prediction[..., dataset.Y_VECTOR_FROM_BRODER_ROW] *= -1
        if self.h_flip:
            prediction = np.fliplr(prediction).copy()
            prediction[..., dataset.Y_OFFSET_TO_CENTER_COL] *= -1
            prediction[..., dataset.Y_VECTOR_FROM_BRODER_COL] *= -1

        return prediction

    def __str__(self):
        res = ''
        if self.transpose:
            res = res + ' tr'
        if self.v_flip:
            res = res + ' v_flip'
        if self.h_flip:
            res = res + ' h_flip'
        return res