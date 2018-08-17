import random
import colorsys
import numpy as np
import skimage.morphology


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def apply_masks(image, masks):
    nb_masks = masks.shape[0]
    colors = random_colors(nb_masks)
    masked_image = image.copy() / 255.0

    for i in range(nb_masks):
        mask = masks[i]
        # utils.print_stats('mask', mask)
        # utils.print_stats('masked image', masked_image)
        # print('color', colors[i])
        masked_image = apply_mask(masked_image, mask, colors[i])

        contour = mask - skimage.morphology.erosion(mask, skimage.morphology.disk(1))
        masked_image = apply_mask(masked_image, contour, np.power(colors[i], 0.5))

    return masked_image
