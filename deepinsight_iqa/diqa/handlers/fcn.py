from typing import Callable
from keras.models import Model
import keras.layers as KL
from .utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .resnet50 import get_resnet50_encoder
from . import IMAGE_ORDERING


def vanilla_encoder(input_height=224, input_width=224, channels=3):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    if IMAGE_ORDERING == 'channels_first':
        img_input = KL.Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = KL.Input(shape=(input_height, input_width, channels))

    x = img_input
    levels = []

    x = (KL.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (KL.Conv2D(filter_size, (kernel, kernel),
                   data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (KL.BatchNormalization())(x)
    x = (KL.Activation('relu'))(x)
    x = (KL.MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    x = (KL.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (KL.Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING,
                   padding='valid'))(x)
    x = (KL.BatchNormalization())(x)
    x = (KL.Activation('relu'))(x)
    x = (KL.MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    for _ in range(3):
        x = (KL.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (KL.Conv2D(256, (kernel, kernel),
                       data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (KL.BatchNormalization())(x)
        x = (KL.Activation('relu'))(x)
        x = (KL.MaxPooling2D((pool_size, pool_size),
                             data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels


# crop o1 wrt o2
def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height2 = o_shape2[2]
        output_width2 = o_shape2[3]
    else:
        output_height2 = o_shape2[1]
        output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    if IMAGE_ORDERING == 'channels_first':
        output_height1 = o_shape1[2]
        output_width1 = o_shape1[3]
    else:
        output_height1 = o_shape1[1]
        output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = KL.Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = KL.Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(o2)

    if output_height1 > output_height2:
        o1 = KL.Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = KL.Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(o2)

    return o1, o2


def fcn_8(
    n_classes: int,
    encoder: Callable = vanilla_encoder,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (KL.Conv2D(4096, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = KL.Dropout(0.5)(o)
    o = (KL.Conv2D(4096, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = KL.Dropout(0.5)(o)

    o = (KL.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o)
    o = KL.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2),
                           use_bias=False, data_format=IMAGE_ORDERING)(o)

    o2 = f4
    o2 = (KL.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o2)

    o, o2 = crop(o, o2, img_input)

    o = KL.Add()([o, o2])

    o = KL.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2),
                           use_bias=False, data_format=IMAGE_ORDERING)(o)
    o2 = f3
    o2 = (KL.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o2)
    o2, o = crop(o2, o, img_input)
    o = KL.Add(name="seg_feats")([o2, o])

    o = KL.Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8),
                           use_bias=False, data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)
    model.model_name = "fcn_8"
    return model


def fcn_32(
    n_classes: int,
    encoder: Callable = vanilla_encoder,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (KL.Conv2D(4096, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = KL.Dropout(0.5)(o)
    o = (KL.Conv2D(4096, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = KL.Dropout(0.5)(o)

    o = (KL.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal',
                   data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = KL.Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(
        32, 32), use_bias=False, data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)
    model.model_name = "fcn_32"
    return model


def fcn_8_vgg(
    n_classes: int,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):
    model = fcn_8(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_8_vgg"
    return model


def fcn_32_vgg(
    n_classes: int,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):
    model = fcn_32(n_classes, get_vgg_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_vgg"
    return model


def fcn_8_resnet50(
    n_classes: int,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):
    model = fcn_8(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_8_resnet50"
    return model


def fcn_32_resnet50(
    n_classes: int,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):
    model = fcn_32(n_classes, get_resnet50_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_resnet50"
    return model


def fcn_8_mobilenet(
    n_classes: int,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):
    model = fcn_8(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_8_mobilenet"
    return model


def fcn_32_mobilenet(
    n_classes: int,
    input_height: int = 416,
    input_width: int = 608,
    channels: int = 3
):
    model = fcn_32(n_classes, get_mobilenet_encoder,
                   input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "fcn_32_mobilenet"
    return model


if __name__ == '__main__':
    m = fcn_8(101)
    m = fcn_32(101)
