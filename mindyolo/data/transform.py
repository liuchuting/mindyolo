from typing import List, Dict, Callable

import cv2
import random
import numpy as np
from .perspective import random_perspective
from .dataset import xyxy2xywh, bbox_ioa
from mindyolo.utils.registry import Registry


TRANSFORM_REGISTRY = Registry("transform")


class BaseTransform:
    def __init__(self):
        self.operations = []

        # keys to update
        self.columns_to_update = []  # keys to update

        # keys to add. Note this value is very important to compat with mindspore transform. keys should keep the same
        # order with the newly added key in the __call__ method
        self.columns_to_add = []



@TRANSFORM_REGISTRY.registry_module('hsv_augment')
class HsvAugment(BaseTransform):
    def __init__(self, prob=1.0, hgain=0.5, sgain=0.5, vgain=0.5):
        """
        Hsv augment that transforms an image from rgb to hsv space, multipy the hsv gain and cast beck to rgb.
        Args:
            hgain (float): h gian factor
            sgain (float): s gian factor
            vgain (float): v gian factor
        """
        super(HsvAugment, self).__init__()
        self.prob = prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, image, labels):
        """
        update or add new keys to data dict
        keys to update:  'image', 'labels'
        keys to add:
        Args:
            data (dict): data dict in the pipeline
        """
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # Modify on the original image

        return image, labels

@TRANSFORM_REGISTRY.registry_module('fliplr')
class Fliplr(BaseTransform):
    def __init__(self, prob=0.5):
        super(Fliplr, self).__init__()
        self.prob = prob

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []


    def __call__(self, image, labels):
        """
        update or add new keys to data dict
        keys to update:  'image', 'labels'
        keys to add:
        Args:
            data (dict): data dict in the pipeline
        """
        # flip left-right
        image = np.fliplr(image)
        if len(labels):
            labels[:, 1] = 1 - labels[:, 1]

        return image, labels

@TRANSFORM_REGISTRY.registry_module('flipud')
class Flipud(BaseTransform):
    def __init__(self):
        super(Flipud, self).__init__()

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []


    def __call__(self, image, labels):
        """
        update or add new keys to data dict
        keys to update:  'image', 'labels'
        keys to add:
        Args:
            data (dict): data dict in the pipeline
        """
        # flip up-down
        image = np.flipud(image)
        if len(labels):
            labels[:, 2] = 1 - labels[:, 2]
        return image, labels

@TRANSFORM_REGISTRY.registry_module('label_norm')
class LabelNorm(BaseTransform):
    def __init__(self, xyxy2xywh_=True):
        super(LabelNorm, self).__init__()
        self.xyxy2xywh_ = xyxy2xywh_

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, image, labels):
        """
        update or add new keys to data dict
        keys to update:  'image', 'labels'
        keys to add:
        Args:
            data (dict): data dict in the pipeline
        """
        # flip up-down
        if len(labels) == 0:
            return image, labels

        if self.xyxy2xywh_:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh

        labels[:, [2, 4]] /= image.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= image.shape[1]  # normalized width 0-1

        return image, labels


@TRANSFORM_REGISTRY.registry_module('label_pad')
class LabelPad(BaseTransform):
    def __init__(self, padding_size=160, padding_value=-1):
        super(LabelPad, self).__init__()
        self.padding_size = padding_size
        self.padding_value = padding_value
        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, image, labels):
        # create fixed label, avoid dynamic shape problem.
        labels_out = np.full((self.padding_size, 6), self.padding_value, dtype=np.float32)
        nL = len(labels)
        if nL:
            labels_out[: min(nL, self.padding_size), 0:1] = 0.0
            labels_out[: min(nL, self.padding_size), 1:] = labels[: min(nL, self.padding_size), :]
        return image, labels_out

@TRANSFORM_REGISTRY.registry_module('image_norm')
class ImageNorm(BaseTransform):
    def __init__(self, scale=255.0):
        super(ImageNorm, self).__init__()
        self.scale = scale

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, image, labels):
        # create fixed label, avoid dynamic shape problem.
        image = image.astype(np.float32, copy=False)
        image /= self.scale
        return image, labels

@TRANSFORM_REGISTRY.registry_module('image_transpose')
class ImageTranspose(BaseTransform):
    def __init__(self, bgr2rgb=True, hwc2chw=True):
        super(ImageTranspose, self).__init__()
        self.bgr2rgb = bgr2rgb
        self.hwc2chw = hwc2chw

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, image, labels):
        if self.bgr2rgb:
            image = image[:, :, ::-1]
        if self.hwc2chw:
            image = image.transpose(2, 0, 1)
        return image, labels

@TRANSFORM_REGISTRY.registry_module('pastein')
class Pastein(BaseTransform):
    def __init__(self, prob=0.05, num_sample=30):
        super(Pastein, self).__init__()
        self.prob = prob
        self.num_sample = num_sample

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, image, labels):
        sample_labels, sample_images, sample_masks = [], [], []
        while len(sample_labels) < self.num_sample:
            sample_labels_, sample_images_, sample_masks_ = self.load_samples(random.randint(0, len(self.labels) - 1))
            sample_labels += sample_labels_
            sample_images += sample_images_
            sample_masks += sample_masks_
            if len(sample_labels) == 0:
                break

        # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
        h, w = image.shape[:2]

        # create random masks
        scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
        for s in scales:
            if random.random() < 0.2:
                continue
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            if len(labels):
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            else:
                ioa = np.zeros(1)

            if (
                (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin + 20) and (ymax > ymin + 20)
            ):  # allow 30% obscuration of existing labels
                sel_ind = random.randint(0, len(sample_labels) - 1)
                hs, ws, cs = sample_images[sel_ind].shape
                r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
                r_w = int(ws * r_scale)
                r_h = int(hs * r_scale)

                if (r_w > 10) and (r_h > 10):
                    r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                    r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                    temp_crop = image[ymin : ymin + r_h, xmin : xmin + r_w]
                    m_ind = r_mask > 0
                    if m_ind.astype(np.int).sum() > 60:
                        temp_crop[m_ind] = r_image[m_ind]
                        box = np.array([xmin, ymin, xmin + r_w, ymin + r_h], dtype=np.float32)
                        if len(labels):
                            labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                        else:
                            labels = np.array([[sample_labels[sel_ind], *box]])

                        image[ymin : ymin + r_h, xmin : xmin + r_w] = temp_crop  # Modify on the original image

        return image, labels

@TRANSFORM_REGISTRY.registry_module('random_perspective')
class RandomPerspective(BaseTransform):
    def __init__(self, segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)):
        super(RandomPerspective, self).__init__()
        self.segments = segments
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, image, labels):
        image, labels = random_perspective(
            image,
            labels,
            self.segments,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            border=self.border,
        )

        return image, labels

def transform_adaptor(transform_class: Callable, input_columns: List[str]) -> Callable:
    """
    Adapt mindyolo transform to mindspore transform style. The input and output of __call__ are remade from dict to list
    """
    def run_transform(*data_tuple):
        kwargs = dict(zip(input_columns, data_tuple))
        transformed_dict = transform_class(kwargs)
        return tuple(transformed_dict.values)
    return run_transform


def create_transforms(transform_pipeline: List[Dict], input_columns: List[str], global_config: Dict = None):
    """
    Create a sequence of callable transforms
    Args:
        transform_pipeline (List[Dict]): list of dict containing transform name and its args. eg:
               >>> [{ 'func_name': 'hsv_augment', 'prob': 1.0, 'hgain': 0.015, 'sgain': 0.7, 'vgain': 0.4 },
               >>>  { 'func_name': 'label_norm', 'xyxy2xywh_': True }]
        input_columns (List[str]): list of input columns to the data pipeline
    Returns:
        transform_list (List): list of dict with transforms that compatible with mindspore dataset.map function
    """
    assert isinstance(transform_pipeline, (list, tuple))
    assert isinstance(input_columns, (list, tuple))
    for tp in transform_pipeline:
        assert isinstance(tp, dict), f'expect transform to be a dict but got {type(tp)} instead'
        assert 'func_name' in tp, 'transform must have its func_name'

    input_columns, output_columns = input_columns.copy(), input_columns.copy()
    transform_list = []
    for trans_config in transform_pipeline:
        trans_name = trans_config['func_name']
        # mixup and mosaic are performed in dataset
        if trans_name in ['mosaic', 'mixup', 'letterbox', 'pastein']:
            continue

        kwargs = {k: v for k, v in trans_config.items() if k != 'func_name'}
        if global_config is not None:
            kwargs.update(global_config)
        trans_class = TRANSFORM_REGISTRY.get(trans_name)
        output_columns.extend([k for k in trans_class().columns_to_add if k not in output_columns])
        # adapted_trans_class = transform_adaptor(trans_class, input_columns)
        # trans_dict = dict(
        #     operations=[adapted_trans_class],  # make a list to compat with mindspore dateset.map
        #     input_columns=input_columns,
        #     output_columns=output_columns,
        # )
        trans_dict = dict(
            operations=trans_class(**kwargs),  # make a list to compat with mindspore dateset.map
            input_columns=input_columns[:2],
            output_columns=output_columns[:2],
        )

        transform_list.append(trans_dict)
        input_columns = output_columns.copy()
    return transform_list
