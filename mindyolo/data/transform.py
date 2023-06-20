from typing import List, Dict, Callable

import cv2
import numpy as np

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
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        """
        Hsv augment that transforms an image from rgb to hsv space, multipy the hsv gain and cast beck to rgb.
        Args:
            hgain (float): h gian factor
            sgain (float): s gian factor
            vgain (float): v gian factor
        """
        super(HsvAugment, self).__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

        # keys to update
        self.columns_to_update = ['image', 'labels']  # keys to update

        # new keys to add that are not in the input data dict.
        # Note this value is very important to compat with mindspore transform. keys should keep the same order with the
        # newly added key in the __call__ method
        self.columns_to_add = []

    def __call__(self, data: dict):
        """
        update or add new keys to data dict
        keys to update:  'image', 'labels'
        keys to add:
        Args:
            data (dict): data dict in the pipeline
        """
        image, labels = data['image'], data['labels']
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # Modify on the original image

        data['image'] = image
        data['labels'] = labels
        return data


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
        if trans_name in ['mosaic', 'mixup']:
            continue

        kwargs = {k: v for k, v in trans_config if k != 'func_name'}
        if global_config is not None:
            kwargs.update(global_config)
        trans_class = TRANSFORM_REGISTRY.get(trans_name)
        output_columns = output_columns.extend([k for k in trans_class.columns_to_add if k not in output_columns])

        adapted_trans_class = transform_adaptor(trans_class, input_columns)
        trans_dict = dict(
            operations=[adapted_trans_class],  # make a list to compat with mindspore dateset.map
            input_columns=input_columns,
            output_columns=output_columns,
        )

        transform_list.append(trans_dict)
        input_columns = output_columns.copy()
    return transform_list
