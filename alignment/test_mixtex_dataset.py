import os
import sys
import numpy as np
sys.path.append('D:/study/dl/MixTex')

from PaddleOCR.ppocr.data.mixtex_dataset import MixTexDataSet
from PaddleOCR.tools.program import load_config
from ppocr.utils.logging import get_logger


logger = get_logger()


config_path = "D:/study/dl/MixTex/PaddleOCR/configs/rec/MixTex.yml"
config = load_config(file_path=config_path)
mixtex_ds = MixTexDataSet(config=config, mode='Train', logger=logger)
img, label = mixtex_ds[0]

origin_img_path = 'D:/study/dl/MixTex/data/exp/alignment/after_preprocessor_img.npy'
origin_label_path = 'D:/study/dl/MixTex/data/exp/alignment/label.npy'

origin_label = np.load(origin_label_path)

assert len(label) == len(origin_label), f'len(label): {len(label)}, origin_label: {len(origin_label)}'
assert label == [0, 203, 262, 264, 22, 265, 5850, 989, 1912, 1150, 319, 437, 95, 46, 348, 781, 13768, 1519, 1677, 291, 7452, 3160, 5698, 412, 2659, 1756, 7747, 10865, 11657, 388, 264, 29, 2183, 265, 225, 8516, 11657, 203, 311, 225, 203, 64, 619, 95, 1638, 656, 654, 15, 77, 64, 520, 64, 676, 64, 520, 325, 874, 82, 15, 21, 5137, 20, 64, 1919, 64, 676, 651, 64, 677, 700, 3531, 654, 15, 77, 64, 520, 64, 676, 64, 520, 325, 23518, 87, 3363, 677, 700, 10, 34, 20, 283, 64, 266, 64, 676, 651, 345, 534, 95, 1638, 97, 203, 312, 225, 203, 8877, 1756, 1674, 291, 12920, 225, 264, 74, 58, 265, 554, 7089, 1549, 554, 7089, 7268, 1747, 631, 554, 5181, 4505, 1586, 6576, 2759, 13768, 1519, 1677, 303, 4220, 1747, 631, 554, 5181, 4505, 1586, 264, 21, 265, 781, 9022, 1677, 291, 1225, 13768, 1519, 7430, 631, 2612, 924, 3425, 303, 9450, 1912, 1070, 203, 262, 25678, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]

assert np.array_equal(origin_label, np.array(label))

origin_img = np.load(origin_img_path)
assert np.array_equal(origin_img, img)

