# alignment preprocess

# add dependencies path
import sys
sys.path.append('D:/study/dl/MixTex/MixTeX-Latex-OCR')
sys.path.append('D:/study/dl/MixTex/PaddleOCR')

import os
from PIL import Image
import numpy as np


def test_preprocess():
    imgen = Image.open("D:/study/dl/MixTex/MixTeX-Latex-OCR/demo/zh_test.png")
    img = np.array(imgen)

    # load array before preprocess
    origin_img_path = os.path.join('data/exp/alignment/before_resize.npy')
    origin_img = np.load(origin_img_path)

    assert (np.array_equal(img, origin_img))
