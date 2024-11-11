#coding=utf8, Wenyu Qiu
import os
import cv2
import json
import lmdb
import logging
import numpy as np
from tools.program import load_config
from paddle.io import Dataset
from ppocr.data.imaug import transform


class LMDBDealer(Dataset):
    def __init__(self, data_dir, logger):
        super().__init__()
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        logger.info("Initialize indexs of datasets:%s" % data_dir)
        self.data_idx_order_list = self.dataset_traversal()

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + "/"):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                txn = env.begin(write=False)
                num_samples = int(txn.get("num-samples".encode()))
                lmdb_sets[dataset_idx] = {
                    "dirpath": dirpath,
                    "env": env,
                    "txn": txn,
                    "num_samples": num_samples,
                }
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]["num_samples"]
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]["num_samples"]
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype="uint8")
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_lmdb_sample_info(self, txn, index):
        label_key = "label-%09d".encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode("utf-8")
        img_key = "image-%09d".encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {"image": img, "label": label}
        outs = transform(data, None)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]


if __name__ == "__main__":
    logger = logging.getLogger(name=__name__)
    # db_fp = "F:/nets/OCR/ocr_optimize/data/DTRB/evaluation/CUTE80"
    db_fp = "F:/nets/OCR/ocr_optimize/data/DTRB/ST_spe"
    lmdb_dealer = LMDBDealer(data_dir=db_fp, logger=logger)
    # value = lmdb_dealer[0]
    # print(value['label'])
    # img = lmdb_dealer.get_img_data(value['image'])
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    print(len(lmdb_dealer.data_idx_order_list))