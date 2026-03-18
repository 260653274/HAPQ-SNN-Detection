#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os

from yolox.exp import EventExp as MyEventExp


class Exp(MyEventExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.max_epoch = 60
        self.enable_hapq = True
        self.hapq_mode = "full_nas"
        self.hapq_lambda_dsp = 0.05  # Lower penalty to allow larger models (default might be higher)
        self.hapq_lambda_bram = 0.05
        self.data_name = "gen1"
        self.num_classes = 2
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import EventTrainTransform, GEN1Dataset, GEN1_CLASSES
        slice_args = self.get_slice_args()
        
        # Update paths to match the nested directory structure
        train_path = os.path.join(self.data_dir, 'train', 'detection_dataset_duration_60s_ratio_1.0', 'train')
        val_path = os.path.join(self.data_dir, 'val', 'detection_dataset_duration_60s_ratio_1.0', 'val')
        data_dir = [train_path, val_path]
        
        return GEN1Dataset(data_path=data_dir, class_names=GEN1_CLASSES, input_size=self.input_size,
                           random_aug=True, target_transform=EventTrainTransform(box_norm=False),
                           **slice_args)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import EventValTransform, GEN1Dataset, GEN1_CLASSES
        testdev = kwargs.get("testdev", False)
        slice_args = self.get_slice_args()
        
        # Update path to match the nested directory structure
        data_path = os.path.join(self.data_dir, 'test', 'detection_dataset_duration_60s_ratio_1.0', 'test')

        return GEN1Dataset(data_path=data_path, class_names=GEN1_CLASSES,
                           input_size=self.input_size, map_val=True, letterbox_image=True, format='xywh',
                           random_aug=False, target_transform=EventValTransform(box_norm=False), cache_path='ram',
                           **slice_args)
