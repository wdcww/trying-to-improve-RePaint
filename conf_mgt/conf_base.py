# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

from functools import lru_cache
import os
import torch
from utils import imwrite
from collections import defaultdict
from os.path import isfile, expanduser

def to_file_ext(img_names, ext):
    """
    该函数接受一个图像文件名列表 (img_names) 和一个文件扩展名 (ext) 作为参数。
    构造新的文件名（将原文件名的扩展名替换为 ext），并将其添加到 img_names_out 列表中。
    """
    img_names_out = []
    for img_name in img_names:
        splits = img_name.split('.')
        if not len(splits) == 2:
            raise RuntimeError("File name needs exactly one '.':", img_name)
        img_names_out.append(splits[0] + '.' + ext)

    return img_names_out

def write_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        imwrite(img=image, path=out_path)


class NoneDict(defaultdict):
    """
    此类继承自defaultdict
    defaultdict 是 Python 标准库 collections 模块中的一个字典子类。
    与普通字典不同，defaultdict 允许你为字典定义一个默认值，
    当访问一个不存在的键时，它会自动使用这个默认值，而不是抛出 KeyError
    """
    def __init__(self):
        super().__init__(self.return_None)

    # 使用self.return_None作为默认工厂函数，如果访问的键不存在，返回None，而不是抛出KeyError
    @staticmethod
    def return_None():
        return None

    def __getattr__(self, attr):
        return self.get(attr)


# ############### 以上的三个东西都为下面这个东西服务 ###################

class Default_Conf(NoneDict):
    def __init__(self):
        pass

    def get_dataloader(self, dset='train', dsName=None, batch_size=None, return_dataset=False):
        """
        参数:
        dset：指定数据集的类型，默认为 'train'（训练集）。
        dsName：数据集的名称，默认为 None。
        batch_size：批处理大小，默认为 None。
        return_dataset：是否返回数据集对象本身，默认为 False。
        """

        # if batch_size is None:
        #     batch_size = self.batch_size
        # print("batch_size：",end='')
        # print(batch_size)
        # # batch_size会从下面的load_data_inpa(**ds_conf, conf=self)去解包.yml文件的

        candidates = self['data'][dset]
        ds_conf = candidates[dsName].copy()
        # ds_conf拿到了 那内层的 14个 key-value
        if ds_conf.get('mask_loader', False):
            # 从 ds_conf 字典中获取键 'mask_loader' 对应的值。如果该键不存在，则返回默认值 False
            from guided_diffusion.image_datasets import load_data_inpa
            return load_data_inpa(**ds_conf, conf=self)
        else:
            raise NotImplementedError()

    # def get_debug_variance_path(self): # 这个没有用到，先注释起来
    #     return os.path.expanduser(os.path.join(self.get_default_eval_conf()['paths']['root'], 'debug/debug_variance'))

    @ staticmethod
    def device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def eval_imswrite(self, srs=None, img_names=None, dset=None, name=None, ext='png', lrs=None, gts=None, gt_keep_masks=None, verify_same=True):
        """
        用于将图像数据写入指定的目录
        """
        img_names = to_file_ext(img_names, ext)

        # if dset is None: # 如果没有说明数据集的形式是train还是eval，那么就按数据集名字处理
        #     dset = self.get_default_eval_name()

        # max_len = self['data'][dset][name].get('max_len') # 这个没有用到，先注释起来

        if srs is not None:
            # # srs是inpainted
            sr_dir_path = expanduser(self['data'][dset][name]['paths']['srs'])
            write_images(srs, img_names, sr_dir_path)

        if gt_keep_masks is not None:
            # #gt_keep_masks是 掩码图
            mask_dir_path = expanduser(
                self['data'][dset][name]['paths']['gt_keep_masks'])
            write_images(gt_keep_masks, img_names, mask_dir_path)

        gts_path = self['data'][dset][name]['paths'].get('gts')
        if gts is not None and gts_path:
            # # gts是gt
            gt_dir_path = expanduser(gts_path)
            write_images(gts, img_names, gt_dir_path)

        if lrs is not None:
            # #lrs是gt_masked,加了掩码的gt图
            lrs_dir_path = expanduser(
                self['data'][dset][name]['paths']['lrs'])
            write_images(lrs, img_names, lrs_dir_path)

    def get_default_eval_name(self):

        candidates = self['data']['eval'].keys()
        if len(candidates) != 1:
            raise RuntimeError(
                f"Need exactly one candidate for {self.name}: {candidates}")
        return list(candidates)[0]

    # def pget(self, name, default=None):
    #     if '.' in name:
    #         names = name.split('.')
    #     else:
    #         names = [name]
    #
    #     sub_dict = self
    #     for name in names:
    #         sub_dict = sub_dict.get(name, default)
    #
    #         if sub_dict == None:
    #             return default
    #
    #     return sub_dict
