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

import random
import os

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_data_yield(loader):
    """
    这个函数的主要用途是在需要连续获取数据的场景中，不断地从给定的loader中获取数据,
    yield from loader 会逐个返回 loader 中的每个数据项，直到 loader 迭代完毕.

    例如: 在训练过程中，当你希望在每个 epoch 中循环使用数据集而不需要重新初始化数据加载器时，它会非常方便。
    使用这个函数时，可以通过 next() 来获取下一个数据批次，而不会因为数据加载器迭代完毕而停止。
    这样可以有效地管理内存并提高数据读取的灵活性。
    """
    while True:
        yield from loader

def load_data_inpa(
    *,
    gt_path=None,
    mask_path=None,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    return_dataloader=False,
    return_dict=False,
    max_len=None,
    drop_last=True,
    conf=None,
    offset=0,
    ** kwargs
):
    """
    对于数据集，创建一个基于（图像、kwargs）对的生成器。
    每个图像都是一个 NCHW 浮点张量，并且 kwargs 字典包含零个或多个键，每个键都映射到自己的批处理张量。
    kwargs字典可用于类标签，在这种情况下键是“y”值是类标签的整数张量。

    data_dir: 数据集目录。
    batch_size: 每个返回对的批量大小。
    image_size: 图像调整后的大小。
    class_cond：如果为 True，则在类的返回字典中包含“y”键标签。如果类不可获取但还是True，则将引发异常。
    deterministic：如果为True，则产生确定性顺序的结果。
    random_crop: 如果为 True，则随机裁剪图像以进行增强。
    random_flip: 如果为 True，则随机翻转图像以进行增强。
    return_dataloader: 如果为False,那么使用load_data_yield(loader) 如果为True,直接返回loader
    drop_last: DataLoader的参数，当数据集中的样本数量不能被 batch_size 整除时，是否丢弃最后一个不足完整批次的样本。
    offset: 如果数据集需要动态划分，offset提供了一种简单的实现方式。例如：offset=2 时，跳过前两个数据，从第三个数据开始加载
    """

    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)

    gt_paths = _list_image_files_recursively(gt_dir)
    mask_paths = _list_image_files_recursively(mask_dir)

    # 如果只有一张mask图，重复使用
    if len(mask_paths) == 1 and len(gt_paths) > 1:
        mask_paths = [mask_paths[0]] * len(gt_paths)

    assert len(gt_paths) == len(mask_paths)

    classes = None
    if class_cond:
        raise NotImplementedError()

    dataset = ImageDatasetInpa(
        image_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        return_dict=return_dict,
        max_len=max_len,
        conf=conf,
        offset=offset
    )
    # deterministic 也就是决定 shuffle
    if deterministic:
        # True时不打乱,shuffle=False
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=drop_last
        )

    else:
        # False时打乱,shuffle=True
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last
        )

    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(
        self,
        resolution,
        gt_paths,
        mask_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        return_dict=False,
        max_len=None,
        conf=None,
        offset=0
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]

        self.local_gts = gt_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]

        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len

    # def __len__(self):
    #     if self.max_len is not None:
    #         return self.max_len
    #
    #     return len(self.local_gts)

    def __len__(self):
        # 如果 max_len 是 0，则返回 len(self.local_gts)
        if self.max_len == 0:
            return len(self.local_gts)
        # 否则，返回 max_len 的值
        return self.max_len

    def __getitem__(self, idx):
        """
        读取、处理和返回图像及其掩码,
        返回一个字典:
        {
        'GT': arr_gt,          # 处理后的真实图像
        'GT_name': name,      # 真实图像的文件名
        'gt_keep_mask': arr_mask,  # 处理后的掩码
        'y': 类标签 (可选)    # 类标签 (_____________________这里还没有加上_____________________)
        }
        """
        gt_path = self.local_gts[idx]
        pil_gt = self.imread(gt_path)

        mask_path = self.local_masks[idx]
        pil_mask = self.imread(mask_path)

        if self.random_crop:
            raise NotImplementedError()
        else:
            arr_gt = center_crop_arr(pil_gt, self.resolution)
            arr_mask = center_crop_arr(pil_mask, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_mask = arr_mask[:, ::-1]

        arr_gt = arr_gt.astype(np.float32) / 127.5 - 1
        arr_mask = arr_mask.astype(np.float32) / 255.0

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.return_dict:
            name = os.path.basename(gt_path)
            return {
                'GT': np.transpose(arr_gt, [2, 0, 1]),
                'GT_name': name,
                'gt_keep_mask': np.transpose(arr_mask, [2, 0, 1]),
            }
        else:
            raise NotImplementedError()

    def imread(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
