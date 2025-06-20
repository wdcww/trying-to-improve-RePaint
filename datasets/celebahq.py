import os
import pandas as pd
from PIL import Image
import numpy as np
import torch

from datasets.masks import generate_half_mask, mask_generators
from datasets.utils import normalize, _list_image_files_recursively


def load_celebahq(
    train=False, valid=False, last_100=True, shape=(256, 256), mask_type="half"
):
    assert mask_type in mask_generators, "Not Implemented for Others."

    dir_data = os.path.join(os.getcwd(), "datasets", "celebahq")
    train_split = pd.read_csv(
        os.path.join(dir_data, "celebahqvalidation.txt"), header=None
    ).values
    valid_split = pd.read_csv(
        os.path.join(dir_data, "celebahqtrain.txt"), header=None
    ).values

    def load_image(idx): return normalize(
        Image.open(os.path.join(dir_data, "CelebA-HQ-img", "%d.jpg" % idx)), shape=shape
    )
    mask_generator = mask_generators[mask_type]

    def load_image_mask_name(idx): return (
        load_image(idx),
        mask_generator(shape),
        "%05d" % idx,
    )

    tr_ds, val_ds, last_100_ds = None, None, None
    if train:
        tr_ds = [load_image_mask_name(int(i[0][5:10])) for i in train_split]
    if valid:
        val_ds = [load_image_mask_name(int(i[0][5:10])) for i in valid_split]
    if last_100:
        last_100_ds = (
            val_ds[-100:]
            if valid
            else [load_image_mask_name(int(i[0][5:10])) for i in valid_split[-100:]]
        )

    return tr_ds, val_ds, last_100_ds

def load_custom_dataset(
    image_dir, single_mask_path, shape=(256, 256), max_len=None,name=None
):
    """

      Args:
        image_dir: 文件夹路径，里面存放你自己的图片（任意大小）
        single_mask_path: 一张提前准备好的 mask，会重复使用
        shape: 输出图像和 mask 的目标尺寸
        max_len: 限制加载多少张图（None 表示全加载）
        name:

      Returns:
        List of (image_tensor, mask_tensor, name)

    """
    image_paths = sorted(_list_image_files_recursively(image_dir))
    if max_len is not None:
        image_paths = image_paths[:max_len]

    def load_image(path):
        # 由 normalize 自动 resize + 归一化 [-1, 1]
        return normalize(Image.open(path), shape=shape)

    def load_mask(path):
        # resize + 归一化到 [0, 1]，并加一个 channel 维度 (1, H, W)
        mask = Image.open(path).resize(shape).convert("L")
        mask = np.array(mask, dtype=np.float32) / 255.0
        return torch.from_numpy(mask).unsqueeze(0)

    # 只加载一张共享 mask
    shared_mask = load_mask(single_mask_path)

    data = []
    for img_path in image_paths:
        image = load_image(img_path)
        # name = os.path.splitext(os.path.basename(img_path))[0]
        name = name
        data.append((image, shared_mask.clone(), name))  # clone 避免 inplace 修改

    return data

def load_lama_celebahq(
    offset=0,
    max_len=100,
    shape=(256, 256),
    mask_type="half",
):
    """Load first 100 images in lama celeba test set"""
    gt_dir = os.path.join(
        os.getcwd(), "./datasets/lama-celeba/visual_test_source_256/")
    gt_paths = _list_image_files_recursively(gt_dir)
    gt_paths.sort()

    def load_image(path): return normalize(Image.open(path), shape=shape)
    if mask_type not in ["narrow", "wide"]:
        # simple masks
        mask_generator = mask_generators[mask_type]

        def load_image_mask_name(path): return (
            load_image(path),
            mask_generator(shape),
            "%05d" % int(os.path.splitext(os.path.basename(path))[0]),
        )
        res = [
            load_image_mask_name(path) for path in gt_paths[offset: offset + max_len]
        ]
    else:
        mask_dir = os.path.join(
            os.getcwd(), f"datasets/Repaint_mask/{mask_type}")
        mask_paths = _list_image_files_recursively(mask_dir)
        mask_paths.sort()

        def load_mask(path): return torch.from_numpy(
            np.array(Image.open(path).resize(
                shape).convert("L"), dtype=np.float32)
            / 255.0,
        )

        def load_image_mask_name(i): return (
            load_image(gt_paths[i]),
            load_mask(mask_paths[i]),
            "%05d" % int(os.path.splitext(os.path.basename(gt_paths[i]))[0]),
        )
        res = [load_image_mask_name(i) for i in range(offset, offset + max_len)]
    return res
