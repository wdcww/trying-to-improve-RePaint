import torch
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


def normalize_image(tensor_img):
    tensor_img = (tensor_img + 1.0) / 2.0
    return tensor_img


def save_grid(tensor_img, path, nrow=5):
    """
    tensor_img: [B, 3, H, W] or [tensor(3, H, W)]
    """
    if isinstance(tensor_img, list):
        tensor_img = torch.stack(tensor_img)
    assert len(tensor_img.shape) == 4
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    grid = make_grid(tensor_img, nrow=nrow)
    pil = ToPILImage()(grid)
    pil.save(path)


def save_image(tensor_img, path):
    """
    tensor_img : [3, H, W]
    """
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    pil = ToPILImage()(tensor_img)
    pil.save(path)


"""
txtread(path)：读取指定路径的文本文件内容，并返回为字符串。
yamlread(path)：调用 txtread 函数读取 YAML 文件，并使用 yaml.safe_load 解析其内容，返回为 Python 对象。
imwrite(path, img)：将给定的图像数组（img）保存为指定路径的图像文件。
"""
import yaml
import os
from PIL import Image

def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))


def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)