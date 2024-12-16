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

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time

from torch_fidelity.metric_fid import calculate_fid

import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util


# # Workaround
# try:
#     import ctypes
#     libgcc_s = ctypes.CDLL('libgcc_s.so.1')
# except:
#     pass


from guided_diffusion.script_util import (
    # NUM_CLASSES,
    model_and_diffusion_defaults,
    # classifier_defaults,
    create_model_and_diffusion,
    # create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    """
    将张量转换为 8 位无符号整数（uint8）格式的 NumPy 数组，表示处理后的图像数据
    """
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

# def load_custom_image(image_path, image_size, batch_size):
#     """
#     加载一张图片做ref_img
#     """
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 转换为 [-1, 1]
#     ])
#     img = Image.open(image_path).convert("RGB")
#     img_tensor = transform(img).unsqueeze(0)  # 增加 batch 维度
#     img_tensor = img_tensor.repeat(batch_size, 1, 1, 1)  # 扩展到 batch_size
#     return img_tensor


def main(conf: conf_mgt.Default_Conf):

    # print("Start........", conf['name']) # 'name'就是confs里的"name"
    print("Start........")
    # device = dist_util.dev(conf.get('device')) 这句和下面这句效果一样
    device = dist_util.dev(None)

    # model 专注于学习如何从噪声预测图像。
    # diffusion 专注于扩散过程的算法设计和流程控制。
    model, diffusion = create_model_and_diffusion(
        # 从 conf 中提取所有 键在model_and_diffusion_defaults().keys() 中的键值对
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )


    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")  # checkpoint地址conf.model_path
    )

    model.to(device)


    model.eval() # repaint只是推理

    show_progress = conf.show_progress
    cond_fn = None

    print("sampling...")

    dset = 'eval'
    eval_name = conf.get_default_eval_name() # # 获取.yml里的 data字典中的 eval字典中的 键的名字
                                             # eval_name='paper_face_mask'
    # ###### dataloader ######################################### #
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)
    # ###### dataloader ######################################### #

    for batch in iter(dl):

        for k in batch.keys():
            # 把dl中是Tensor的放到device
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}
        model_kwargs["gt"] = batch['GT'] # gt的相关kwargs

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask # mask二值图的相关kwargs

        batch_size = model_kwargs["gt"].shape[0]

        if conf.use_ref_imgs:
            # ref_img = load_custom_image(r"00003.png", image_size=256, batch_size=batch_size)
            # model_kwargs["ref_img"] = ref_img.to(device)
            model_kwargs["ref_img"] = model_kwargs["gt"] # 参考图片就弄成真实图片
            from resizer import Resizer
            shape = (batch_size, 3, conf.image_size, conf.image_size)
            shape_d = (batch_size, 3, int(conf.image_size / conf.down_N), int(conf.image_size / conf.down_N))
            # print(f"shape: {shape}, shape_d: {shape_d}")
            down = Resizer(shape, 1 / conf.down_N).to(next(model.parameters()).device)
            up = Resizer(shape_d, conf.down_N).to(next(model.parameters()).device)
            resizers = (down, up)
        else:
            resizers = None

        if not conf.use_ddim:
            print("test.py--- ddpm --")
        else:
            print("test.py--- ddim --")

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        def model_fn(x, t, y=None, gt=None, **kwargs):
            # assert y is not None
            # print(y) # y目前就是none
            return model(x, t, y if conf.class_cond else None, gt=gt)

        result = sample_fn(
            model_fn, # 上面不远有个函数就叫model_fn
            (batch_size, 3, conf.image_size, conf.image_size), # 传递给 shape 参数的元组
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            resizers=resizers,
            conf=conf
        )

        srs = toU8(result['sample']) # srs是inpainted
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') +
                   (-1) * th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask'))) #lrs是gt_masked
        gts = toU8(result['gt'])     # gts是gt
        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1)) #gt_keep_masks是gt_keep_mask

        # conf.eval_imswrite(
        #     srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
        #     img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

        # # # gt_keep_masks()就先不看了
        # conf.eval_imswrite(
        #     srs=srs, gts=gts, lrs=lrs,
        #     img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

        # gt和gt_keep_masks就先不看了
        conf.eval_imswrite(
            srs=srs, lrs=lrs,img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread('confs/face_example1.yml'))
    main(conf_arg)
