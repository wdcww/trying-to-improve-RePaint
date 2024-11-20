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
    # print(model)

    model.to(device)

    # if conf.use_fp16: # face_example.yml中，这里是false
    #     model.convert_to_fp16()

    model.eval() # repaint只是推理

    show_progress = conf.show_progress

    # print('test.py conf.classifier_path: ',end='')
    # print(conf.classifier_path)
    cond_fn = None # 这里cond_fn直接是下面这个if-else的else的结果
    # if conf.classifier_scale > 0 and conf.classifier_path:
    #     print("loading classifier...")
    #     classifier = create_classifier(
    #         **select_args(conf, classifier_defaults().keys()))
    #     classifier.load_state_dict(
    #         dist_util.load_state_dict(os.path.expanduser(
    #             conf.classifier_path), map_location="cpu")
    #     )
    #
    #     classifier.to(device)
    #     if conf.classifier_use_fp16:
    #         classifier.convert_to_fp16()
    #     classifier.eval()
    #
    #     def cond_fn(x, t, y=None, gt=None, **kwargs):
    #         assert y is not None
    #         with th.enable_grad():
    #             x_in = x.detach().requires_grad_(True)
    #             logits = classifier(x_in, t)
    #             log_probs = F.log_softmax(logits, dim=-1)
    #             selected = log_probs[range(len(logits)), y.view(-1)]
    #             return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    # else:
    #     cond_fn = None

    # def model_fn(x, t, y=None, gt=None, **kwargs):
    #     """
    #     这个函数允许在条件生成任务中根据需求进行不同的设置，
    #     比如是否使用类别信息来指导生成过程。
    #
    #     根据 conf.class_cond的值来决定是否将 y 参数传递给 model 函数
    #     conf.class_cond 为 True，则传递 y
    #     如果conf.class_cond为 False，则传递 None
    #     x：输入张量，通常是模型的输入数据（如噪声或先前生成的图像）。
    #     t：时间步长，通常用于扩散模型中的时间编码。
    #     y：类别标签或条件信息（如果有的话）。
    #     gt：真实图像（如果有的话），通常用于条件生成任务中的损失计算。
    #     **kwargs：其他额外的关键字参数
    #     """
    #     # assert y is not None
    #     return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    # all_images = [] # 没有用到,就先注释掉

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
        # print("test.py --batch_size : ",batch_size)

        # if conf.cond_y is not None: # conf.cond_y是None
        #     classes = th.ones(batch_size, dtype=th.long, device=device)
        #     model_kwargs["y"] = classes * conf.cond_y # 使用固定的类标签
        # else:
        #     pass
        #     # classes = th.randint(
        #     #     low=0, high=NUM_CLASSES, size=(batch_size,), device=device
        #     # ) # 一个大小为batch_size随机整数张量classes,值的范围在0到NUM_CLASSES之间
        #     # model_kwargs["y"] = classes # 使用随机生成的类标签


        if not conf.use_ddim:
            print("test.py--- ddpm --")
        else:
            print("test.py--- ddim --")

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        def model_fn(x, t, y=None, gt=None, **kwargs):
            """
            这个函数允许在条件生成任务中根据需求进行不同的设置，
            比如是否使用类别信息来指导生成过程。

            根据 conf.class_cond的值来决定是否将 y 参数传递给 model 函数
            conf.class_cond 为 True，则传递 y
            如果conf.class_cond为 False，则传递 None
            x：输入张量，通常是模型的输入数据（如噪声或先前生成的图像）。
            t：时间步长，通常用于扩散模型中的时间编码。
            y：类别标签或条件信息（如果有的话）。
            gt：真实图像（如果有的话），通常用于条件生成任务中的损失计算。
            **kwargs：其他额外的关键字参数
            """
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
            conf=conf
        )

        srs = toU8(result['sample']) # srs是inpainted
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') +
                   (-1) * th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask'))) #lrs是gt_masked
        # gts = toU8(result['gt'])     # gts是gt
        # gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1)) #gt_keep_masks是gt_keep_mask

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




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--conf_path', type=str, required=False, default=None)
#     args = vars(parser.parse_args())
#
#     conf_arg = conf_mgt.conf_base.Default_Conf()
#     conf_arg.update(yamlread(args.get('conf_path')))
#     main(conf_arg)


if __name__ == "__main__":

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread('confs/face_example1.yml'))
    main(conf_arg)
