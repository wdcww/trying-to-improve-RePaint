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

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    num_timesteps 是 .yml的diffusion_steps
    section_counts 是 .yml的 timestep_respacing
    """
    # section_counts 是 str
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            # section_counts 以 "ddim" 开头的字符串时
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
        else:
            # 而section_counts 不是以 "ddim" 开头的字符串时
            # 下面这句程序，会将其视为一个以逗号分隔的整数列表
            section_counts = [int(x) for x in section_counts.split(",")]

    # section_counts 是 int
    if isinstance(section_counts, int):
        section_counts = [section_counts]

    # 如果能到这行,section_counts终究是个list,里面元素是int
    size_per = num_timesteps // len(section_counts) # size_per 是每个部分的基准大小
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []

    if len(section_counts) == 1 and section_counts[0] > num_timesteps:
        # 在section_counts列表 中 唯一的数字还比num_timesteps大，
        # 直接返回 section_counts[0] 个 从0~num_timesteps的等间距数值
        return set(np.linspace(start=0, stop=num_timesteps, num=section_counts[0]))

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                # 表示不能将当前部分(大小size)划分为 section_count步
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            # 如果 section_count 为 1 或更小，
            # 意味着当前部分只需要一个时间步，所以步长为 1
            frac_stride = 1
        else:
            # 否则，计算分配给每个步骤的步长 frac_stride
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = [] # 准备存储 当前部分 的时间步索引
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return all_steps
    # return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, conf=None, **kwargs):
        self.use_timesteps = set(use_timesteps) # space_timesteps()使用timestep_respacing压缩的
        self.original_num_steps = len(kwargs["betas"]) # 和.yml的diffusion_steps相等的1000
        self.conf = conf

        base_diffusion = GaussianDiffusion(conf=conf,**kwargs)  # pylint: disable=missing-kwoa

        if conf.respace_interpolate:
            pass
            # print("respace.py--111")
            # new_betas = resample_betas( kwargs["betas"], int(conf.timestep_respacing) )
            # self.timestep_map = list(range(len(new_betas)))
        else:
            # print("respace.py--222 ,conf.respace_interpolate: ",end='')
            # print(conf.respace_interpolate)
            # print("respace.py--self.original_num_steps  ",self.original_num_steps)

            self.timestep_map = []
            new_betas = []
            last_alpha_cumprod = 1.0
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            # print("respace.py--self.timestep_map( 此list可追溯至respace.py的space_timesteps()函数 )   ",self.timestep_map)
            # print("self.timestep_map的长度  ",len(self.timestep_map) )
            # print("respace.py--( 长度 ) : new_betas   (", len(new_betas),") : ",new_betas)
            kwargs["betas"] = np.array(new_betas)

        # if conf.use_value_logger:
        #     conf.value_logger.add_value(
        #         new_betas, 'new_betas SpacedDiffusion')

        super().__init__(conf=conf, **kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model),*args, **kwargs)
        # return super().p_mean_variance(self.model, *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps,
            self.original_num_steps, self.conf
        )
    # def training_losses(
    #     self, model, *args, **kwargs
    # ):  # pylint: disable=signature-differs
    #     return super().training_losses(self._wrap_model(model), *args, **kwargs)
    #
    # def condition_mean(self, cond_fn, *args, **kwargs):
    #     return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)
    #
    # def condition_score(self, cond_fn, *args, **kwargs):
    #     return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)


    # def _scale_timesteps(self, t):
    #     # Scaling is done by the wrapped model.
    #     return t


class _WrappedModel:
    """
    这个 _WrappedModel 类的功能是包装（封装）一个已有的模型，并对输入的时间步 (ts) 和模型的行为进行处理，
    具体来说，它的作用是:
    在Diffusion Models 推理时, 对时间步进行变换或重映射。
    """
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps, conf):
        self.model = model
        self.timestep_map = timestep_map
        # print("class _WrappedModel----- self.timestep_map   ",self.timestep_map)
        self.rescale_timesteps = rescale_timesteps # if True,then NotImplementedError.
        self.original_num_steps = original_num_steps
        self.conf = conf

    def __call__(self, x, ts, **kwargs):
        # print("class _WrappedModel----- x  ", x) # 这里的x是某时刻的一个状态图!

        map_tensor = th.tensor(  self.timestep_map, device=ts.device, dtype=ts.dtype )
        # self.timestep_map涉及函数space_timesteps()，且与 .yml的diffusion_steps 和 timestep_respacing 有关。
        # ts 是某个源于scheduler.py的 “真”时间步。

        new_ts = map_tensor[ts] # 取出map_tensor在索引ts对应位置的值
        # print("respace.py class _WrappedModel 给model传入的t  ",new_ts)
        # print("                               你的t_T中的t  ", ts)
        return self.model(x, new_ts, **kwargs) # # # # # # # # # # # # # # 在这里给模型传入 # # # # # # # # #

        # a=self.model(x, new_ts, **kwargs)
        # print("a= ",a)
        # return a



        # if self.rescale_timesteps:
        #     # # 如果在.yml的rescale_timesteps为true,那么就可以raise下面的Error了,
        #     raise NotImplementedError()
        #     # new_ts = self.do_rescale_timesteps(new_ts)
        # if self.conf.respace_interpolate:
        #     print("respace.py--333")
        #     new_ts = new_ts.float() * (
        #         (self.conf.diffusion_steps - 1) / (float(self.conf.timestep_respacing) - 1.0))

        # return self.model(x, new_ts, **kwargs)

    # def do_rescale_timesteps(self, new_ts):
    #     """
    #     一个被Repaint放弃实现的函数
    #     """
    #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
    #     return new_ts
