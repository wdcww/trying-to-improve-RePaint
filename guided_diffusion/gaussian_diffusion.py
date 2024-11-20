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
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from collections import defaultdict

from guided_diffusion.scheduler import get_schedule_jump


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, use_scale):
    """
    此函数用在了 script_util.py
    schedule_name 是 linear, 因为.yml文件里的 noise_schedule: linear
    num_diffusion_timesteps , .yml文件里的 diffusion_steps: 1000
    use_scale 传过来是true
    """
    if schedule_name == "linear":
        if use_scale:
            # 如果扩散步数变少，例如 100，直接使用[0.0001, 0.02]的范围会导致扩散过程较短时的噪声不合理。
            # 因此通过scale = 1000 / num_diffusion_timesteps
            # 进行比例调整，保持扩散强度的一致性。
            scale = 1000 / num_diffusion_timesteps
        else:
            # 原始的扩散模型（如论文 Denoising Diffusion Probabilistic Models 中）通常使用 1000 个扩散步长，
            # scale = 1 时，beta_start 和 beta_end 的默认范围是 [0.0001, 0.02]。
            scale = 1

        beta_start = scale * 0.0001 # β
        beta_end = scale * 0.02

        # 会生成 num_diffusion_timesteps 个均匀分布在 [beta_start, beta_end ] 区间的 beta 值。
        return np.linspace( beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64 )

    elif schedule_name == "cosine":
        # cosine schedule
        # 一个非线性分布的 beta 序列，根据 alpha_bar(t) 函数定义的扩散强度分布。
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

# # 如果打开上面那个函数的elif,下面这个得跟着打开
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# 模型预测的输出类型 枚举类1
class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


# 模型输出的方差类型 枚举类2
class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


# 模型损失函数类型 枚举类3
class LossType(enum.Enum):
    """
    该枚举类定义了模型训练中的损失函数类型。不同的损失函数类型对模型训练有不同的影响，具体有以下四种选择：

    MSE：使用普通的均方误差（MSE）损失。
    RESCALED_MSE：使用重新缩放的均方误差损失，通常在学习方差时与 RESCALED_KL 配合使用。
    KL：使用变分下界（KL散度）损失。
    RESCALED_KL：类似于 KL，但对其进行了重新缩放，以便估计完整的变分下界（VLB）。

     is_vb(): 用于判断当前损失是否为变分下界类型（KL或重新缩放KL）。
    """
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
        Extract values from a 1-D numpy array for a batch of indices.
        从1-D numpy数组arr中提取一批索引的值。

        :param arr: the 1-D numpy array.
                    1-D numpy 数组
        :param timesteps: a tensor of indices into the array to extract.
                         要提取到数组中的索引张量。
        :param broadcast_shape: a larger shape of K dimensions with the batch dimension equal to the length of timesteps.
                                一个K维的较大形状，批处理维度等于时间步长的长度。
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
                return形状为 [batch_size, 1, ...] 的张量，其中形状有 K 个维度。
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).

    betas：每个扩散时间步长的beta的一维numpy数组，从T开始到1。
    model_mean_type：决定模型输出均值的 ModelMeanType。
    model_var_type: 决定如何输出方差的 ModelVarType。
    loss_type: 确定要使用的损失函数的 LossType。
    rescale_timesteps: 如果为 True，则将浮点时间步传递到模型，以便它们始终像在原论文（0 到 1000）。
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
            conf=None # 在guided-diffusion基础添加上
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.conf = conf # 在guided-diffusion基础上添加

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_prev_prev = np.append(1.0, self.alphas_cumprod_prev[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) # 根号下的 累积alphas
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev) # 根号下的 累积alphas_prev
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # 根号下的 1-累积alphas
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod) # log 1-累积alphas
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod) # 根号下的 1/累积alphas
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1) # 根号下的 (1/累积alphas)-1

        # 后验方差
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        # 后验均值的两个系数 coef1 coef2
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) /
                (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )


    # def q_mean_variance
    #                   Get the distribution q(x_t | x_0).

    # def q_sample
    #                   sample from distribution q(x_t | x_0).

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0) ∼N(μt−1,posterior_variance)
            posterior_mean 即 μt-1
            posterior_variance
        """
        assert x_start.shape == x_t.shape
        posterior_mean = ( _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = _extract_into_tensor( self.posterior_variance, t, x_t.shape)

        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        预测噪声eps 版本
        """
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            -
            _extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        专门搞回来为了ddim采样
        """
        return (
          (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2] # Batch size和Channel是状态图x的前两维
        assert t.shape == (B,) # 且scheduler.py的“真”时间步的shape应该与Batch size一致

        model_output = model(x, t, **model_kwargs) # 当由于子类的调用到这里时，此“model”就已是<guided_diffusion.respace._WrappedModel 对象了>
        # print("model_out=  ",model_output)
        # # ++++++++++++++++++++++++++++ 1 model_var_type 深受.yml里面的 learn_sigma 影响+++++++++++++++++++++++++++++++
        # # 若learn_sigma为True,则model_var_type为gd.ModelVarType.LEARNED_RANGE （script_util.py）
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            #  self.model_var_type 是 ModelVarType.LEARNED 或者 ModelVarType.LEARNED_RANGE
            assert model_output.shape == (B, C * 2, *x.shape[2:])

            model_output, model_var_values = th.split(model_output, C, dim=1)

            if self.model_var_type == ModelVarType.LEARNED:
                # if模型方差类型是 ModelVarType.LEARNED
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                # 模型方差类型是 ModelVarType.LEARNED_RANGE
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)

        else:
            # self.model_var_type既不是 ModelVarType.LEARNED 也不是 ModelVarType.LEARNED_RANGE
            model_variance, model_log_variance = \
            {
                # for ModelVarType.FIXED_LARGE, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            # model_variance, model_log_variance 再
            # 分别通过下面的_extract_into_tensor() 从序列中提取当前时间步的值
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def process_xstart(x):
            """
            注意！ 这个函数还在p_mean_variance()函数里面
            """
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # 对去噪后的图像进行裁剪。裁剪是为了确保输出图像在有效范围内。
                return x.clamp(-1, 1)
            return x
        # # ————————————————————————————— 2 model_mean_type 深受.yml文件里 predict_xstart 影响——————————————————————————
        # # predict_xstart为false, gd.ModelMeanType.EPSILON （predict_xstart为true,  gd.ModelMeanType.START_X）
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # 模型直接输出上一个时间步的 x_t-1 (_predict_xstart_from_xprev)
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))

            model_mean = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:

            if self.model_mean_type == ModelMeanType.START_X:
                # 模型直接预测 x_0（原始图像） (预测 x_0 版本)
                # 如果启用了 predict_xstart=True，model_mean_type 就是这个
                pred_xstart = process_xstart(model_output)
            else:
                # self.model_mean_type 是 ModelMeanType.EPSILON (当.yml文件里 predict_xstart: false) ###
                # 模型预测的是 噪声项epsilon，
                # 通过噪声和当前 x_t 推导出原始图像 x_0
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        else:
            raise NotImplementedError(self.model_mean_type)
        # # ——————————————————————————————————————————————————————————————————————————————————————————————————————————
        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
    #     """
    #     Compute the mean for the previous step, given a function cond_fn that
    #     computes the gradient of a conditional log probability with respect to
    #     x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
    #     condition on y.
    #
    #     This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
    #     """
    #
    #     gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
    #
    #     new_mean = (
    #             p_mean_var["mean"].float() + p_mean_var["variance"] *
    #             gradient.float()
    #     )
    #     return new_mean

    # #
    # #####
    # # # 要去做Resampling时调用的undo() # ############################################### # ###########################
    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
            return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
            beta = _extract_into_tensor(self.betas, t, img_out.shape)

            img_in_est = th.sqrt(1 - beta) * img_out + th.sqrt(beta) * th.randn_like(img_out)  # 正向扩散

            return img_in_est

    # ##### ddpm #########################################################################################
    def p_sample_loop(
            self,
            model, # # 此时的model早已是test.py里面那个model_fn了,即可以往model里面传x, t, y, gt
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            return_all=False,
            conf=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.模型模块。
        :param shape: the shape of the samples, (N, C, H, W). 样本的形状（N、C、H、W）
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
                      如果指定，则为来自编码器采样的噪声。应该与“shape”具有相同的形状
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
                      如果为 True，则将 x_start 预测剪辑为 [-1, 1]。
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
            如果不是 None，则适用于x_start 在用于采样之前进行预测。
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
                        如果不是 None，这是一个起作用的梯度函数(与模型类似。)
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
            如果不是 None，则为额外关键字参数的字典传递给模型。这可以用于调节。
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
                       如果指定，则为在其上创建示例的设备。如果未指定，则使用模型参数的设备。
        :param progress: if True, show a tqdm progress bar.
                       如果为 True，则显示 tqdm 进度条。
        :return: a non-differentiable batch of samples.
                返回一批不可微分的样本。
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                conf=conf
        ):
            final = sample

        if return_all:
            return final
        else:
            return final["sample"]

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            conf=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        从模型生成样本，并从每个扩散时间步生成中间样本。
        参数与 p_sample_loop() 相同。
        返回一个字典生成器，其中每个字典都是 p_sample() 的返回值。
        :return: 一个字典
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        # debug_steps = conf.pget('debug.num_timesteps')

        self.gt_noises = None  # reset for next image

        pred_xstart = None

        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        if conf.schedule_jump_params:

            times = get_schedule_jump(**conf.schedule_jump_params) #### 获取那个scheduler.py的list # ###

            time_pairs = list(zip(times[:-1], times[1:]))
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = th.tensor([t_last] * shape[0],  # pylint: disable=not-callable
                                     device=device) # # shape[0]是batch_size

                if t_cur < t_last:
                    # 只要“for t_last, t_cur in time_pairs:”中 后一步是比前一步小的，就是正常的推理
                    with th.no_grad():
                        # image_before_step = image_after_step.clone() ## 这个没用到,先注释了！

                        # print("p_sample_loop_progressive() ",t_last_t)

                        out = self.p_sample(
                            model,
                            image_after_step, ## 因为第一次过来时noise是None,被初始化为 image_after_step = th.randn(*shape, device=device)
                            t_last_t, # # # # # 这里才有了时间步 # # # # # # #
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                            conf=conf,
                            pred_xstart=pred_xstart
                        )
                        image_after_step = out["sample"]  # 更新状态图
                        pred_xstart = out["pred_xstart"]

                        sample_idxs[t_cur] += 1

                        yield out

                else:  # 要去做Resampling
                    t_shift = conf.get('inpa_inj_time_shift', 1)

                    image_before_step = image_after_step.clone()
                    image_after_step = self.undo(image_before_step,
                                                 image_after_step,
                                                 est_x_0=out['pred_xstart'],
                                                 t=t_last_t + t_shift,
                                                 debug=False)
                    pred_xstart = out["pred_xstart"]

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            conf=None, meas_fn=None, pred_xstart=None, idx_wall=-1
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        在给定的时间步长【合成】 x_{t-1}。

        model：要从中采样的模型。

        x：传进来的x_{t-1}处的当前张量。

        t：t 的值，从 0 开始，表示第一个扩散步骤。

        clip_denoised：如果为 True，则将 x_start 预测剪辑到 [-1, 1]。

        denoised_fn：如果不是 None，则在用于采样之前应用于 x_start 预测的函数。

        cond_fn：如果不是 None，则这是一个作用类似于模型的梯度函数。

        model_kwargs：如果不是 None，则传递给模型的额外关键字参数字典。这可用于条件。

        conf :
        meas_fn: （可选）用于度量的函数（未在代码中使用）。
        pred_xstart: （可选）对 x_{0} 的预测，若存在则使用。
        idx_wall: （可选）用于墙的索引（未在代码中使用）

        return包含['sample'：来自模型的随机样本][‘pred_xstart’：x_0 的预测]的字典.

        """

        if conf.inpa_inj_sched_prev: # 这个是Repaint的

            if pred_xstart is not None: # 第一次是none,跳过了。
                gt_keep_mask = model_kwargs.get('gt_keep_mask')
                if gt_keep_mask is None:
                    gt_keep_mask = conf.get_inpa_mask(x)

                gt = model_kwargs['gt']

                alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

                if conf.inpa_inj_sched_prev_cumnoise:
                    pass # weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
                else:
                    gt_weight = th.sqrt(alpha_cumprod)
                    gt_part = gt_weight * gt

                    noise_weight = th.sqrt((1 - alpha_cumprod))
                    noise_part = noise_weight * th.randn_like(x)

                    weighed_gt = gt_part + noise_part
                # x_ = x # ##########################################################
                x = (gt_keep_mask * (weighed_gt) + (1 - gt_keep_mask) * (x))
                # print(x==x_) # #######################################################

        out = self.p_mean_variance(  # 调用这个p_mean_variance前，就会去涉及respace.py里面的那些东西
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        # if cond_fn is not None:
        #     out["mean"] = self.condition_mean(
        #         cond_fn, out, x, t, model_kwargs=model_kwargs
        #     )

        noise = th.randn_like(x)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        # sample = out["mean"] # ddpm不引入随机噪声时

        result = {"sample": sample,
                  "pred_xstart": out["pred_xstart"],
                  'gt': model_kwargs.get('gt')
                  }

        return result

    # ##### ddim #########################################################################################################
    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0, #########################################################################
            return_all=False,
            conf=None
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        eta=conf.eta
        print("gaussion_diffusion.py/ddim_sample_loop() -- eta --: ", eta)
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                conf=conf
        ):
            final = sample
        # return final["sample"]
        if return_all:
            return final
        else:
            return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            conf=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM, adapted for Repaint.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        # Set up variables
        pred_xstart = None
        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        # Handle jump scheduling if specified in conf
        if conf.schedule_jump_params:
            times = get_schedule_jump(**conf.schedule_jump_params)
            time_pairs = list(zip(times[:-1], times[1:]))

            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = th.tensor([t_last] * shape[0], device=device)

                if t_cur < t_last:  # Regular sampling
                    with th.no_grad():
                        img_before_step = img.clone()
                        out = self.ddim_sample(
                            model,
                            img,
                            t_last_t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                            eta=eta,
                            conf=conf,
                            pred_xstart=pred_xstart
                        )
                        img = out["sample"]
                        pred_xstart = out["pred_xstart"]

                        sample_idxs[t_cur] += 1

                        yield out

                else:  # Resampling case
                    t_shift = conf.get('inpa_inj_time_shift', 1)

                    img_before_step = img.clone()
                    img = self.undo(img_before_step,
                                    img,
                                    est_x_0=out['pred_xstart'],
                                    t=t_last_t + t_shift,
                                    debug=False)
                    pred_xstart = out["pred_xstart"]

        else:
            # Regular DDIM sampling without jump scheduling
            indices = list(range(self.num_timesteps))[::-1]
            if progress:
                from tqdm.auto import tqdm
                indices = tqdm(indices)

            for i in indices:
                t = th.tensor([i] * shape[0], device=device)
                with th.no_grad():
                    out = self.ddim_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                        conf=conf,
                        pred_xstart=pred_xstart
                    )
                    img = out["sample"]
                    pred_xstart = out["pred_xstart"]

                    yield out

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
            conf=None, meas_fn=None, pred_xstart=None, idx_wall=-1
    ):
        """
        Sample x_{t-1} from the model using DDIM, adapted for Repaint.
        """
        # Inject ground truth for inpainting if specified in conf
        if conf.inpa_inj_sched_prev:
            if pred_xstart is not None:
                # Retrieve mask and ground truth
                gt_keep_mask = model_kwargs.get('gt_keep_mask')
                if gt_keep_mask is None:
                    gt_keep_mask = conf.get_inpa_mask(x)

                gt = model_kwargs['gt']
                alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

                # Optionally add noise to ground truth
                if conf.inpa_inj_sched_prev_cumnoise:
                    weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
                else:
                    gt_weight = th.sqrt(alpha_cumprod)
                    gt_part = gt_weight * gt

                    noise_weight = th.sqrt((1 - alpha_cumprod))
                    noise_part = noise_weight * th.randn_like(x)

                    weighed_gt = gt_part + noise_part

                # Merge noised ground truth with input x
                x = (gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x)

        # Compute model output using DDIM
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # if cond_fn is not None:
        #     out = self.condition_score(
        #         cond_fn, out, x, t, model_kwargs=model_kwargs
        #     )

        # Calculate epsilon prediction from x_start prediction
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        # Compute DDIM sampling parameters
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = \
        (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Compute sample mean and add noise if t > 0
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"], 'gt': model_kwargs.get('gt')}

# ############################################################################################################################
