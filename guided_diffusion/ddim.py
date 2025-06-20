import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm

from utils.logger import logging_info
from .gaussian_diffusion import _extract_into_tensor
from .new_scheduler import ddim_timesteps, ddim_repaint_timesteps
from .respace import SpacedDiffusion


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class DDIMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.ddim_sigma = conf.get("ddim.ddim_sigma", 0.0)

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = torch.split(model_output, C, dim=1)
        return model_output

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        with torch.no_grad():
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )

            def process_xstart(_x):
                if denoised_fn is not None:
                    _x = denoised_fn(_x)
                if clip_denoised:
                    return _x.clamp(-1, 1)
                return _x

            e_t = self._get_et(model_fn, x, t, model_kwargs)
            pred_x0 = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=e_t))

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * e_t
            )
            noise = noise_like(x.shape, x.device, repeat=False)

            nonzero_mask = (t != 0).float().view(-1, *
                                                 ([1] * (len(x.shape) - 1)))
            x_prev = mean_pred + noise * sigmas * nonzero_mask # # #

        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }

    def q_sample_middle(self, x, cur_t, tar_t, no_noise=False):
        assert cur_t <= tar_t
        device = x.device
        while cur_t < tar_t:
            if no_noise:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            _cur_t = torch.tensor(cur_t, device=device)
            beta = _extract_into_tensor(self.betas, _cur_t, x.shape)
            x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
            cur_t += 1
        return x

    def q_sample(self, x_start, t, no_noise=False):
        if no_noise:
            noise = torch.zeros_like(x_start)
        else:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod,
                                 t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def x_forward_sample(self, x0, forward_method="from_0", no_noise=False):
        x_forward = [self.q_sample(x0, torch.tensor(0, device=x0.device))]
        if forward_method == "from_middle":
            for _step in range(0, len(self.timestep_map) - 1):
                x_forward.append(
                    self.q_sample_middle(
                        x=x_forward[-1][0].unsqueeze(0),
                        cur_t=_step,
                        tar_t=_step + 1,
                        no_noise=no_noise,
                    )
                )
        elif forward_method == "from_0":
            for _step in range(1, len(self.timestep_map)):
                x_forward.append(
                    self.q_sample(
                        x_start=x0[0].unsqueeze(0),
                        t=torch.tensor(_step, device=x0.device),
                        no_noise=no_noise,
                    )
                )
        return x_forward

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        x_forwards = self.x_forward_sample(x0)
        mask = model_kwargs["gt_keep_mask"]

        x_t = img
        import os
        from utils import normalize_image, save_grid

        for cur_t, prev_t in tqdm(time_pairs):
            # replace surrounding
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            cur_t = torch.tensor([cur_t] * shape[0], device=device)
            prev_t = torch.tensor([prev_t] * shape[0], device=device)

            output = self.p_sample(
                model_fn,
                x=x_t,
                t=cur_t,
                prev_t=prev_t,
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )
            x_t = output["x_prev"]

            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles",
                                 f"mid-{prev_t[0].item()}.png"),
                )
                save_grid(
                    normalize_image(output["pred_x0"]),
                    os.path.join(sample_dir, "middles",
                                 f"pred-{prev_t[0].item()}.png"),
                )

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }


class R_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )

    @staticmethod
    def resample(m, w, n):
        """
        m: max number of index
        w: un-normalized probability
        n: number of indices to be selected
        """
        if max([(math.isnan(i) or math.isinf(i)) for i in w]):
            w = np.ones_like(w)
        if w.sum() < 1e-6:
            w = np.ones_like(w)

        w = n * (w / w.sum())
        c = [int(i) for i in w]
        r = [i - int(i) for i in w]
        added_indices = []
        for i in range(m):
            for j in range(c[i]):
                added_indices.append(i)
        if len(added_indices) != n:
            R = n - sum(c)
            indices_r = torch.multinomial(torch.tensor(r), R)
            for i in indices_r:
                added_indices.append(i)
        logging_info(
            "Indices after Resampling: %s"
            % (" ".join(["%.d" % i for i in sorted(added_indices)]))
        )
        return added_indices

    @staticmethod
    def gaussian_pdf(x, mean, std=1):
        return (
            1
            / (math.sqrt(2 * torch.pi) * std)
            * torch.exp(-((x - mean) ** 2).sum() / (2 * std**2))
        )

    def resample_based_on_x_prev(
        self,
        x_t,
        x_prev,
        x_pred_prev,
        mask,
        keep_n_samples=None,
        temperature=100,
        p_cal_method="mse_inverse",
        pred_x0=None,
    ):
        if p_cal_method == "mse_inverse":  # same intuition but empirically better
            mse = torch.tensor(
                [((x_prev * mask - i * mask) ** 2).sum() for i in x_pred_prev]
            )
            mse /= mse.mean()
            p = torch.softmax(temperature / mse, dim=-1)
        elif p_cal_method == "gaussian":
            p = torch.tensor(
                [self.gaussian_pdf(x_prev * mask, i * mask)
                 for i in x_pred_prev]
            )
        else:
            raise NotImplementedError
        resample_indices = self.resample(
            x_t.shape[0], p, x_t.shape[0] if keep_n_samples is None else keep_n_samples
        )
        x_t = torch.stack([x_t[i] for i in resample_indices], dim=0)
        x_pred_prev = torch.stack([x_pred_prev[i]
                                  for i in resample_indices], dim=0)
        pred_x0 = (
            torch.stack([pred_x0[i] for i in resample_indices], dim=0)
            if pred_x0 is not None
            else None
        )
        logging_info(
            "Resampling with probability %s" % (
                " ".join(["%.3lf" % i for i in p]))
        )
        return x_t, x_pred_prev, pred_x0

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]
        # x_forwards = self.x_forward_sample(x0, "from_middle")
        x_forwards = self.x_forward_sample(x0, "from_0")
        x_t = img

        for cur_t, prev_t in tqdm(time_pairs):
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            x_prev = x_forwards[prev_t]
            output = self.p_sample(
                model_fn,
                x=x_t,
                t=torch.tensor([cur_t] * shape[0], device=device),
                prev_t=torch.tensor([prev_t] * shape[0], device=device),
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )

            x_pred_prev, x_pred_x0 = output["x_prev"], output["pred_x0"]
            x_t, x_pred_prev, pred_x0 = self.resample_based_on_x_prev(
                x_t=x_t,
                x_prev=x_prev,
                x_pred_prev=x_pred_prev,
                mask=mask,
                pred_x0=x_pred_x0,
            )
            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles", f"mid-{prev_t}.png"),
                )
                save_grid(
                    normalize_image(pred_x0),
                    os.path.join(sample_dir, "middles", f"pred-{prev_t}.png"),
                )

        x_t = self.resample_based_on_x_prev(
            x_t=x_t,
            x_prev=x0,
            x_pred_prev=x_t,
            mask=mask,
            keep_n_samples=conf["resample.keep_n_samples"],
        )[0]

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }


# implemenet
class O_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )

        assert conf.get("optimize_xt.optimize_xt",
                        False), "Double check on optimize"
        self.ddpm_num_steps = conf.get(
            "ddim.schedule_params.ddpm_num_steps", 250)
        self.coef_xt_reg = conf.get("optimize_xt.coef_xt_reg", 0.001)
        self.coef_xt_reg_decay = conf.get("optimize_xt.coef_xt_reg_decay", 1.0)
        self.num_iteration_optimize_xt = conf.get(
            "optimize_xt.num_iteration_optimize_xt", 1
        )
        self.lr_xt = conf.get("optimize_xt.lr_xt", 0.001)
        self.lr_xt_decay = conf.get("optimize_xt.lr_xt_decay", 1.0)
        self.use_smart_lr_xt_decay = conf.get(
            "optimize_xt.use_smart_lr_xt_decay", False
        )
        self.use_adaptive_lr_xt = conf.get(
            "optimize_xt.use_adaptive_lr_xt", False)
        self.mid_interval_num = int(conf.get("optimize_xt.mid_interval_num", 1))
        if conf.get("ddim.schedule_params.use_timetravel"):
            self.steps = ddim_repaint_timesteps(**conf["ddim.schedule_params"])
        else:
            self.steps = ddim_timesteps(**conf["ddim.schedule_params"])

        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)

    def p_sample(self, model_fn, x, t, prev_t, model_kwargs, lr_xt, coef_xt_reg, clip_denoised=True, denoised_fn=None,
                 cond_fn=None,return_plot=False,
                 **kwargs):
        if self.mode == "inpaint":
            def loss_fn(_x0, _pred_x0, _mask):
                """
                用来约束模型不要破坏原图中已知的区域，保持语义一致性和稳定性。
                """
                return torch.sum((_x0 * _mask - _pred_x0 * _mask) ** 2)
        elif self.mode == "super_resolution":
            raise NotImplementedError("Super-resolution mode is disabled.")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        def reg_fn(_origin_xt, _xt):
            """
            优化的平滑正则项，
            """
            return torch.sum((_origin_xt - _xt) ** 2)

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1, 1)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                return grad_ckpt(self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False)
            else:
                return self._get_et(model_fn, _x, _t, model_kwargs)

        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (np.arange(0, interval_num) * interval).round()[::-1].astype(np.int32).tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            ret = 1
            for _cur_t, _prev_t in zip(steps[:-1], steps[1:]):
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(self.alphas_cumprod[_prev_t])
            return ret

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (np.arange(0, interval_num) * interval).round()[::-1].astype(np.int32).tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            x_t = _x
            for _cur_t, _prev_t in zip(steps[:-1], steps[1:]):
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor([_prev_t] * _x.shape[0], device=_x.device)
                _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False)
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False)
                return process_xstart(_pred_x0)

        def get_update(_x, cur_t, _prev_t, _et=None, _pred_x0=None):
            """
            实现DDIM采样中的某一步 从 xt→xt−1的公式

            Args:
                _x: 当前扩散图像xt
                cur_t:当前时间步 cur_t
                _prev_t:前一步 prev_t
                _et:模型预测的噪声 （可选）
                _pred_x0:模型预测的原图 （可选）

            Returns:估计出的上一步图像 xt−1 即 x_prev

            """
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)
            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                    self.ddim_sigma * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) *
                    torch.sqrt(1 - alpha_t / alpha_prev)
            )
            mean_pred = (
                    _pred_x0 * torch.sqrt(alpha_prev) +
                    torch.sqrt(1 - alpha_prev - sigmas ** 2) * _et
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1, *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)
        x0 = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]

        if self.use_smart_lr_xt_decay:
            lr_xt /= get_smart_lr_decay_rate(t, self.mid_interval_num)

        with torch.enable_grad():
            origin_x = x.clone().detach()
            x = x.detach().requires_grad_()
            e_t = get_et(x, t)
            pred_x0 = get_predx0(x, t, e_t, interval_num=self.mid_interval_num)
            prev_loss = loss_fn(x0, pred_x0, mask).item() # 对比 pred_x0 和真实图像 x0 的已知区域, 初始损失 prev_loss，也是优化之前的损失

            logging_info(f" t_{t[0].item()} lr_xt {lr_xt:.8f}")

            grad_xt = None # 画图
            grad_xt_reg = None # 画图

            for step in range(self.num_iteration_optimize_xt):

                loss = loss_fn(x0, pred_x0, mask) + coef_xt_reg * reg_fn(origin_x, x)
                # 第一项，仅在保留区域对 x0 和 pred_x0 做一致性检查。
                # 第二项是正则项 reg_fn，惩罚 x 距离 origin_x 变化过大（平滑优化，防止梯度爆炸）
                # 第二项，是一个 正则化项（regularization term），coef_xt_reg: 超参数（λ），控制正则项的权重

                x_grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0].detach()
                new_x = x - lr_xt * x_grad # # x ← x − α · ∇loss(x) 计算梯度并手动更新

                if step == 0: # 画图
                    grad_xt = x_grad.clone().detach() # 画图
                    grad_xt_reg = coef_xt_reg * (x.detach() - origin_x) # 画图

                # logging_info(
                #     f"grad norm: {torch.norm(x_grad, p=2).item():.3f} "
                #     f"{torch.norm(x_grad * mask, p=2).item():.3f} "
                #     f"{torch.norm(x_grad * (1. - mask), p=2).item():.3f}"
                # )

                while self.use_adaptive_lr_xt:
                    with torch.no_grad():
                        e_t = get_et(new_x, t)
                        pred_x0 = get_predx0(new_x, t, e_t, interval_num=self.mid_interval_num)
                        new_loss = loss_fn(x0, pred_x0, mask) + coef_xt_reg * reg_fn(origin_x, new_x)
                        if not torch.isnan(new_loss) and new_loss <= loss:
                            break
                        else:
                            lr_xt *= 0.8
                            logging_info(
                                "Loss too large (%.3lf->%.3lf)! Learning rate decreased to %.5lf."
                                % (loss.item(), new_loss.item(), lr_xt)
                            )
                            del new_x, e_t, pred_x0, new_loss
                            new_x = x - lr_xt * x_grad

                x = new_x.detach().requires_grad_()
                e_t = get_et(x, t)
                pred_x0 = get_predx0(x, t, e_t, interval_num=self.mid_interval_num)
                del loss, x_grad
                torch.cuda.empty_cache()

        with torch.no_grad():
            new_loss = loss_fn(x0, pred_x0, mask).item()
            new_reg = reg_fn(origin_x, x).item() # new_reg越大,说明更新越剧烈【reg_fn计算的是 当前优化后x 和初始x（origin_x）之间的距离】
            logging_info("Loss Change ( prev_loss -> new_loss) : (%.3lf -> %.3lf)" % (prev_loss, new_loss))
            logging_info("Regularization Change ( 0 -> reg_fn(origin_x, x‘) ): %.3lf -> %.3lf" % (0, new_reg))

            pred_x0, e_t, x = pred_x0.detach(), e_t.detach(), x.detach()
            loss_prev = torch.tensor(prev_loss, device=x.device) # 画图
            reg_xt = torch.tensor(new_reg, device=x.device)      # 画图

            del origin_x, prev_loss

            x_prev = get_update(x, t, prev_t, e_t, _pred_x0=pred_x0 if self.mid_interval_num == 1 else None)

        if return_plot: # 画图
            return {
                "x": x,
                "x_prev": x_prev,
                "pred_x0": pred_x0,
                "loss_prev": loss_prev.detach(),
                "reg_xt": reg_xt.detach(),
                "loss": torch.tensor(new_loss, device=x.device).detach(),
                "grad_xt": grad_xt.detach(),
                "grad_xt_reg": grad_xt_reg.detach(),
            }
        else:
            return {
                "x": x,
                "x_prev": x_prev,
                "pred_x0": pred_x0,
                "loss": torch.tensor(new_loss, device=x.device).detach(),
            }

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            assert not conf["optimize_xt.filter_xT"]
            img = noise
        else:
            xT_shape = (
                shape
                if not conf["optimize_xt.filter_xT"]
                else tuple([20] + list(shape[1:]))
            )
            img = torch.randn(xT_shape, device=device)

        if conf["optimize_xt.filter_xT"]: # celebahq.yaml 的 filter_xT: false
            xT_losses = []
            for img_i in img:
                xT_losses.append(
                    self.p_sample(model_fn, x=img_i.unsqueeze(0), t=torch.tensor([self.steps[0]] * 1, device=device),
                                  prev_t=torch.tensor([0] * 1, device=device), model_kwargs=model_kwargs,
                                  lr_xt=self.lr_xt, coef_xt_reg=self.coef_xt_reg, pred_xstart=None)["loss"]
                )
            img = img[torch.argsort(torch.tensor(xT_losses))[: shape[0]]]

        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))

        x_t = img
        # set up hyper paramer for this run
        lr_xt = self.lr_xt
        coef_xt_reg = self.coef_xt_reg
        loss = None

        status = None

        # 添加记录用的变量
        log_time_steps = [] # 画图
        log_lr_xt = [] # 画图
        log_grad_norms = [] # 画图
        log_loss_prev = [] # 画图
        log_loss_new = [] # 画图
        log_reg_new = [] # 画图

        for cur_t, prev_t in tqdm(time_pairs):
            if cur_t > prev_t:  # denoise
                status = "reverse"
                cur_t = torch.tensor([cur_t] * shape[0], device=device)
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                output = self.p_sample(model_fn, x=x_t, t=cur_t, prev_t=prev_t, model_kwargs=model_kwargs, lr_xt=lr_xt,
                                       coef_xt_reg=coef_xt_reg, pred_xstart=None,
                                       return_plot=True, # 画图, 是否在p_sample()的过程中返回需要画图的东西
                                       )
                # # #这里output["x_prev"]就是优化完的x_t-1了

                gt_keep_mask = model_kwargs.get('gt_keep_mask')  # # # # repaint原来的流程
                gt = model_kwargs['gt']  # # # # repaint原来的流程
                alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, prev_t, x_t.shape)  # # # # repaint原来的流程
                gt_weight = torch.sqrt(alpha_cumprod)  # # # # repaint原来的流程
                gt_part = gt_weight * gt  # # # # repaint原来的流程

                noise_weight = torch.sqrt((1 - alpha_cumprod))  # # # # repaint原来的流程
                noise_part = noise_weight * torch.randn_like(x_t)  # # # # repaint原来的流程

                xt_konwn = gt_part + noise_part   # # # # repaint原来的流程

                # if t.equal(th.tensor([48], device='cuda:0')):
                #     self.show_pic_of(x,"noise.png",False)
                #
                # if t.equal(th.tensor([0], device='cuda:0')):
                #     self.show_pic_of( x,"x.png",False)
                #
                # if t.equal(th.tensor([0], device='cuda:0')):
                #     self.show_pic_of( (1 - gt_keep_mask) * x,"one_minus_mask_x.png",False)


                log_time_steps.append(prev_t[0].item()) # 画图


                log_lr_xt.append(lr_xt) # 画图

                if "grad_xt" in output and "grad_xt_reg" in output: # 画图
                    grad_xt = output["grad_xt"]
                    grad_xt_reg = output["grad_xt_reg"]
                    grad_xt_norm = grad_xt.norm().item()
                    grad_xt_reg_norm = grad_xt_reg.norm().item()
                    grad_diff = abs(grad_xt_norm - grad_xt_reg_norm)
                    log_grad_norms.append((grad_xt_norm, grad_xt_reg_norm, grad_diff))

                if "loss_prev" in output and "loss" in output: # 画图
                    log_loss_prev.append(output["loss_prev"])
                    log_loss_new.append(output["loss"])

                if "reg_xt" in output: # 画图
                    log_reg_new.append(output["reg_xt"])

                x_t = output["x_prev"] # # 拿出优化结束后的x_t-1
                x_t = (gt_keep_mask * (xt_konwn) + (1 - gt_keep_mask) * (x_t)) # # # # repaint原来的流程
                loss = output["loss"]

                # lr decay

                logging_info("当前的 lr_xt: %.5lf " % lr_xt)

                if self.lr_xt_decay != 1.0:
                    logging_info(
                        "Learning rate of xt decay: %.5lf -> %.5lf."
                        % (lr_xt, lr_xt * self.lr_xt_decay)
                    )
                lr_xt *= self.lr_xt_decay

                logging_info("当前的 coef_xt_reg: %.5lf " % coef_xt_reg)

                if self.coef_xt_reg_decay != 1.0:
                    logging_info(
                        "Coefficient of regularization decay: %.5lf -> %.5lf."
                        % (coef_xt_reg, coef_xt_reg * self.coef_xt_reg_decay)
                    )
                coef_xt_reg *= self.coef_xt_reg_decay

                # if conf["debug"]:
                #     from utils import normalize_image, save_grid
                #
                #     os.makedirs(os.path.join(
                #         sample_dir, "middles"), exist_ok=True)
                #     save_grid(
                #         normalize_image(x_t),
                #         os.path.join(
                #             sample_dir, "middles", f"mid-{prev_t[0].item()}.png"
                #         ),
                #     )
                #     save_grid(
                #         normalize_image(output["pred_x0"]),
                #         os.path.join(
                #             sample_dir, "middles", f"pred-{prev_t[0].item()}.png"
                #         ),
                #     )
            else:  # time travel back
                if status == "reverse" and conf.get(
                    "optimize_xt.optimize_before_time_travel", False
                ):
                    # update xt if previous status is reverse
                    x_t = self.get_updated_xt(
                        model_fn,
                        x=x_t,
                        t=torch.tensor([cur_t] * shape[0], device=device),
                        model_kwargs=model_kwargs,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                    )
                status = "forward"
                assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                with torch.no_grad():
                    x_t = self._undo(x_t, prev_t)
                # undo lr decay
                logging_info(f"Undo step: {cur_t}")
                lr_xt /= self.lr_xt_decay
                coef_xt_reg /= self.coef_xt_reg_decay

        x_t = x_t.clamp(-1.0, 1.0)  # normalize

        # 画图函数定义
        def plot_logs(
                time_steps, lr_xt_list, grad_norms, loss_prev_list, loss_new_list, reg_new_list,
                save_path="log_metrics_plot.png"
        ):
            """
            图在一起的
            """
            import matplotlib.pyplot as plt
            import torch

            # 处理 grad_norms 解包
            if grad_norms:
                grad1, grad2, grad_diff = zip(*grad_norms)
            else:
                grad1 = grad2 = grad_diff = []

            # 确保 time_steps 都是 CPU 上的标量
            time_steps = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in time_steps]

            # 确保 lr_xt_list 是可用的标量列表
            lr_xt_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in lr_xt_list]

            plt.figure(figsize=(12, 8))

            # Learning Rate 图1 #################################
            plt.subplot(2, 2, 1)
            plt.plot(time_steps, lr_xt_list, marker='o', color='blue')
            plt.title("Learning Rate (lr_xt) over t")
            plt.xlabel("Time Step t")
            plt.ylabel("lr_xt")
            plt.grid(True)
            plt.gca().invert_xaxis()

            # # Gradient Norms 图2 ################################
            # if grad1:
            #     # 同样处理 grad_norms，保证是 CPU 上的数值
            #     grad1 = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in grad1]
            #     grad2 = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in grad2]
            #     grad_diff = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in grad_diff]
            #
            #     plt.subplot(2, 2, 2)
            #     plt.plot(time_steps, grad1, label='grad_xt')
            #     plt.plot(time_steps, grad2, label='grad_xt_reg')
            #     plt.plot(time_steps, grad_diff, label='grad_diff')
            #     plt.title("Gradient Norms over t")
            #     plt.xlabel("Time Step t")
            #     plt.ylabel("Norms")
            #     plt.legend()
            #     plt.grid(True)
            # plt.gca().invert_xaxis()

            # Loss 图3 #################################
            if loss_prev_list and loss_new_list:
                plt.subplot(2, 2, 3)

                loss_prev_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in loss_prev_list]
                loss_new_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in loss_new_list]

                plt.plot(time_steps, loss_prev_list, label='loss_prev', linestyle='--')
                plt.plot(time_steps, loss_new_list, label='loss_new', linestyle='-')
                plt.title("Loss Change over t")
                plt.xlabel("Time Step t")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True)
                plt.gca().invert_xaxis()

            # Reg 图4 ##############################
            if reg_new_list:
                plt.subplot(2, 2, 4)
                reg_new_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in reg_new_list]
                plt.plot(time_steps, reg_new_list, label='reg_xt', color='green')
                plt.title("Reg over t")
                plt.xlabel("Time Step t")
                plt.ylabel("reg_xt")
                plt.grid(True)
                plt.gca().invert_xaxis()

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            print(f"[Log] 图像保存为 {save_path}")

        def plot_logs_1(
                time_steps, lr_xt_list, grad_norms, loss_prev_list, loss_new_list, reg_new_list,
                save_dir="."
        ):
            """
            图片分开保存
            """
            import matplotlib.pyplot as plt
            import torch
            import os

            os.makedirs(save_dir, exist_ok=True)

            # 解包 grad_norms
            if grad_norms:
                grad1, grad2, grad_diff = zip(*grad_norms)
            else:
                grad1 = grad2 = grad_diff = []

            # 标量化
            time_steps = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in time_steps]
            lr_xt_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in lr_xt_list]

            # 图1：Learning Rate
            plt.figure()
            plt.plot(time_steps, lr_xt_list, marker='o', color='blue')
            plt.title("Learning Rate (lr_xt) over t")
            plt.xlabel("Time Step t")
            plt.ylabel("lr_xt")
            plt.grid(True)
            plt.gca().invert_xaxis()
            save_path = os.path.join(save_dir, "lr_xt_plot.png")
            plt.savefig(save_path, dpi=300)
            print(f"[Log] 图像保存为 {save_path}")
            plt.close()

            # 图2：Gradient Norms
            if grad1:
                grad1 = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in grad1]
                grad2 = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in grad2]
                grad_diff = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in grad_diff]

                plt.figure()
                plt.plot(time_steps, grad1, label='grad_xt')
                plt.plot(time_steps, grad2, label='grad_xt_reg')
                plt.plot(time_steps, grad_diff, label='grad_diff')
                plt.title("Gradient Norms over t")
                plt.xlabel("Time Step t")
                plt.ylabel("Norms")
                plt.legend()
                plt.grid(True)
                plt.gca().invert_xaxis()
                save_path = os.path.join(save_dir, "grad_norms_plot.png")
                plt.savefig(save_path, dpi=300)
                print(f"[Log] 图像保存为 {save_path}")
                plt.close()

            # 图3：Loss
            if loss_prev_list and loss_new_list:
                loss_prev_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in loss_prev_list]
                loss_new_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in loss_new_list]

                plt.figure()
                plt.plot(time_steps, loss_prev_list, label='loss_prev', linestyle='--')
                plt.plot(time_steps, loss_new_list, label='loss_new', linestyle='-')
                plt.title("Loss Change over t")
                plt.xlabel("Time Step t")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True)
                plt.gca().invert_xaxis()
                save_path = os.path.join(save_dir, "loss_plot.png")
                plt.savefig(save_path, dpi=300)
                print(f"[Log] 图像保存为 {save_path}")
                plt.close()

            # 图4：Reg
            if reg_new_list:
                reg_new_list = [x.detach().cpu().item() if torch.is_tensor(x) else x for x in reg_new_list]

                plt.figure()
                plt.plot(time_steps, reg_new_list, label='reg_xt', color='green')
                plt.title("Reg over t")
                plt.xlabel("Time Step t")
                plt.ylabel("reg_xt")
                plt.grid(True)
                plt.gca().invert_xaxis()
                save_path = os.path.join(save_dir, "reg_plot.png")
                plt.savefig(save_path, dpi=300)
                print(f"[Log] 图像保存为 {save_path}")
                plt.close()

        # 调用函数画图 # 四合一图
        # plot_logs( # 四合一图
        #     time_steps=log_time_steps,
        #     lr_xt_list=log_lr_xt,
        #     grad_norms=log_grad_norms,
        #     loss_prev_list=log_loss_prev,
        #     loss_new_list=log_loss_new,
        #     reg_new_list=log_reg_new,
        #     save_path=os.path.join("./log_metrics_plot.png")
        # )

        # 分开保存
        plot_logs_1(
            time_steps=log_time_steps,
            lr_xt_list=log_lr_xt,
            grad_norms=log_grad_norms,
            loss_prev_list=log_loss_prev,
            loss_new_list=log_loss_new,
            reg_new_list=log_reg_new,
            save_dir="./myplots"
        )

        return {"sample": x_t, "loss": loss}

    def get_updated_xt(self, model_fn, x, t, model_kwargs, lr_xt, coef_xt_reg):
        return self.p_sample(
            model_fn,
            x=x,
            t=t,
            prev_t=torch.zeros_like(t, device=t.device),
            model_kwargs=model_kwargs,
            pred_xstart=None,
            lr_xt=lr_xt,
            coef_xt_reg=coef_xt_reg,
        )["x"]
