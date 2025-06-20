import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from datasets import load_lama_celebahq, load_imagenet
from datasets.celebahq import load_custom_dataset
from datasets.utils import normalize
from guided_diffusion import (
    DDIMSampler,
    O_DDIMSampler,
    DDNMSampler,
    DDRMSampler,
    DPSSampler,
)
from guided_diffusion import dist_util
from guided_diffusion.ddim import R_DDIMSampler
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
    create_classifier,
    classifier_defaults,
)
from metrics import LPIPS, PSNR, SSIM, Metric
from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer


def prepare_model(algorithm, conf, device):
    logging_info("Prepare model...")
    unet = create_model(**select_args(conf, model_defaults().keys()), conf=conf)
    SAMPLER_CLS = {
        "repaint": SpacedDiffusion,
        "ddim": DDIMSampler,
        "o_ddim": O_DDIMSampler,
        "resample": R_DDIMSampler,
        "ddnm": DDNMSampler,
        "ddrm": DDRMSampler,
        "dps": DPSSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    logging_info(f"Loading model from {conf.model_path}...")  # 这里会输出 加载了'./checkpoints/celeba256_250000.pt'
    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu" # unet使用./checkpoints/celeba256_250000.pt ??
        ), strict=False
    )
    unet.to(device)
    if conf.use_fp16:
        unet.convert_to_fp16()
    unet.eval()
    return unet, sampler


def prepare_classifier(conf, device):
    logging_info("Prepare classifier...")
    classifier = create_classifier(
        **select_args(conf, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.classifier_path), map_location="cpu"
        )
    )
    classifier.to(device)
    classifier.eval()
    return classifier


def prepare_data(
    dataset_name, mask_type="half", dataset_starting_index=-1, dataset_ending_index=-1
):
    if dataset_name == "custom":
        datas = load_custom_dataset(
            image_dir="./your_dataset/celeba", # # # # # # # # ## ## # # ## # #
            single_mask_path="./your_dataset/half.jpg", # 不管有多少张图（假设最多 max_len 张），都使用一张指定的 mask
            shape=(256, 256),
            max_len=1,  # 根据需要限制最大数量
            name="half" # # # # # # # 你的mask类型
        )
    elif dataset_name == "celebahq":
        datas = load_lama_celebahq(mask_type=mask_type)
    elif dataset_name == "imagenet":
        datas = load_imagenet(mask_type=mask_type)
    elif dataset_name == "imagenet64":
        datas = load_imagenet(mask_type=mask_type, shape=(64, 64))
    elif dataset_name == "imagenet128":
        datas = load_imagenet(mask_type=mask_type, shape=(128, 128))
    elif dataset_name == "imagenet512":
        datas = load_imagenet(mask_type=mask_type, shape=(512, 512))
    else:
        raise NotImplementedError

    dataset_starting_index = (
        0 if dataset_starting_index == -1 else dataset_starting_index
    )
    dataset_ending_index = (
        len(datas) if dataset_ending_index == -1 else dataset_ending_index
    )
    datas = datas[dataset_starting_index:dataset_ending_index]

    logging_info(f"Load {len(datas)} samples")
    return datas


def all_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            return False
    return True


def main():
    ###################################################################################
    # prepare config, logger and recorder
    ###################################################################################
    config = Config(default_config_file="configs/celebahq.yaml", use_argparse=True)
    # config.show()

    all_paths = get_all_paths(config.outdir)
    config.dump(all_paths["path_config"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    # recorder = ResultRecorder(
    #     path_record=all_paths["path_record"],
    #     initial_record=config,
    #     use_git=config.use_git,
    # )
    set_random_seed(config.seed, deterministic=False, no_torch=False, no_tf=True)

    ###################################################################################
    # prepare data
    ###################################################################################
    if config.input_image == "":  # if --input_image is not given, load dataset
        datas = prepare_data(
            config.dataset_name,
            config.mask_type,
            config.dataset_starting_index,
            config.dataset_ending_index,
        )
    else:
        # NOTE: the model should accepet this input image size
        image = normalize(Image.open(config.input_image).convert("RGB"))
        if config.mode != "super_resolution":
            mask = (
                torch.from_numpy(np.array(Image.open(config.mask).convert("1"), dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            mask = torch.from_numpy(np.array([0]))  #config.mode是"super_resolution"时, mask is just a dummy value

        datas = [(image, mask, "sample0")]

    ###################################################################################
    # prepare model and device
    ###################################################################################
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    unet, sampler = prepare_model(config.algorithm, config, device)

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return unet(x, t, y if config.class_cond else None, gt=gt)
    
    cond_fn = None

    METRICS = {
        "lpips": Metric(LPIPS("alex", device)), # 衡量人眼感知上的差异，但不是像素级的。越小越好（越接近原图，人眼越看不出差别）
        "psnr": Metric(PSNR(), eval_type="max"), # 衡量图像的像素级误差大小（与原图的差别）。越大越好
        "ssim": Metric(SSIM(), eval_type="max"), # 衡量图像的结构、亮度、对比度的相似度。越大越好
    }
    final_loss = []

    ###################################################################################
    # start sampling
    ###################################################################################
    logging_info("Start sampling")
    timer, num_image = Timer(), 0
    batch_size = config.n_samples

    for data in tqdm(datas):
        if config.class_cond: # 这个是false
            image, mask, image_name, class_id = data
        else:
            image, mask, image_name = data # 上面定义过 datas = [(image, mask, "half")]  image_name是sample0
            class_id = None
        # prepare save dir
        outpath = os.path.join(config.outdir, image_name)
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples") # 路径最后一部分被指定为了"samples" # # # # # # # # # # # # # # # #
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path)) # 含义：当前sample_path目录下已有的图像样本数量。
        grid_count = max(len(os.listdir(outpath)) - 3, 0) # 含义：估算当前图像对应的结果图像网格（如拼图图）数量。

        # image: [1, 3, 256, 256]
        # mask: [1, 256, 256]
        outpath = os.path.join(config.outdir, image_name)
        os.makedirs(outpath, exist_ok=True)
        # 扩展 mask 维度以匹配 image
        mask_expanded = mask.unsqueeze(1)  # -> [1, 1, 256, 256]
        # 生成 masked 图像
        masked_image = image * mask_expanded  # 自动广播到 [1, 3, 256, 256]
        # # # # # # # # # # # # # # # # 保存 gt 与 masked # # # # # # # # # # # # # # # # # # #
        save_grid(normalize_image(image), os.path.join(outpath, f"gt_{base_count}.png"))
        save_grid(normalize_image(masked_image), os.path.join(outpath, f"masked_{base_count}.png"))

        # prepare batch data for processing
        batch = {"image": image.to(device), "mask": mask.to(device)}
        model_kwargs = {
            "gt": batch["image"].repeat(batch_size, 1, 1, 1),
            "gt_keep_mask": batch["mask"].repeat(batch_size, 1, 1, 1),
        }
        # if config.class_cond:
        #     if config.cond_y is not None:
        #         classes = torch.ones(batch_size, dtype=torch.long, device=device)
        #         model_kwargs["y"] = classes * config.cond_y
        #     elif config.classifier_path is not None:
        #         classes = torch.full((batch_size,), class_id, device=device)
        #         model_kwargs["y"] = classes

        shape = (batch_size, 3, config.image_size, config.image_size)

        all_metric_paths = [
            os.path.join(outpath, i + ".txt")
            for i in (list(METRICS.keys()) + ["final_loss"])
        ]


        if config.get("resume", False) and all_exist(all_metric_paths):
            # 接着上一次的权重继续的
            #把已经保存的指标文件.last (我已经换为了.txt)
            # 加载回来，继续加到当前的
            # metric.dataset_scores
            # 上，防止重复计算。
            # for metric_name, metric in METRICS.items():
            #     metric.dataset_scores += torch.load(
            #         os.path.join(outpath, metric_name + ".txt")
            #     )
            for metric_name, metric in METRICS.items():
                data_path = os.path.join(outpath, metric_name + ".txt")
                loaded_scores = np.loadtxt(data_path).tolist()
                metric.dataset_scores += loaded_scores
            logging_info("Results exists. Skip!")
        else:
            # 否则去采样
            # sample images
            base_idx = base_count
            samples = []
            for n in range(config.n_iter): # config.n_iter = 1：每张图像只采样一次（典型设置） config.n_iter = 4：每张图像采样 4 次，得到 4 个不同的修复版本
                timer.start()
                result = sampler.p_sample_loop( # # p_sample_loop # #
                    model_fn,
                    shape=shape,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    progress=True,
                    return_all=True,
                    conf=config,
                    sample_dir=outpath if config["debug"] else None,
                )
                timer.end()

                for metric in METRICS.values():
                    metric.update(result["sample"], batch["image"])
                # 这句话是在采样完成之后，将生成的图像 result["sample"] 和原始图像 batch["image"] 送入各个评价指标对象中，
                # 调用它们的 update() 方法更新得分。

                if "loss" in result.keys() and result["loss"] is not None:
                    # recorder.add_with_logging(
                    #     key=f"loss_{image_name}_{n}", value=result["loss"]
                    # )
                    final_loss.append(result["loss"])
                else:
                    final_loss.append(None)

                inpainted = normalize_image(result["sample"])
                samples.append(inpainted.detach().cpu())

            samples = torch.cat(samples)

            # save generations
            for sample in samples:
                save_image(sample, os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1

            # save metrics
            # ---- 保存每张图的指标分数到对应的 txt 文件（文件名 + 分数） ----
            for metric_name, metric in METRICS.items():
                score_values = metric.dataset_scores[-config.n_iter:]
                score_lines = [
                    f"{base_idx + i:05d}.png {score_values[i].item():.6f}"
                    for i in range(len(score_values))
                ]
                with open(os.path.join(outpath, metric_name + ".txt"), "a") as f:
                    for line in score_lines:
                        f.write(line + "\n")

            # ---- 保存 final_loss，每张图一行 ----
            if final_loss[0] is not None:
                lines = [
                    f"{base_idx + i:05d}.png {final_loss[-config.n_iter + i]:.6f}"
                    for i in range(config.n_iter)
                ]
                with open(os.path.join(outpath, "final_loss.txt"), "a") as f:
                    for line in lines:
                        f.write(line + "\n")

        # report batch scores
        # 汇报每张图像（image_name）在所有指标下的得分
        # 写入日志或文件的 key----value 类似于： PSNR_score_00001.png----0.00125
        # for metric_name, metric in METRICS.items():
        #     recorder.add_with_logging(
        #         key=f"{metric_name}_score_{image_name}",
        #         value=metric.report_batch(),
        #     )

    # report over all results
    # 汇报所有图像的总体统计信息（平均值和最优平均）
    # for metric_name, metric in METRICS.items():
    #     mean, colbest_mean = metric.report_all()
    #     recorder.add_with_logging(key=f"mean_{metric_name}", value=mean)
    #     recorder.add_with_logging(
    #         key=f"best_mean_{metric_name}", value=colbest_mean)
    # if len(final_loss) > 0 and final_loss[0] is not None:
    #     recorder.add_with_logging(
    #         key="final_loss",
    #         value=np.mean(final_loss),
    #     )
    # if num_image > 0:
    #     recorder.add_with_logging(
    #         key="mean time", value=timer.get_cumulative_duration() / num_image
    #     )

    logging_info(
        f"Your samples are ready and waiting for you here: \n{config.outdir} \n"
        f" \nEnjoy."
    )
    # recorder.end_recording()


if __name__ == "__main__":
    main()


# .yaml中的x_t优化参数
# optimize_xt:
#   optimize_xt: true                     # 是否启用 x_t 优化
#   num_iteration_optimize_xt: 2          # 对每一个时间步的 x_t 迭代优化 2次
#   lr_xt: 0.02                           # 初始学习率 lr_xt，用于梯度下降更新：new_x = x - lr_xt * x_grad

#   lr_xt_decay: 1.012                    # 每一步优化之后，学习率乘以这个衰减因子，使得后期更新更小，避免震荡

#   use_smart_lr_xt_decay: true           # 开启自适应学习率调整策略
#   use_adaptive_lr_xt: true              # 每次尝试更新 x 后，会比较 loss 是否下降，如果没有就降低 lr_xt（乘0.8），重新尝试每次尝试更新 x 后，会比较 loss 是否下降，如果没有就降低 lr_xt（乘0.8），重新尝试
#   coef_xt_reg: 0.0001                   # 控制 reg_fn(origin_x, x) 正则项的权重，惩罚 x 偏离 origin_x 太远，防止发散或过拟合

#   coef_xt_reg_decay: 1.01               # 每一次时间步优化后，将 coef_xt_reg 乘以这个值，逐渐加大正则强度，鼓励结果保持稳定。

#   mid_interval_num: 1                   # 控制 get_predx0(..., interval_num=1) 中使用的 step 数。设为 1 表示只预测当前 step。
#   optimize_before_time_travel: true     # 如果你启用了 RePaint 中的时间旅行（比如跳步、重采样），这个选项表示在时间跳跃之前，先对 x_t 做一次优化。
#   filter_xT: false                      # 有些方法会在 t=T（最开始的纯噪声）上加一些筛选或先验修正，比如过滤异常噪声。 设置为 false 表示你 不对初始噪声 x_T 做任何处理，直接从随机采样开始。