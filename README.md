Hi everyone, I'm trying to improve the code in the RePaint repository.
I created this repository to back up some of my own modifications.

I refer to the code in repository [RePaint](https://github.com/andreas128/RePaint) and the code in the [guided-diffuion](https://github.com/openai/guided-diffusion) repository that RePaint is based on.


## 提交记录
```
commit 1 : 重新添加了ddim采样
commit 2 : 把scheduler.py好好整理了一下, 同时.yml中schedule_jump_params不再需要n_sample
commit 3 : ddpm的p_sample()添加了"预测噪声"版本的采样,详情见 issue 1 

```