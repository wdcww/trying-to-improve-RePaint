Hi everyone, I'm trying to improve the code in the RePaint repository.
I created this repository to back up some of my own modifications.

I refer to the code in repository [RePaint](https://github.com/andreas128/RePaint) and the code in the [guided-diffuion](https://github.com/openai/guided-diffusion) repository that RePaint is based on.


## 提交记录
```
commit 1 : 重新添加了ddim采样
commit 2 : 把scheduler.py好好整理了一下, 同时.yml中schedule_jump_params不再需要n_sample
commit 3 : ddpm的p_sample()添加了"预测噪声"版本的采样,详情见 issue 1 
commit 4 : 为方便阅读删去了之前注释掉的代码；从《ILVR》搞过来一点东西,详情见 issue 2；
commit 5 : 去掉了commit_4的ILVR; 修改main.py使得 一张gt可得到某张mask的 n_iter张 采样结果; 详情见 issue 3  

```