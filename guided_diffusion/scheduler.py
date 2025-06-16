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

def _check_times(times, t_0, t_T):
    # Check end
    # 要求时间序列中的第一个时间步 times[0] 要大于第二个时间步 times[1]。
    # 如果不满足这个条件，会抛出一个错误，并显示当前这两个时间步的值: (times[0], times[1])
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    # 这条断言要求时间序列的最后一个时间步 times[-1] 必须是 -1。
    # 如果不是 -1，会抛出一个错误，并显示该时间步的值: times[-1]
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    # 这部分检查时间步之间的差值是否始终为 1，即相邻的时间步是否是连续的。
    # 如果差值不是 1，会抛出一个错误，并显示出这两个时间步的值: (t_last, t_cur)
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        # 这条断言检查每个时间步 t 。
        # 如果某个时间步小于 t_0，会抛出错误并显示该时间步与 t_0 的值。(t, t_0)
        # 如果某个时间步大于 t_T，会抛出错误并显示该时间步与 t_T 的值。(t, t_T)
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)

def _check_times_1(times, t_0, t_T):
    """
    专为 “每隔5个时间步，跳过5个时间步”版本 使用的函数
    其实就是：注释掉 _check_times()中的“检查时间步之间的差值是否始终为1”
    """
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)

# # #############################################################################


# ####### 禁用Resampling版本：
# def get_schedule_jump(t_T,
#                       jump_length=10, jump_n_sample=10,
#                            ):
#     """
#     返回值是一个递减的列表 ts，仅需要：
#     t_T : 总步数
#     """
#     t = t_T
#     ts = []
#     while t >= 1:  # while循环中,t从t_T开始递减到1,表示总的时间步数
#         t = t - 1
#         ts.append(t)
#     ts.append(-1)
#     _check_times(ts, -1, t_T)
#     return ts


# # #############################################################################


###### Resampling版本：
def get_schedule_jump(t_T,
                      jump_length,
                      jump_n_sample,
                      ):
    """
    t_T : 总步数

    jump_length : 在所有的t_T步里，是 jump_length的倍数 且 小于t_T- jump_length 的点算一个‘特殊点’，
                  在'特殊点'处，就是处于Resample状态，每进入Resample状态，需要给当前的t加jump_length个点。

    jump_n_sample : 对于某个我们能取到的'特殊点'，该点一共会经历 jump_n_sample-1 次Resample状态


    """
    # 初始化一个空字典jumps
    jumps = {}
    # 使用for循环遍历从0开始，到小于t_T - jump_length为止，步长为jump_length的整数序列
    for j in range(0, t_T - jump_length, jump_length):
        # 对于序列中的每个j，字典jumps的键j的值都是下面的 jump_n_sample - 1
        jumps[j] = jump_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:  # while循环中,t从t_T开始递减到1,表示总的时间步数
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0 :
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, t_T)
    # print("scheduler.py的ts:  ",ts)
    return ts

# # #############################################################################

# # # “每隔5个时间步，跳过5个时间步”版本：
# def get_schedule_jump(t_T,
#                       jump_length,jump_n_sample,
#                       ):
#     """
#     返回值是一个列表 ts : 在每添加 5 个连续点后，跳过接下来的 5 个点。
#     t_T : 总步数
#     """
#     ts = []
#     t = t_T - 1
#     consecutive_count = 0 # 当前添加的连续点数
#     while t >= 0:  # while循环中,t从t_T-1开始递减到0
#         ts.append(t)
#         consecutive_count += 1
#
#         # 每添加 5 个连续点，就跳过接下来的 5 个点
#         if consecutive_count == 5:
#             t -= 5  # 跳过 5 个点
#             consecutive_count = 0  # 重置连续计数器
#
#         t -= 1
#     # 添加 -1 作为最后的点
#     ts.append(-1)
#     _check_times_1(ts, -1, t_T)
#     # print("每隔5个时间步，跳过5个时间步 的ts: ", ts)
#     return ts


def get_schedule_jump_test(to_supplement=False):
    """
       减少 t_T，即总步数。它越低，每一步消除的噪音就越多。肯定呀，用更少的步骤从T到0，那么每步去噪就多了
       减少jump_n_sample以减少重新采样的次数。
       不是从一开始就应用重采样，而是通过设置 start_resampling 在特定时间之后应用重采样。
       """
    ts = get_schedule_jump(t_T=50,
                           jump_length=10, jump_n_sample=2
                           )
    print(len(ts))
    print(ts) ####################################################### print一下！
    import matplotlib.pyplot as plt
    SMALL_SIZE = 8*3
    MEDIUM_SIZE = 10*3
    BIGGER_SIZE = 12*3

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.plot(ts)
    plt.show() ################################################# show一下!
    fig = plt.gcf()
    fig.set_size_inches(20, 10)

    ax = plt.gca()
    ax.set_xlabel('Number of Transitions')
    ax.set_ylabel('Diffusion time $t$')

    fig.tight_layout()
    # if to_supplement:
    #     out_path = "/cluster/home/alugmayr/gdiff/paper/supplement/figures/jump_sched.pdf"
    #     plt.savefig(out_path)

    # out_path = "./schedule_n_sample_my.png"
    # plt.savefig(out_path)
    # print(out_path)


def main():
    get_schedule_jump_test()


if __name__ == "__main__":
    main()
