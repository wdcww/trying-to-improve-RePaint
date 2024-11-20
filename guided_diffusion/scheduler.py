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

# def get_schedule(t_T, t_0, n_sample, n_steplength, debug=0):
#     if n_steplength > 1:
#         if not n_sample > 1:
#             raise RuntimeError('n_steplength has no effect if n_sample=1')
#
#     t = t_T
#     times = [t]
#     while t >= 0:
#         t = t - 1
#         times.append(t)
#         n_steplength_cur = min(n_steplength, t_T - t)
#
#         for _ in range(n_sample - 1):
#
#             for _ in range(n_steplength_cur):
#                 t = t + 1
#                 times.append(t)
#             for _ in range(n_steplength_cur):
#                 t = t - 1
#                 times.append(t)
#
#     _check_times(times, t_0, t_T)
#
#     if debug == 2:
#         for x in [list(range(0, 50)), list(range(-1, -50, -1))]:
#             _plot_times(x=x, times=[times[i] for i in x])
#
#     return times


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


# def _plot_times(x, times):
#     import matplotlib.pyplot as plt
#     plt.plot(x, times)
#     plt.show()


# def get_schedule_jump(t_T,
#                       n_sample,
#                       jump_length,
#                       jump_n_sample,
#                       jump2_length=1,
#                       jump2_n_sample=1,
#                       jump3_length=1,
#                       jump3_n_sample=1,
#                       start_resampling=100000000):
#     """
#     返回值是一个列表 ts，其中包含以下内容：
#     从 t_T 开始向下递减的时间步。会根据条件插入的时间步（例如通过 n_sample、jump_length 等参数控制的跳跃）。
#     最后会附加一个 -1，表示结束。
#
#     :param n_sample 在时间步 t 到达某个条件时，函数会根据这个参数添加额外的时间步到 ts 中
#     """
#     # 初始化一个空字典jumps
#     jumps = {}
#     # 使用for循环遍历从0开始，到小于t_T - jump_length为止，步长为jump_length的整数序列
#     for j in range(0, t_T - jump_length, jump_length):
#         # 对于序列中的每个j，字典jumps的键j的值都是下面的 jump_n_sample - 1
#         jumps[j] = jump_n_sample - 1
#     # # 初始化一个空字典jumps2
#     # jumps2 = {}
#     # for j in range(0, t_T - jump2_length, jump2_length):
#     #     # 字典jumps2中存储的是 "j"——"jump2_n_sample-1"
#     #     jumps2[j] = jump2_n_sample - 1
#     # # 初始化一个空字典jumps3
#     # jumps3 = {}
#     # for j in range(0, t_T - jump3_length, jump3_length):
#     #     # 字典jumps3中是 "j"——"jump3_n_sample-1"
#     #     jumps3[j] = jump3_n_sample - 1
#
#     t = t_T
#     ts = []
#
#     while t >= 1: # while循环中,t从t_T开始递减到1,表示总的时间步数
#         t = t-1
#         ts.append(t)
#         # # 每一步的t都会Resampling
#         # if ( t + 1 < t_T - 1 and t <= start_resampling ): # 进入此if：折磨xt,本来要减的xt反而去加...又减...回到xt
#         #     for _ in range(n_sample - 1):
#         #     # 循环 n_sample - 1 次。
#         #         t = t + 1 # 将 t 增加 1，生成下一个时间步。
#         #         ts.append(t)
#         #
#         #         if t >= 0:
#         #             t = t - 1 # 如果 t 仍然是非负的（即在有效范围内），则将 t 减去 1
#         #             ts.append(t)
#
#         # if ( jumps3.get(t, 0) > 0 and t <= start_resampling - jump3_length ):
#         #     jumps3[t] = jumps3[t] - 1
#         #     for _ in range(jump3_length):
#         #         t = t + 1
#         #         ts.append(t)
#         #
#         # if ( jumps2.get(t, 0) > 0 and t <= start_resampling - jump2_length ):
#         #     jumps2[t] = jumps2[t] - 1
#         #     for _ in range(jump2_length):
#         #         t = t + 1
#         #         ts.append(t)
#         #     jumps3 = {}
#         #     for j in range(0, t_T - jump3_length, jump3_length):
#         #         jumps3[j] = jump3_n_sample - 1
#
#         if ( jumps.get(t, 0) > 0 and t <= start_resampling - jump_length ):
#             jumps[t] = jumps[t] - 1
#             for _ in range(jump_length):
#                 t = t + 1
#                 ts.append(t)
#             # jumps2 = {}
#             # for j in range(0, t_T - jump2_length, jump2_length):
#             #     # 和最开始一摸一样的？字典jumps2中存储的是 "j"——"jump2_n_sample-1"
#             #     jumps2[j] = jump2_n_sample - 1
#             #
#             # jumps3 = {}
#             # for j in range(0, t_T - jump3_length, jump3_length):
#             #     # 也和最开始初始化的一模一样？字典jumps3中是 "j"——"jump3_n_sample-1"
#             #     jumps3[j] = jump3_n_sample - 1
#
#     ts.append(-1)
#
#     _check_times(ts, -1, t_T)
#
#     return ts

# ####### 禁用Resampling版本：
# def get_schedule_jump(t_T,n_sample=2,
#                            jump_length=10, jump_n_sample=10,
#                            jump2_length=1, jump2_n_sample=1,
#                            jump3_length=1, jump3_n_sample=1,
#                            start_resampling=100000000):
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

# ###### 简单的Resampling版本：
def get_schedule_jump(t_T,
                      n_sample,
                      jump_length,
                      jump_n_sample,
                      jump2_length=1,
                      jump2_n_sample=1,
                      jump3_length=1,
                      jump3_n_sample=1,
                      start_resampling=100000000):
    """
    返回值是一个列表 ts，我搞一个最简单的，仅需要：
    t_T : 总步数

    jump_length : 在所有的t_T步里，是jump_length的倍数的点算一个‘特殊点’，但最接近t_T那个倍数值点不算'特殊点'

    jump_n_sample ：在这些'特殊点'操作几次？jump_n_sample-1 次
    """
    # 初始化一个空字典jumps
    jumps = {}
#     使用for循环遍历从0开始，到小于t_T - jump_length为止，步长为jump_length的整数序列
    for j in range(0, t_T - jump_length, jump_length):
        # 对于序列中的每个j，字典jumps的键j的值都是下面的 jump_n_sample - 1
        jumps[j] = jump_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:  # while循环中,t从t_T开始递减到1,表示总的时间步数
        t = t - 1
        ts.append(t)

        if (jumps.get(t, 0) > 0 and t <= start_resampling - jump_length):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, t_T)
    # print("scheduler.py的ts:  ",ts)
    return ts

# def get_schedule_jump_paper():
#     t_T = 250
#     jump_length = 10
#     jump_n_sample = 10
#
#     jumps = {}
#     for j in range(0, t_T - jump_length, jump_length):
#         jumps[j] = jump_n_sample - 1
#
#     t = t_T
#     ts = []
#
#     while t >= 1:
#         t = t-1
#         ts.append(t)
#
#         if jumps.get(t, 0) > 0:
#             jumps[t] = jumps[t] - 1
#             for _ in range(jump_length):
#                 t = t + 1
#                 ts.append(t)
#
#     ts.append(-1)
#
#     _check_times(ts, -1, t_T)
#
#     return ts


def get_schedule_jump_test(to_supplement=False):
    """
       减少 t_T，即总步数。它越低，每一步消除的噪音就越多。肯定呀，用更少的步骤从T到0，那么每步去噪就多了
       减少jump_n_sample以减少重新采样的次数。
       不是从一开始就应用重采样，而是通过设置 start_resampling 在特定时间之后应用重采样。
       """
    ts = get_schedule_jump(t_T=20, n_sample=2,
                           jump_length=10, jump_n_sample=10,
                           jump2_length=1, jump2_n_sample=1,
                           jump3_length=1, jump3_n_sample=1,
                           start_resampling=100000000)
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
