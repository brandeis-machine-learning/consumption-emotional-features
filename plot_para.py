from turtle import color
from unicodedata import category
import matplotlib.pylab as plt
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
from sympy import false
from torch import alpha_dropout

del font_manager.weight_dict['roman']
font_manager._rebuild()
rcParams['font.family'] = 'Times New Roman'
rcParams["mathtext.fontset"] = 'stix'
FONT_SIZE = 16
parameters = {'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
          'legend.fontsize': FONT_SIZE,
          'xtick.labelsize': FONT_SIZE,'ytick.labelsize':FONT_SIZE}
plt.rcParams.update(parameters)


# # ==============draw features of samples================
file_name = "feature_sept5_.csv"
data = pd.read_csv(file_name)

label_data = data.iloc[:,2:4].values
plot_data = data.iloc[:,4:].values

norm = np.max(np.abs(plot_data), axis=0)
for i in range(len(norm)):
    if norm[i] >= 100:
        norm[i] = 100.0
    elif norm[i] >= 10:
        norm[i] = 10.0
    elif norm[i] >= 1:
        norm[i] = 1.0
    elif norm[i] >= 0.1:
        norm[i] = 0.1
    else:
        norm[i] = 0.01
plot_data = plot_data / norm


pick1 = plot_data[53]
pick2 = plot_data[97]

neg_set, neg_set_e = set(), set()
pos_set, pos_set_e = set(), set()
# view
highlight_neg = [[2,3], [5,6], [9,10], [10,11], [15,16], [18,19], [21,22], [22,23], [25,26], [29,30]]
highlight_pos = [[0,1], [1,2], [3,4], [4,5], [6,7], [12,13], [24,25]]
for spot in highlight_neg:
    # plt.axvspan(spot[0], spot[1], color='mistyrose', lw=0)
    neg_set.add(spot[0])
for spot in highlight_pos:
    # plt.axvspan(spot[0], spot[1], color='palegreen', lw=0)
    pos_set.add(spot[0])
# # emoition
# highlight_neg = [[14,15], [23,24], [28,29], [29,30], [32,33], [36,37]]
# highlight_pos = [[12,13], [30,31], [31,32]]
# for spot in highlight_neg:
#     plt.axvspan(spot[0], spot[1], color='mistyrose', lw=0)
#     neg_set_e.add(spot[0])
# for spot in highlight_pos:
#     plt.axvspan(spot[0], spot[1], color='palegreen', lw=0)
#     pos_set_e.add(spot[0])
# # overlap
# # highlight_overlap = [[14,15], [36,37]]
# # for spot in highlight_overlap:
# #     plt.axvspan(spot[0], spot[1], color='lightgray', lw=0)
# #     pos_set.remove(spot[0])
# #     neg_set_e.remove(spot[0])


# x = np.arange(len(pick1)) + 0.5
# fig, ax = plt.subplots()
# l1=plt.plot(x, pick1,'tab:orange', label='S3E8', lw=1, marker='x')
# l2=plt.plot(x, pick2,'tab:blue', label='S5E10', lw=1, marker='.')

# # ax.set_xlabel('Episode', fontsize = 12)
# ax.set_ylabel('Normalized Value', fontsize = 12)
# # ax.set_xlabel('Variables', fontsize = 12)
# # ax.set_title('View')
# ax.set_xticks([0, 6, 18, 22, 26, 30, 34, 38, 42])
# ax.set_xticklabels([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
# plt.xlim([0, 42])
# plt.legend(loc='upper right')
# fig.set_size_inches(10, 3.5)
# plt.tight_layout()
# plt.show()


# ==============draw linear fit wave================

from generate_tables import emotion_wave, find_peaks, get_peak, thresholding_algo
import matplotlib.patches as patches
FONT_SIZE = 16  # 12, 16
LINE_WIDTH = 2    # 1.5, 2

def get_peak_2(time_series):
    indices = find_peaks(time_series,prominence=10)[0]
    try:
        x = np.array(indices)
        y = np.array([time_series[j] for j in indices])
        z1 = np.polyfit(x, y, 1)
        peak_end = x[-1]
    except Exception:
        return 0,0,indices,np.max(time_series),np.argmax(time_series)
    return z1[0], z1[1], indices.tolist(), np.max(time_series), np.argmax(time_series)


def draw_wave(season=1, episode=3):
    emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, _, _ = emotion_wave(season, episode)
    end_time = 124
    for i in range(len(emotion_all_percent_abs)):
        if emotion_all_percent_abs[123-i] != 0:
            end_time = 124-i
            break
    emotion_negative = emotion_negative[:end_time][79:111]
    emotion_positive = emotion_positive[:end_time][79:111]

    duration_signal = get_peak(emotion_all_percent)[79:111]

    x = np.arange(len(emotion_negative))
    fig, ax = plt.subplots()
    plt.plot(x, emotion_positive,'tab:blue', label='positive emotion wave', lw=LINE_WIDTH, zorder=2)
    plt.plot(x, emotion_negative,'salmon', label='negative emotion wave', lw=LINE_WIDTH, zorder=2)


    # ====== linear fitting ========
    pos_k,pos_b,pos_idx,pos_max,pos_argmax = get_peak_2(emotion_positive)
    neg_k,neg_b,neg_idx,neg_max,neg_argmax = get_peak_2(emotion_negative)
    pos_line = x * pos_k + pos_b
    neg_line = x * neg_k + neg_b
    pos_idx.remove(pos_argmax)
    neg_idx.remove(neg_argmax)
    plt.scatter(pos_idx, np.array(emotion_positive)[pos_idx], marker='^', s=100,color='darkblue', label='local positive peak', zorder=3)
    plt.scatter(neg_idx, np.array(emotion_negative)[neg_idx], marker='^', s=100,color='orangered', label='local negative peak', zorder=3)
    plt.scatter([pos_argmax], [pos_max], marker='*', s=300, color='darkblue', label='segment positive peak', zorder=4)
    plt.scatter([neg_argmax], [neg_max], marker='*', s=300, color='orangered', label='segment negative peak', zorder=4)
    plt.plot(x, pos_line,'darkblue', label='linear fitting over local positive peaks', lw=LINE_WIDTH, linestyle='dashed', dashes=(10, 2), zorder=3)
    plt.plot(x, neg_line,'orangered', label='linear fitting over local negative peaks', lw=LINE_WIDTH, linestyle='dashed', dashes=(10, 2), zorder=3)


    # ====== duration ========
    pos_label, neg_label = True, True
    for i in range(len(duration_signal)):
        diff = duration_signal[i]
        if diff > 0:
            if pos_label:
                plt.axvspan(i-0.5, i+0.5, color='lightblue', lw=0, label='positive momentum', zorder=1)
                pos_label = False
            else:
                plt.axvspan(i-0.5, i+0.5, color='lightblue', lw=0, zorder=1)
        elif diff < 0:
            if neg_label:
                plt.axvspan(i-0.5, i+0.5, color='mistyrose', lw=0, label='negative momentum', zorder=1)
                neg_label = False
            else:
                plt.axvspan(i-0.5, i+0.5, color='mistyrose', lw=0, zorder=1)
                
    # ====== wave ========
    # build a rectangle in axes coords
    # left, width = .25, .4
    # bottom, height = .1, .8
    # right = left + width
    # top = bottom + height
    # ax.text(right, top, r'$emotion\_comment$: 0.0414', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=FONT_SIZE)
    # ax.text(right, top, r'$\# view$: 3.8873', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=FONT_SIZE)


    # ====== view ========
    ax.set_ylabel('Emotion Valence', fontsize=FONT_SIZE)
    ax.set_xlabel('Moment (minute)', fontsize=FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE)
    plt.xlim([0, len(emotion_negative)-1])

    # plt.ylim([-10, 290]) # 220, 
    # plt.legend(loc='lower center', fontsize=FONT_SIZE, bbox_to_anchor=(0.7, 0.0), ncol=1)
    # fig.set_size_inches(7, 4)

    ax.set_xticks(np.arange(0,32,5))
    ax.set_xticklabels(np.arange(0,32,5)+79)
    plt.legend(loc='lower left', fontsize=FONT_SIZE*0.8, ncol=5, bbox_to_anchor=(-0.012, -0.25))
    fig.set_size_inches(15, 7)

    plt.tight_layout()
    # plt.savefig('figures/emotion_wave_S' + str(season) + '_E' + str(episode) + '.pdf')
    plt.savefig('figures/k.pdf')
    plt.show()





def draw_2waves(season=1, episode=3):
    emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, _, _ = emotion_wave(season, episode)
    end_time = 124
    for i in range(len(emotion_all_percent_abs)):
        if emotion_all_percent_abs[123-i] != 0:
            end_time = 124-i
            break
    emotion_negative = emotion_negative[:end_time][79:111]
    emotion_positive = emotion_positive[:end_time][79:111]

    duration_analysis = thresholding_algo(emotion_all_percent, lag=15, threshold=1, influence=0.5)
    duration_signal = duration_analysis['signals'][79:111]
    duration_avg = duration_analysis['avgFilter'][79:111]
    duration_std = duration_analysis['stdFilter'][79:111]

    x = np.arange(len(emotion_negative))
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    l1, = ax.plot(x, emotion_positive,'tab:blue', label='positive emotion wave', lw=LINE_WIDTH, zorder=2)
    l2, = ax.plot(x, emotion_negative,'salmon', label='negative emotion wave', lw=LINE_WIDTH, zorder=2)


    # ====== linear fitting ========
    pos_k,pos_b,pos_idx,pos_max,pos_argmax = get_peak_2(emotion_positive)
    neg_k,neg_b,neg_idx,neg_max,neg_argmax = get_peak_2(emotion_negative)
    pos_line = x * pos_k + pos_b
    neg_line = x * neg_k + neg_b
    pos_idx.remove(pos_argmax)
    neg_idx.remove(neg_argmax)
    s1 = ax.scatter(pos_idx, np.array(emotion_positive)[pos_idx], marker='^', s=100,color='darkblue', label='local positive peak', zorder=3)
    s2 = ax.scatter(neg_idx, np.array(emotion_negative)[neg_idx], marker='^', s=100,color='orangered', label='local negative peak', zorder=3)
    s3 = ax.scatter([pos_argmax], [pos_max], marker='*', s=300, color='darkblue', label='segment positive peak', zorder=4)
    s4 = ax.scatter([neg_argmax], [neg_max], marker='*', s=300, color='orangered', label='segment negative peak', zorder=4)
    l3, = ax.plot(x, pos_line,'darkblue', label='linear fitting over local positive peaks', lw=LINE_WIDTH, linestyle='dashed', dashes=(10, 2), zorder=3)
    l4, = ax.plot(x, neg_line,'orangered', label='linear fitting over local negative peaks', lw=LINE_WIDTH, linestyle='dashed', dashes=(10, 2), zorder=3)


    # ====== duration ========
    l5, = ax2.plot(x, np.array(emotion_positive)-np.array(emotion_negative),'black', label='emotion wave of pos. minus neg.', lw=LINE_WIDTH, zorder=2)

    l6, = ax2.plot(x, duration_avg,'tab:green', label='sliding window avg.', lw=LINE_WIDTH, zorder=2, ls='--')

    f1 = ax2.fill_between(x, y1=duration_avg-duration_std, y2=duration_avg+duration_std, color='tab:green', alpha=0.2, zorder=2, label='sliding window avg. ± std.')


    pos_label, neg_label = True, True
    for i in range(len(duration_signal)):
        diff = duration_signal[i]
        if diff > 0:
            if pos_label:
                f2 = ax2.axvspan(i-0.5, i+0.5, color='lightblue', lw=0, label='positive momentum', zorder=1)
                pos_label = False
            else:
                ax2.axvspan(i-0.5, i+0.5, color='lightblue', lw=0, zorder=1)
        elif diff < 0:
            if neg_label:
                f3 = ax2.axvspan(i-0.5, i+0.5, color='mistyrose', lw=0, label='negative momentum', zorder=1)
                neg_label = False
            else:
                ax2.axvspan(i-0.5, i+0.5, color='mistyrose', lw=0, zorder=1)

    e1, = ax2.plot([0,1], [0,0], color='white', alpha=0.0, label=' ')
                

    # ====== view ========
    ax.set_ylabel('Emotion Valence', fontsize=FONT_SIZE)
    ax2.set_ylabel('Emotion Valence', fontsize=FONT_SIZE)
    ax2.set_xlabel('Moment (minute)', fontsize=FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE)
    plt.xlim([0, len(emotion_negative)-1])
    ax2.set_xticks(np.arange(0,32,5))
    ax2.set_xticklabels(np.arange(0,32,5)+79)


    # labels = ['positive emotion wave', 'negative emotion wave', 'emotion wave of pos. minus neg.',
    #           'linear fitting over local positive peaks', 'linear fitting over local negative peaks', 'sliding window avg.',
    #           ' ', ' ', 'local positive peak', 'local negative peak', 'segment positive peak', 'segment negative peak',
    #           'positive momentum', 'negative momentum', 'sliding window avg. ± std.']
    # plt.legend(handles=[l1, l2, l5, l3, l4, l6, e1, e1, s1, s2, s3, s4, f2, f3, f1], labels=labels, mode='expand', ncol=5)

    labels1 = ['positive emotion wave', 'negative emotion wave',
              'linear fitting over local positive peaks', 'linear fitting over local negative peaks',
              'local positive peak', 'local negative peak', 'segment positive peak', 'segment negative peak']
    labels2 = ['emotion wave of positive minus negative', ' ',
               'sliding window average', 'sliding window average ± standard deviation', 
              'positive momentum', 'negative momentum']
    ax.legend(handles=[l1, l2, l3, l4, s1, s2, s3, s4], labels=labels1, mode='expand', ncol=4)
    ax2.legend(handles=[l5, e1, l6, f1, f2, f3], labels=labels2, mode='expand', ncol=3)
    fig.set_size_inches(15, 7)

    plt.tight_layout()
    plt.savefig('figures/k.pdf')
    plt.show()

draw_wave(6, 1)




# # ==============draw episode wave================

# from generate_tables import emotion_wave, find_peaks
# import matplotlib.patches as patches
# FONT_SIZE = 16  # 12, 16
# LINE_WIDTH = 2    # 1.5, 2

# def get_peak_2(time_series):
#     indices = find_peaks(time_series,prominence=10)[0]
#     try:
#         x = np.array(indices)
#         y = np.array([time_series[j] for j in indices])
#         z1 = np.polyfit(x, y, 1)
#         peak_end = x[-1]
#     except Exception:
#         return 0,0,indices,np.max(time_series),np.argmax(time_series)
#     return z1[0], z1[1], indices, np.max(time_series), np.argmax(time_series)


# def draw_wave(season=1, episode=3):
#     emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, _, _ = emotion_wave(season, episode)
#     end_time = 124
#     for i in range(len(emotion_all_percent_abs)):
#         if emotion_all_percent_abs[123-i] != 0:
#             end_time = 124-i
#             break
#     emotion_negative = emotion_negative[:end_time]#[79:111]
#     emotion_positive = emotion_positive[:end_time]#[79:111]

#     x = np.arange(len(emotion_negative))
#     fig, ax = plt.subplots()
#     plt.plot(x, emotion_positive,'tab:blue', label='positive emotion wave', lw=LINE_WIDTH, zorder=2)
#     plt.plot(x, emotion_negative,'salmon', label='negative emotion wave', lw=LINE_WIDTH, zorder=2)

#     ax.set_ylabel('Emotion Valence', fontsize=FONT_SIZE)
#     ax.set_xlabel('Timeline (minute)', fontsize=FONT_SIZE)
#     plt.tick_params(labelsize=FONT_SIZE)
#     plt.xlim([10, len(emotion_negative)-11])

#     # ====== view ========
#     if season==6 and episode==4:
#         plt.text(53, 25, 'view: 4.4461', fontsize=FONT_SIZE)
#         segs = [0,52,68,len(emotion_negative)-11]
#     elif season==6 and episode==21:
#         plt.text(53, 25, 'view: 4.1784', fontsize=FONT_SIZE)
#         segs = [0,49,71,len(emotion_negative)-11]
#     for i in range(len(segs)):
#         if i > 0 and i < len(segs)-1:
#             plt.vlines(segs[i], 0, 290, color='tab:gray', ls='--')
#         # if i > 0:
#         #     plt.text((segs[i-1] + segs[i])/2, 270, 'Segment ' + str(i), ha='center', va='center', fontsize=FONT_SIZE)

#     plt.ylim([0, 290]) # 220, 
#     plt.legend(loc='lower left', fontsize=FONT_SIZE, ncol=1)
#     fig.set_size_inches(7, 4)

#     plt.tight_layout()
#     plt.savefig('figures/emotion_wave_S' + str(season) + '_E' + str(episode) + '.pdf')
#     plt.show()

# draw_wave(6, 4)
# draw_wave(6, 21)



# # ==============select best samples================
# print(neg_set, neg_set_e)
# print(pos_set, pos_set_e)

# for i in range(30, len(plot_data)):
#     for j in range(30, len(plot_data)):
#         if i == j:
#             continue
#         count = 0.0
#         for k in neg_set:
#             if (plot_data[i,k] < plot_data[j,k] and label_data[i,0] < label_data[j,0]) or (plot_data[i,k] > plot_data[j,k] and label_data[i,0] > label_data[j,0]):
#                 count += 1
#             if plot_data[i,k] == plot_data[j,k] and plot_data[j,k] == 0:
#                 count += 0.5
#         for k in pos_set:
#             if (plot_data[i,k] > plot_data[j,k] and label_data[i,0] < label_data[j,0]) or (plot_data[i,k] < plot_data[j,k] and label_data[i,0] > label_data[j,0]):
#                 count += 1
#             if plot_data[i,k] == plot_data[j,k] and plot_data[j,k] == 0:
#                 count += 0.5
#         # for k in neg_set_e:
#         #     if (plot_data[i,k] < plot_data[j,k] and label_data[i,1] < label_data[j,1]) or (plot_data[i,k] > plot_data[j,k] and label_data[i,1] > label_data[j,1]):
#         #         count += 1
#         #     if plot_data[i,k] == plot_data[j,k] and plot_data[j,k] == 0:
#         #         count += 0.5
#         # for k in pos_set_e:
#         #     if (plot_data[i,k] > plot_data[j,k] and label_data[i,1] < label_data[j,1]) or (plot_data[i,k] < plot_data[j,k] and label_data[i,1] > label_data[j,1]):
#         #         count += 1
#         #     if plot_data[i,k] == plot_data[j,k] and plot_data[j,k] == 0:
#         #         count += 0.5
#         if count >= 3 or label_data[i,0] > label_data[j,0]:
#             continue
#         else:
#             print(count, 'correct:', i, j, label_data[i,1] < label_data[j,1])










# # ==============line for feature selection from graph================
# x_category = ['                pre_outcome', '        pos_duration', 'neg_duration', 'pos_k', 'neg_k', 'pos_b', 'neg_b']

# FONT_SIZE = 16
# dists = ['S6E4', 'S6E21']
# colors = ['tab:orange', 'tab:blue']
# x_label = data.columns[4:]
# x_label = ['']*len(x_label)
# # for i in [2, 8, 15, 20, 26, 33, 39]:
# #     x_label[i] = x_category[int(i/6)]


# list1 = plot_data[132]
# list2 = plot_data[115]

# norm = list1.copy()
# for i in range(len(norm)):
#     norm[i] = max(np.abs(list1[i]), np.abs(list2[i]))
#     if norm[i] >= 4:
#         norm[i] = 5.0
#     elif norm[i] >= 3:
#         norm[i] = 4.0
#     elif norm[i] >= 2:
#         norm[i] = 3.0
#     elif norm[i] >= 1:
#         norm[i] = 2.0
#     else:
#         norm[i] = 1.0
# list1 = list1 / norm
# list2 = list2 / norm


# x = np.arange(len(list1))
# # 绘图
# fig, ax = plt.subplots()
# # 绘制折线图
# ax.plot(x, list1, 's-', linewidth=2, label=dists[0], color=colors[0])
# # ax.fill(angles, list1, alpha=0.25, color=colors[0])
# # 绘制第二条折线图
# ax.plot(x, list2, '>-', linewidth=2, label=dists[1], color=colors[1])
# # ax.fill(angles, list2, alpha=0.25, color=colors[1])


# # view
# highlight_neg = [spot[0] for spot in highlight_neg]
# highlight_pos = [spot[0] for spot in highlight_pos]


# label = True
# for spot in highlight_neg:
#     if label:
#         plt.axvspan(spot-0.5, spot+0.5, color='mistyrose', lw=0, label='negative significant variable', zorder=0)
#         label = False
#     else:
#         plt.axvspan(spot-0.5, spot+0.5, color='mistyrose', lw=0, zorder=0)
# label = True
# for spot in highlight_pos:
#     if label:
#         plt.axvspan(spot-0.5, spot+0.5, color='lightblue', lw=0, label='positive significant variable', zorder=0)
#         label = False
#     else:
#         plt.axvspan(spot-0.5, spot+0.5, color='lightblue', lw=0, zorder=0)

# plt.xlim([-0.5, len(list1)-0.5])
# plt.ylim([-1,1])

# plt.ylabel('Normalized Value', fontsize=FONT_SIZE)
# ticks_y = [-1, -0.5, 0 ,0.5, 1]
# ax.set_yticks(ticks_y)
# ax.set_yticklabels(ticks_y, fontsize=FONT_SIZE)

# ticks = [0,4,12,18,24,len(list1)]
# ax.set_xticks(np.array(ticks)-0.5)
# ax.set_xticklabels(['']*len(ticks), fontsize=FONT_SIZE)
# plt.legend(loc='lower left', fontsize=FONT_SIZE, ncol=4, bbox_to_anchor=(0.1, -0.3))
# fig.set_size_inches(16, 5)
# plt.tight_layout()
# plt.savefig('figures/radar_emotion.png')
# plt.show()








# # ==============Radar for feature selection from graph================
# import seaborn as sns
# import matplotlib as mpl

# x_category = ['                pre_outcome', '        pos_duration', 'neg_duration', 'pos_k', 'neg_k', 'pos_b', 'neg_b']

# FONT_SIZE = 18
# dists = ['S3E8', 'S5E10']
# colors = ['tab:orange', 'tab:blue']
# x_label = data.columns[4:]
# x_label = ['']*len(x_label)
# # for i in [2, 8, 15, 20, 26, 33, 39]:
# #     x_label[i] = x_category[int(i/6)]


# list1 = plot_data[132]
# list2 = plot_data[115]

# norm = list1.copy()
# for i in range(len(norm)):
#     norm[i] = max(list1[i], list2[i])
#     if norm[i] >= 4:
#         norm[i] = 5.0
#     elif norm[i] >= 3:
#         norm[i] = 4.0
#     elif norm[i] >= 2:
#         norm[i] = 3.0
#     elif norm[i] >= 1:
#         norm[i] = 2.0
#     else:
#         norm[i] = 1.0
# list1 = list1 / norm
# list2 = list2 / norm

# # 使用ggplot的绘图风格
# # plt.style.use('ggplot')
# N = len(x_label)
# # 设置雷达图的角度，用于平分切开一个圆面
# angles=np.linspace(0, 2*np.pi, N, endpoint=False)
# # 为了使雷达图一圈封闭起来，需要下面的步骤
# list1=np.concatenate((list1,[list1[0]]))
# list2=np.concatenate((list2,[list2[0]]))
# angles=np.concatenate((angles,[angles[0]]))

# rotation_angle = 0.2*np.pi
# for i in range(len(angles)):
#     if angles[i] < 2*np.pi - rotation_angle:
#         angles[i] += rotation_angle
#     else:
#         angles[i] += rotation_angle - 2*np.pi

# # 绘图
# fig=plt.figure()
# ax = fig.add_subplot(111, polar=True)
# # 绘制折线图
# ax.plot(angles, list1[::-1], 's-', linewidth=2, label=dists[0], color=colors[0])
# # ax.fill(angles, list1, alpha=0.25, color=colors[0])
# # 绘制第二条折线图
# ax.plot(angles, list2[::-1], '>-', linewidth=2, label=dists[1], color=colors[1])
# # ax.fill(angles, list2, alpha=0.25, color=colors[1])


# # view
# highlight_neg = [3,10,11,12,31,32,33,]
# highlight_pos = [4,5,29,30,34,35,37]

# # emoition
# highlight_neg = [14,22,23,28,30,34]
# highlight_pos = [24,26,27]

# plt.axes(polar=True)
# for spot in highlight_neg:
#     plt.bar(angles[len(angles)-1-spot], 1, width=2*np.pi/N, color='mistyrose', align='center')
#     plt.bar(angles[len(angles)-1-spot], -1, width=2*np.pi/N, color='mistyrose', align='center')
# for spot in highlight_pos:
#     plt.bar(angles[len(angles)-1-spot], 1, width=2*np.pi/N, color='limegreen', align='center')
#     plt.bar(angles[len(angles)-1-spot], -1, width=2*np.pi/N, color='limegreen', align='center')

# for i in range(N):
#     if i % 6 == 0:
#         plt.vlines(angles[i]+np.pi/N, 0.8, 1.2, colors = "black", lw=2)

# # 添加每个特征的标签
# ax.set_thetagrids(angles * 180/np.pi, x_label, fontsize=FONT_SIZE, fontstyle='italic')
# # 设置雷达图的范围
# ax.set_ylim(-1,1)
# ax.set_yticks(np.arange(-1, 1, 0.5))
# ax.set_yticklabels([-1,-0.5,0,0.5], fontsize=FONT_SIZE)

# # 添加网格线
# ax.grid(True, color='lightgrey')
# # 设置图例
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=FONT_SIZE)
# # 显示图形
# fig.set_size_inches(9, 5)
# plt.tight_layout()
# plt.show()









# # ==============bar for feature selection from graph================
# def plot_bar_features(plot_type, highlight, x_label):
#     FONT_SIZE = 16
#     dists = ['S6E4', 'S6E21']
#     colors = ['lightcyan', 'lightblue']#['sandybrown', 'peachpuff']#['palegoldenrod', 'peachpuff']
#     edge_color = 'black'
#     style = ['', '']
#     # x_label = data.columns[4:]
#     # x_label = ['']*len(x_label)

#     bar_width = 0.15

#     list1 = plot_data[115][12:]
#     list2 = plot_data[132][12:]
#     norm = np.max(plot_data[:,12:], axis=0) - np.min(plot_data[:,12:], axis=0)
#     list1 = (list1 - np.min(plot_data[:,12:], axis=0)) / norm
#     list2 = (list2 - np.min(plot_data[:,12:], axis=0)) / norm
#     for i in range(len(norm)):
#         if norm[i] == 0:
#             list1[i] = 0.5
#             list2[i] = 0.5
#     if plot_type == 'positive':
#         list1 = (list1 + 0.1) / 1.1
#         list2 = (list2 + 0.1) / 1.1


#     list1 = list1[highlight]
#     list2 = list2[highlight]
#     # x_label = x_label[highlight]


#     x = np.arange(len(list1))
#     fig, ax = plt.subplots()
#     plt.bar(x-bar_width, list1, label=dists[0], color=colors[0], width=bar_width*2, hatch=style[0], ec=edge_color)
#     plt.bar(x+bar_width, list2, label=dists[1], color=colors[1], width=bar_width*2, hatch=style[1], ec=edge_color)

#     plt.xlim([-0.5, len(list1)-0.5])
#     plt.ylim([0,1])

#     plt.ylabel('Normalized Value', fontsize=FONT_SIZE)
#     ticks_y = [0, 0.2, 0.4, 0.6, 0.8, 1]
#     ax.set_yticks(ticks_y)
#     ax.set_yticklabels(ticks_y, fontsize=FONT_SIZE)


#     ax.set_xticks(x)
#     ax.set_xticklabels(x_label, fontsize=FONT_SIZE, rotation=18)

#     # categorys = [r'$emotion\_wave_{1,t}$', r'$emotion\_wave_{2,t}$', r'$emotion\_wave_{3,t}$']
#     # for i in range(len(categorys)):
#     #     plt.text((ticks[i] + ticks[i+1])/2-0.5, -0.07, categorys[i], ha='center', va='center', fontsize=FONT_SIZE)

#     plt.legend(loc='upper right', fontsize=FONT_SIZE, ncol=1)
#     fig.set_size_inches(7, 4.5)
#     plt.tight_layout()
#     plt.savefig('figures/bar_emotion_' + plot_type + '.pdf')
#     plt.show()

# plot_bar_features('positive', [0,12], [r'$pos\_momentum^1_t$',r'$pos\_momentum^3_t$'])
# plot_bar_features('negative', [3,6,9,10,13,17], [r'$neg\_peak\_esca^1_t$', r'$pos\_momentum^2_t$', r'$neg\_peak\_esca^2_t$', r'$peak\_order^2_t$', r'$neg\_momentum^3_t$', r'$peak\_gap^3_t$'])










# # # ==============count if happy ending================
# from generate_tables import emotion_wave
# import os
# FONT_SIZE = 16  # 12, 16
# LINE_WIDTH = 2    # 1.5, 2


# def happy_ending(season=1, episode=3, ending_position=1):
#     emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, _, _ = emotion_wave(season, episode)
#     end_time = 124
#     for i in range(len(emotion_all_percent_abs)):
#         if emotion_all_percent_abs[123-i] != 0:
#             end_time = 124-i
#             break
#     emotion_negative = emotion_negative[:end_time]
#     emotion_positive = emotion_positive[:end_time]

#     if season <=3:
#         offset = 5
#     else:
#         offset = 10
#     offset = ending_position
#     if np.mean(emotion_negative[-offset]) <= np.mean(emotion_positive[-offset]):
#         return 1, np.sum(emotion_positive), np.sum(emotion_negative)
#     else:
#         return 0, np.sum(emotion_positive), np.sum(emotion_negative)


# for position in range(1,2):
#     num_episode, count_pos = 0.0, 0.0
#     num_pos, num_neg = 0.0, 0.0
#     for season in range(1, 7):
#         for episode in range(1, 30):
#             if os.path.exists('season' + str(season) + '/Bullet_content_time_uid_emotion/Bullet_S' + str(season) + 'E' + str(episode) + '.json'):
#                 res, cur_pos, cur_neg = happy_ending(season, episode, position)
#                 num_episode += 1
#                 count_pos += res
#                 num_pos += cur_pos
#                 num_neg += cur_neg
#     print(position, count_pos/num_episode, num_pos, num_neg, num_pos/(num_pos+num_neg))