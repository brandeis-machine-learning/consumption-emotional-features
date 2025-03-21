from scipy import stats
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.font_manager as font_manager
from matplotlib import rcParams

del font_manager.weight_dict['roman']
font_manager._rebuild()
rcParams['font.family'] = 'Times New Roman'
rcParams["mathtext.fontset"] = 'stix'
FONT_SIZE = 24
parameters = {'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
          'legend.fontsize': FONT_SIZE,
          'xtick.labelsize': FONT_SIZE,'ytick.labelsize':FONT_SIZE}
plt.rcParams.update(parameters)

thresholds = [10, 15, 20, 25, 30]

df_plot = pd.DataFrame()
df_dur, df_pos_neg, df_threshold = [], [], []
pos_ttest, neg_ttest = [], []
mean_pos, mean_neg, std_pos, std_neg = [], [], [], []
for threshold in thresholds:
    cur_data = pd.read_csv("feature_1.0_5_" + str(threshold) + ".csv", index_col=None)
    cur_dur_pos = cur_data['pre_all_positive_duration'].values
    cur_dur_neg = cur_data['pre_all_negative_duration'].values

    df_dur += list(cur_dur_pos)
    df_dur += list(cur_dur_neg)
    df_pos_neg += ['Positive'] * len(cur_dur_pos)
    df_pos_neg += ['Negative'] * len(cur_dur_neg)
    df_threshold += ['Threshold ' + str(threshold)] * (len(cur_dur_pos) + len(cur_dur_neg))

    mean_pos.append(np.mean(cur_dur_pos))
    mean_neg.append(np.mean(cur_dur_neg))
    std_pos.append(np.std(cur_dur_pos))
    std_neg.append(np.std(cur_dur_neg))

print(mean_pos)
print(std_pos)
print(mean_neg)
print(std_neg)

for i in range(len(mean_pos)):
    res = stats.ttest_ind_from_stats(mean1=mean_pos[i], std1=std_pos[i], nobs1=len(cur_dur_pos),
                                    mean2=mean_pos[1], std2=std_pos[1], nobs2=len(cur_dur_pos))
    print(thresholds[i], "pos=====", res)

    res = stats.ttest_ind_from_stats(mean1=mean_neg[i], std1=std_neg[i], nobs1=len(cur_dur_pos),
                                    mean2=mean_neg[1], std2=std_neg[1], nobs2=len(cur_dur_pos))
    print(thresholds[i], "neg=====", res)

# df_plot['Duration'] = df_dur
# df_plot['Emotion'] = df_pos_neg
# df_plot['Threshold'] = df_threshold

# fig, ax = plt.subplots()
# sns.violinplot(data=df_plot, x="Emotion", y="Duration", hue="Threshold")


# pos_ttest, neg_ttest = [], []
# pos_anc = df_dur[2*len(cur_dur_pos):(2+1)*len(cur_dur_pos)]
# neg_anc = df_dur[(2+1)*len(cur_dur_pos):(2+2)*len(cur_dur_pos)]
# for thres_idx in range(len(thresholds)):
#     pos_data = df_dur[thres_idx*2*len(cur_dur_pos):(thres_idx*2+1)*len(cur_dur_pos)]
#     neg_data = df_dur[(thres_idx*2+1)*len(cur_dur_pos):(thres_idx*2+2)*len(cur_dur_pos)]

#     _, pvalue = stats.ttest_ind(pos_anc, pos_data, alternative='two-sided')
#     pos_ttest.append(pvalue)

#     _, pvalue = stats.ttest_ind(neg_anc, neg_data, alternative='two-sided')
#     neg_ttest.append(pvalue)

# print(pos_ttest)
# print(neg_ttest)

# for i in range(len(pos_ttest)):
#     plt.text((i-2)/6, 30, '%.2f' % pos_ttest[i], fontsize=FONT_SIZE, verticalalignment='center', horizontalalignment='center')
#     plt.text((i-2)/6 + 1, 30, '%.2f' % neg_ttest[i], fontsize=FONT_SIZE, verticalalignment='center', horizontalalignment='center')

# # plt.xlabel('Episode in Season 6', fontsize=FONT_SIZE)
# # plt.ylabel('Squared Error', fontsize=FONT_SIZE)
# # plt.title(titles[i])
# plt.tick_params(labelsize=FONT_SIZE)
# plt.legend(fontsize=FONT_SIZE, loc='upper right', bbox_to_anchor=(1.4,0.9))
# fig.set_size_inches(15, 6)
# plt.tight_layout()
# plt.show()