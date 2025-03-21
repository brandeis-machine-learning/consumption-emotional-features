import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
del font_manager.weight_dict['roman']
font_manager._rebuild()
rcParams['font.family'] = 'Times New Roman'
rcParams["mathtext.fontset"] = 'stix'
FONT_SIZE = 16
parameters = {'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
          'legend.fontsize': FONT_SIZE,
          'xtick.labelsize': FONT_SIZE,'ytick.labelsize':FONT_SIZE}
plt.rcParams.update(parameters)


df_feature = pd.read_csv("feature_sept5_.csv", index_col=None)
COLORS = ['medium blue', 'salmon', 'orchid', 'kelly green']

labels = ['SIG', 'CAR', 'PRE_EN', 'ALL', 'MI_5', 'MI_10', 'MI_15', 'MI_20', 'LapScore_5', 'LapScore_10', 'LapScore_15', 'LapScore_20']
# labels = ['SIG', 'CAR', 'PRE_EN', 'ALL', 'MI_20', 'LapScore_20']
features = [['dummy','baidu_index','survey','is_first','pre_viewcount','pre_textcomment','pre_emojicomment','pre_all_negative_duration','pre_pos_k',
            '1_all_positive_duration','1_neg_k','2_all_positive_duration','2_neg_k','2_pos_neg','3_all_positive_duration','3_all_negative_duration','3_minus'],
            ['pre_viewcount'],
            ['pre_viewcount','dummy','baidu_index','survey','is_first'],]
features.append(df_feature.columns[4:])
pred_var = 'viewcount'

# labels = ['SIG', 'CAR', 'CAR+DUM', 'ALL']
# features = [['survey','pre_emotioncomment','1_pos_k','2_minus','3_all_positive_duration','3_neg_k'],
#             ['pre_emotioncomment'],
#             ['pre_emotioncomment','dummy']]
# features.append(df_feature.columns[4:])
# pred_var = 'emotioncomment'


# =========== generate plot data ================
# episode, squared_error, feature = [], [], []
# for i in range(len(features)):
#     error = np.loadtxt('prediction/result_' + pred_var + '_' + labels[i] + '.csv', delimiter=',')
#     error = error ** 2
#     print(i, error.shape)
#     error = np.sort(error, axis=0)[:]
#     error = np.mean(error, axis=0)
#     episode += list(np.arange(len(error))+1)
#     squared_error += list(error)
#     feature += [labels[i]] * len(error)
#     print(labels[i], np.mean(error))

# df_plot = pd.DataFrame()
# df_plot['Episode'] = episode
# df_plot['Squared Error'] = squared_error
# df_plot['Feature'] = feature
# df_plot.to_csv('prediction/allresults_' + pred_var + '_.csv', index=None)



df_plot = pd.read_csv('prediction/allresults_' + pred_var + '.csv', index_col=None)
# fig, ax = plt.subplots()
# palette = sns.xkcd_palette(COLORS)
# sns.lineplot(data=df_plot, x='Episode', y='Squared Error', hue='Feature', size='Feature',sizes=[4,1,1,1,1,1])#palette=palette,
# plt.xlim([1, len(df_plot)/len(labels)])
# plt.ylim([-0.005, 0.105])
# fig.set_size_inches(10,4)
# plt.tight_layout()
# plt.xlabel('Episode in Season 6', fontsize=FONT_SIZE)
# plt.ylabel('Squared Error', fontsize=FONT_SIZE)
# plt.tick_params(labelsize=FONT_SIZE)
# plt.legend(title=None, fontsize=FONT_SIZE)
# plt.savefig('figures/allresults_' + pred_var + '.pdf')
# plt.show()


data = {}
for label in labels:
    cur_feature = df_plot[df_plot['Feature']==label]['Squared Error'].values
    print(label, np.mean(cur_feature), np.std(cur_feature))
    data[label] = cur_feature


for label in labels:
    print(label, stats.ttest_rel(data['SIG'], data[label], alternative='less'))

for label in labels:
    print(label, np.corrcoef(data['SIG'], data[label])[0,1])