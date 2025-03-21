import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.font_manager as font_manager
from matplotlib import rcParams

del font_manager.weight_dict['roman']
font_manager._rebuild()
rcParams['font.family'] = 'Times New Roman'


# VAR = ['View', 'Comm', 'Emo']
# YLABEL = ['View Prediction Error', 'Comment Prediction Error', 'Comment Emotion Prediction Error']
# Y_LIM= [[3.5,5],[1,5],[-0.8,0.6]]

# file_name = "comparison.csv"
# data = pd.read_csv(file_name)
# labels = list(data['Method'].values)

# for i in range(len(VAR)):
#     v = VAR[i]
#     mean = data[v].values
#     std = data[v+'_Std'].values

#     x = np.arange(len(labels))

#     fig, ax = plt.subplots()

#     width = 0.4

#     for l in labels:
#         plt.bar(x, mean, width, color='coral', yerr=std)
#     # plt.hlines(mean[0], -1, len(labels), lw=1, color='r')

#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=30)
#     ax.set_ylabel(YLABEL[i])
#     # plt.ylim(Y_LIM[i])
#     # plt.legend()
#     # fig.set_size_inches(6.5, 3.5)
#     plt.tight_layout()

#     plt.show()


data = pd.read_csv('correlation_neg.csv', index_col='Variables')
simi = data.values[:, :]

sns.set()
sns.heatmap(simi, cmap='YlGnBu', )

for index in [4,12,18,24]:
    plt.vlines(index, 0, len(simi), lw=2, color='r')
    plt.hlines(index, 0, len(simi), lw=2, color='r')
plt.axis('off')
cax = plt.gcf().axes[-1] 
cax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('figures/correlation.png')
plt.show()