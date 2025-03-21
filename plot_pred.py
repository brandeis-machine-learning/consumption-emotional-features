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
FONT_SIZE = 32
parameters = {'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
          'legend.fontsize': FONT_SIZE,
          'xtick.labelsize': FONT_SIZE,'ytick.labelsize':FONT_SIZE}
plt.rcParams.update(parameters)


file_names = ['view', 'text', 'emotion']
titles = ['View', 'Text Comment', 'Emotion Comment']

for i in range(len(file_names)):
    df = pd.read_csv('result_rmse_' + file_names[i] + '.csv')

    fig, ax = plt.subplots()
    sns.lineplot(x='Episode',y='only1',data=df,label='CAR',color='salmon', ci=None, lw=2)
    sns.lineplot(x='Episode',y='dummy1',data=df,label='CAR+DUM',color='tab:green', ci=None, lw=2)
    sns.lineplot(x='Episode',y='sig1',data=df,label='SIG',color='mediumblue', ci=None, lw=2)
    sns.lineplot(x='Episode',y='all1',data=df,label='ALL',color='skyblue', ci=None, lw=2)

    # ax.set_xticklabels(df['Episode'].values)
    plt.xlim([1,27])
    # plt.ylim([0,0.1])
    plt.xlabel('Episode in Season 6', fontsize=FONT_SIZE)
    plt.ylabel('Squared Error', fontsize=FONT_SIZE)
    # plt.title(titles[i])
    plt.tick_params(labelsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    fig.set_size_inches(7, 4)
    plt.tight_layout()
    plt.show()