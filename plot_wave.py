import matplotlib.pylab as plt
import pandas as pd
import numpy as np
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





# # =============== plot stat =============

# file_name = "features.csv"
# data = pd.read_csv(file_name)

# num_view = np.power(10, data['viewcount'].values) / 1000
# num_bullet = np.power(10, data['textbullet'].values) / 1000
# num_comment = np.power(10, data['textcomment'].values) / 1000



# x = np.arange(len(num_view))

# fig, ax = plt.subplots()

# l1=plt.plot(x,num_view,'forestgreen',label='view', lw=2)
# l2=plt.plot(x,num_bullet,'darkorange',label='live comment', lw=2)
# # l3=plt.plot(x,num_comment,'limegreen',label='comment', lw=2)

# # ax.set_xlabel('Episode', fontsize = 12)
# ax.set_ylabel('Number by Thousand', fontsize=FONT_SIZE)
# ax.set_xticks([0, 24, 46, 64, 88, 112, 138])
# ax.set_xticklabels(['S1E1', 'S2E1', 'S3E1', 'S4E1', 'S5E1', 'S6E1', 'S6E27'])
# plt.xlim([0, len(num_view)-1])
# plt.tick_params(labelsize=FONT_SIZE)
# plt.legend(loc='upper left', fontsize=FONT_SIZE)

# fig.set_size_inches(14.5, 6)
# plt.tight_layout()
# plt.savefig('figures/raw_data.pdf')
# plt.show()





# =============== plot s6e1 partial=============

file_name = "emotion_wave.csv"
data = pd.read_csv(file_name)

pos = data['pos']#[10:-10]
neg = data['neg']#[10:-10]



x = np.arange(len(pos))

fig, ax = plt.subplots()


l1=plt.plot(x,pos,'tab:blue',label='positive emotion wave', lw=1.5)
l2=plt.plot(x,neg,'salmon',label='negative emotion wave', lw=1.5)


ax.set_ylabel('Emotion Valence', fontsize=FONT_SIZE)
ax.set_xlabel('Moment (minute)', fontsize=FONT_SIZE)


segs = [10, 46, 79, 111]
for i in range(len(segs)):
    plt.vlines(segs[i], 0, 280, color='tab:gray', ls='--')
    if i > 0:
        plt.text((segs[i-1] + segs[i])/2, 20, 'segment ' + str(i), ha='center', va='center', fontsize=FONT_SIZE)


dark_area = [45, 74, 101]
for i in dark_area:
    plt.axvspan(i-0.5, i+0.5, color='lightgray', lw=0, zorder=1)
    plt.text(i, -15, r'$\Delta$', ha='center', va='center', fontsize=FONT_SIZE, color='gray')


plt.xlim([10, 120-10])
plt.ylim([0, 280])
plt.xticks([10,30,50,70,90,110])
plt.tick_params(labelsize=FONT_SIZE)
plt.legend(loc='upper left', fontsize=FONT_SIZE)

fig.set_size_inches(14.2, 3.5)
plt.tight_layout()
plt.savefig('figures/emotion_wave.pdf', transparent=True)
plt.show()