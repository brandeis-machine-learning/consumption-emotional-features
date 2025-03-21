# from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
# from matplotlib import pyplot as plt

# X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
# # X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
# Z = linkage(X, 'ward')  # index1, index2, cost, number_in_cluster
# print(Z)
# # f = fcluster(Z,4,'distance')
# fig = plt.figure()
# dn = dendrogram(Z)
# plt.xlabel('Time')
# plt.ylabel('Duration of Each Part')
# plt.show()




# from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd

# data = pd.read_csv('test.csv')

# y = data.values
# print(y.shape)

# k = 56

# fig, ax = plt.subplots()

# x = np.arange(y.shape[1])
# ax.plot(x, y[0+k], label='S6E1')
# ax.plot(x, y[13+k], label='S6E14')
# ax.plot(x, y[26+k], label='S6E27')
# ax.plot(x, y[27+k], label='AVG')

# # ax.set_title(DATASETS[i], fontsize=12)
# ax.set_xlabel('Bullet Percent')
# ax.set_ylabel('RMSE')
# ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax.set_xticklabels(['5%', '10%', '15%', '20%', '25%', '30%', '100%'])
# plt.xlim([0,6])
# plt.title('Emotion Comment')
# fig.set_size_inches(6.5, 3.5)
# plt.legend()
# plt.tight_layout()
# plt.show()




# view: 8,10
# text: 5,4
# emotion: 3,3

import numpy as np
import pandas as pd
from scipy import stats

name = ['view', 'text', 'emotion']
K_all = [8, 5, 3]
K_sig = [10, 4, 3]

def get_std(feat):
    mean = np.mean(feat**2)
    std = np.sum((feat - np.mean(feat)) ** 2) / len(feat)
    return mean, std

def feature_to_row(name, feat, K):
    file_name = 'results_' + name + '_' + feat +  '.csv'
    data = np.loadtxt(file_name,dtype="float",delimiter=',')
    res = []
    for i in range(len(data)):
        a = data[i]
        res.append(a[np.argpartition(a,-K)[:-K]])
    res = np.array(res)
    res = np.swapaxes(res, 0, 1)
    if len(res)<20:
        res = np.tile(res, (2,1))
    res = res.reshape(-1,1)
    np.savetxt(file_name[:-4] + '_' + file_name[-4:],res,delimiter=',')
    return res

for i in range(len(name)):
    print("==================", name[i], "====================")
    sig_feature = feature_to_row(name[i], 'sig', K_sig[i])
    all_feature = feature_to_row(name[i], 'all', K_all[i])

    df = pd.read_csv('result_rmse_' + name[i] + '.csv')
    only_feature = df['only'].values[:27]
    dum_feature = df['dummy'].values[:27]

    repeat_time = int(max(len(sig_feature), len(all_feature))/27)
    episodes = np.arange(27) + 1
    episodes = np.tile(episodes, repeat_time)

    episodes = episodes.reshape(-1)
    save_only = only_feature.reshape(-1)
    save_dum = dum_feature.reshape(-1)
    save_sig = sig_feature.reshape(-1)
    save_all = all_feature.reshape(-1)

    columns = ['Episode', 'only', 'dummy', 'sig', 'all', 'only1', 'dummy1', 'sig1', 'all1']
    features = [episodes, save_only, save_dum, save_sig, save_all, save_only**2, save_dum**2, save_sig**2, save_all**2]
    df_new = pd.DataFrame({'Episode':episodes})
    for index in range(1, len(columns)):
        df_new = pd.concat([df_new, pd.DataFrame({columns[index]:features[index]})], ignore_index=True, axis=1)
    df_new.columns = columns

    # df_new.append(pd.DataFrame(episodes, columns=['Episode']))
    # df_new.append(pd.DataFrame([save_only], columns=['only']))
    # df_new.append(pd.DataFrame([save_dum], columns=['dummy']))
    # df_new.append(pd.DataFrame([save_sig], columns=['sig']))
    # df_new.append(pd.DataFrame([save_all], columns=['all']))
    # df_new.append(pd.DataFrame([save_only**2], columns=['only1']))
    # df_new.append(pd.DataFrame([save_dum**2], columns=['dummy1']))
    # df_new.append(pd.DataFrame([save_sig**2], columns=['sig1']))
    # df_new.append(pd.DataFrame([save_all**2], columns=['all1']))

    sig_feature = np.mean(sig_feature.reshape(-1, 27), axis=0)
    all_feature = np.mean(all_feature.reshape(-1, 27), axis=0)
    print("Only:", np.mean(only_feature**2), np.std(only_feature))
    print("Dummy:", np.mean(dum_feature**2), np.std(dum_feature))
    print("Sig:", np.mean(sig_feature**2), np.std(sig_feature))
    print("All:", np.mean(all_feature**2), np.std(all_feature))

    sig_feature = sig_feature**2
    all_feature = all_feature**2
    only_feature = only_feature**2
    dum_feature = dum_feature**2

    print(stats.ttest_rel(sig_feature,only_feature, alternative='less'))
    print(stats.ttest_rel(sig_feature,dum_feature, alternative='less'))
    print(stats.ttest_rel(sig_feature,sig_feature, alternative='less'))
    print(stats.ttest_rel(sig_feature,all_feature, alternative='less'))

    df_new.to_csv('result_rmse_' + name[i] + '.csv')
# def paired_t_test(feat1, feat2):
#     return np.mean(feat1-feat2)/np.std(feat1-feat2)/np.sqrt(27)
# print(np.mean(sig_feature-only_feature)/np.std(sig_feature-only_feature)/np.sqrt(27))