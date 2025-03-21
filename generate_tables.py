from turtle import pos
from sklearn.linear_model import LinearRegression
from scipy import stats
import json
import time
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pylab as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import optimizers
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
# del font_manager.weight_dict['roman']
# font_manager._rebuild()
# rcParams['font.family'] = 'Times New Roman'
# rcParams["mathtext.fontset"] = 'stix'
# FONT_SIZE = 16
# parameters = {'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
#           'legend.fontsize': FONT_SIZE,
#           'xtick.labelsize': FONT_SIZE,'ytick.labelsize':FONT_SIZE}
# plt.rcParams.update(parameters)

from random import sample
import warnings
warnings.filterwarnings("ignore")

NUM_EPISODE = [25, 23, 19, 25, 25, 28, 23]
NUM_SEPT = 5
OFFSET_TIME = [5, 5, 5, 10, 10, 10, 10]
CUT_HEAD = True
CUT_END = True
USE_AVERAGE = False     # scale features to each minute
USE_BALANCE = True      # apply weights when clustering
USE_DOUBLE = True       # copy each minute for clustering
USE_LOG = True          # calculte log for clustering
AVG_SEPT = False        # average segments
SELECT_PERCENT = 1.0
STARTING_TEST = 112#60 #63 #112
PERCENTS = [0.05,0.1,0.15,0.2,0.25,0.3]

PREDICT_VAR = ['viewcount','textcomment','emotioncomment']
STAT_VAR = ['dummy','pre_viewcount','pre_textbullet','pre_textcomment','pre_emojibullet','pre_emojicomment','pre_emotioncomment']
PRE_WAVE_VAR = ['pre_all_positive_duration','pre_all_negative_duration','pre_pos_k','pre_pos_b','pre_neg_k','pre_neg_b']
PEAK_END_VAR = ['pre_emojicomment','pre_emotioncomment']
SEPT_VAR = ['0_all_positive_peak','0_all_positive_duration','0_all_negative_peak','0_all_negative_duration',
                    '0_pos_k','0_pos_b','0_neg_k','0_neg_b',
                    '1_all_positive_peak','1_all_positive_duration','1_all_negative_peak','1_all_negative_duration',
                    '1_pos_k','1_pos_b','1_neg_k','1_neg_b',
                    '2_all_positive_peak','2_all_positive_duration','2_all_negative_peak','2_all_negative_duration',
                    '2_pos_k','2_pos_b','2_neg_k','2_neg_b',
                    '3_all_positive_peak','3_all_positive_duration','3_all_negative_peak','3_all_negative_duration',
                    '3_pos_k','3_pos_b','3_neg_k','3_neg_b',
                    '4_all_positive_peak','4_all_positive_duration','4_all_negative_peak','4_all_negative_duration',
                    '4_pos_k','4_pos_b','4_neg_k','4_neg_b']
SEPT_3_VAR = ['pre_all_positive_duration', 'pre_all_negative_duration', 'pre_pos_k','pre_pos_b','pre_neg_k','pre_neg_b',
            '0_all_positive_duration','0_all_negative_duration','0_pos_k','0_pos_b','0_neg_k','0_neg_b',
            '1_all_positive_duration','1_all_negative_duration','1_pos_k','1_pos_b','1_neg_k','1_neg_b',
            '2_all_positive_duration','2_all_negative_duration','2_pos_k','2_pos_b','2_neg_k','2_neg_b']
ALL_VAR = STAT_VAR + PRE_WAVE_VAR + SEPT_VAR
NO_PEAK_VAR = ['pre_all_positive_duration', 'pre_all_negative_duration', 'pre_pos_k','pre_pos_b','pre_neg_k','pre_neg_b',
            '0_all_positive_duration','0_all_negative_duration','0_pos_k','0_pos_b','0_neg_k','0_neg_b',
            '1_all_positive_duration','1_all_negative_duration','1_pos_k','1_pos_b','1_neg_k','1_neg_b',
            '2_all_positive_duration','2_all_negative_duration','2_pos_k','2_pos_b','2_neg_k','2_neg_b',
            '3_all_positive_duration','3_all_negative_duration','3_pos_k','3_pos_b','3_neg_k','3_neg_b',
            '4_all_positive_duration','4_all_negative_duration','4_pos_k','4_pos_b','4_neg_k','4_neg_b']
NO_DUR_VAR = ['pre_all_positive_peak', 'pre_all_negative_peak', 'pre_pos_k','pre_pos_b','pre_neg_k','pre_neg_b',
            '0_all_positive_peak','0_all_negative_peak','0_pos_k','0_pos_b','0_neg_k','0_neg_b',
            '1_all_positive_peak','1_all_negative_peak','1_pos_k','1_pos_b','1_neg_k','1_neg_b',
            '2_all_positive_peak','2_all_negative_peak','2_pos_k','2_pos_b','2_neg_k','2_neg_b',
            '3_all_positive_peak','3_all_negative_peak','3_pos_k','3_pos_b','3_neg_k','3_neg_b',
            '4_all_positive_peak','4_all_negative_peak','4_pos_k','4_pos_b','4_neg_k','4_neg_b']
POS_VAR = ['pre_all_positive_duration','pre_pos_k','pre_pos_b',
            '0_all_positive_duration','0_pos_k','0_pos_b',
            '1_all_positive_duration','1_pos_k','1_pos_b',
            '2_all_positive_duration','2_pos_k','2_pos_b',
            '3_all_positive_duration','3_pos_k','3_pos_b',
            '4_all_positive_duration','4_pos_k','4_pos_b']
NEG_VAR = ['pre_all_negative_duration','pre_neg_k','pre_neg_b',
            '0_all_negative_duration','0_neg_k','0_neg_b',
            '1_all_negative_duration','1_neg_k','1_neg_b',
            '2_all_negative_duration','2_neg_k','2_neg_b',
            '3_all_negative_duration','3_neg_k','3_neg_b',
            '4_all_negative_duration','4_neg_k','4_neg_b']
SIG_VAR = ['dummy','pre_emojibullet','pre_emojicomment','pre_emotioncomment','pre_neg_k','pre_neg_b','0_all_positive_duration','2_all_negative_duration','3_all_positive_duration','3_all_negative_duration','3_pos_k','3_pos_b','3_neg_k','3_neg_b','4_all_negative_peak']

def isEmoji(content):
    if not content:
        return False
    if u"\U0001F600" <= content and content <= u"\U0001F64F":
        return True
    elif u"\U0001F300" <= content and content <= u"\U0001F5FF":
        return True
    elif u"\U0001F680" <= content and content <= u"\U0001F6FF":
        return True
    elif u"\U0001F1E0" <= content and content <= u"\U0001F1FF":
        return True
    else:
        return False
    
def cal_emoji_num(bullet):
    total = 0
    emoji_high = 0
    emoji_all = 0
    for item in bullet:
        total += 1
        flag_high = False
        flag_all = False
        for word in item[0]:
            if word in ['😂','👍','👏','😭','😄','😊','😍','😁','💪','😔','😓','🌚','😘','😞','🙃','👀','🙄','😡','💔','👎']:
                flag_high = True
                flag_all = True
                continue
            if isEmoji(word):
                flag_all = True
        emoji_high += flag_high
        emoji_all += flag_all
    return emoji_high



def cal_view(path):
    with open(path, 'r', encoding='utf-8') as f:
        hot = json.load(f)
    return hot['data'][0]['idx']
def cal_emotion_comment(path):
    with open(path, 'r', encoding='utf-8') as f:
        comment = json.load(f)
    pos_num,neg_num,neu_num = 0,0,0
    for item in comment:
        sentiment = item[-1]['sentiment']
        if sentiment == 0:
            neg_num += 1
        elif sentiment == 1:
            neu_num += 1
        elif sentiment == 2:
            pos_num += 1
    return len(comment), cal_emoji_num(comment), (pos_num-neg_num)/len(comment)


def emotion_wave(season, episode=24):
    # 导入数据
    with open("season" + str(season) + "/Bullet_content_time_uid_emotion/Bullet_S" + str(season) + "E" + str(episode) + ".json",'r',encoding='utf-8') as f:
        data_emotion = json.load(f)
    #画第一个pos和neg图
    divide = 60 
    data_percent = []
    for i in range(int(7414/divide + 1)):
        data_percent.append({"neg_num": 0,'neu_num':0 ,'pos_num':0 ,'second': i })

    if SELECT_PERCENT < 1:
        data_emotion = sample(data_emotion, int(len(data_emotion)*SELECT_PERCENT))
    num_bullet = len(data_emotion)
    num_emoji_bullet = cal_emoji_num(data_emotion)

    for items in data_emotion:
        try: 
            m = int(int(items[1])/divide)
            item = items[3]
            if item['sentiment'] == 0:
                data_percent[m]['neg_num'] += 1
            if item['sentiment'] == 1:
                data_percent[m]['neu_num'] += 1
            if item['sentiment'] == 2:
                data_percent[m]['pos_num'] += 1
        except Exception as e:
            print("Exception:"+str(e))
    df = pd.DataFrame.from_dict(data_percent)
    #get negative positive data
    temp_neg = pd.DataFrame(df, columns= ['neg_num']).values.tolist()
    temp_pos = pd.DataFrame(df, columns= ['pos_num']).values.tolist()
    emotion_negative = []
    emotion_positive = []
    for item in temp_neg:
        emotion_negative.append(item[0])
    for item in temp_pos:
        emotion_positive.append(item[0])

    #画合起来的emotion wave
    emotion_all_percent = []
    for i in range(124):
        emotion_all_percent.append((
            df['pos_num'].values.tolist()[i] - df['neg_num'].values.tolist()[i]))

    # get the abs value
    emotion_all_percent_abs = []
    for i in range(124):
        emotion_all_percent_abs.append((
            (df['pos_num'].values.tolist()[i]) + df['neg_num'].values.tolist()[i] + df['neu_num'].values.tolist()[i]))
    
    return emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, num_bullet, num_emoji_bullet


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    
    lag = lag if lag < len(y) else len(y)
    
    for i in range(0,lag):
        avgFilter[i] = np.mean(y[0:lag])
        stdFilter[i] = np.std(y[0:lag])

    for i in range(1, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            if i >= lag:
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            if i >= lag:
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


def get_average(data):
    average = 0
    for item in data:
        average += abs(item)
    return int(average/len(data))


def get_peak(y,episode=24,lag=20,threshold=1,influence=0.5):
    lag = 15
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
    return result["signals"]


def get_feature(result):
    frequency = 0
    all_positive_peak = 0
    all_negative_peak = 0
    all_positive_duration = 0
    all_negative_duration = 0
    pre = result[0]
    pre_emotion = 0
    index, pos_peak_end, neg_peak_end = 0, 0, 0
    for item in result:
        #pos的peak和freq
        if item != pre and item == 1:
            all_positive_peak += 1
            pos_peak_end = index
            if pre_emotion == -1:
                frequency += 1
            pre_emotion = item
        #neg的peak和freq
        if item != pre and item == -1:
            neg_peak_end = index
            all_negative_peak += 1
            if pre_emotion == 1:
                frequency += 1
            pre_emotion = item
        #pos的duration
        if item == pre and item == 1:
            all_positive_duration += 1
        #neg的duration
        if item == pre and item == -1:
            all_negative_duration += 1
        pre = item
        index += 1
    # print("pos peak:%d dur:%d neg peak:%d dur:%d freq:%d" 
    #       % (all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,frequency))
    return all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,frequency, pos_peak_end, neg_peak_end


def get_peak_2(time_series,title="",episode=24):
    indices = find_peaks(time_series,prominence=10)[0]
    try:
        x = np.array(indices)
        y = np.array([time_series[j] for j in indices])
        z1 = np.polyfit(x, y, 1)
        peak_end = x[-1]
    except Exception:
        return 0,0,len(indices),np.max(time_series),np.argmax(time_series)
    return z1[0], z1[1], len(indices), np.max(time_series), np.argmax(time_series)


# calculate distances between each pair of consecutive parts for clustering
def calculate_error(total, pos, neg, bullet, seq_range):
    x = np.arange(seq_range[0], seq_range[1] + 1)
    A = np.ones((len(x), 2))
    A[:, 0] = x

    error = 0.0
    weight = 1
    if USE_BALANCE:
        weight = seq_range[1] - seq_range[0]

    for st in [total, pos, neg, bullet]:
        y = np.array(st[seq_range[0]:seq_range[1] + 1])
        # 返回回归系数、残差平方和、自变量X的秩、X的奇异值
        (p, residuals, ranks, s) = np.linalg.lstsq(A, y, rcond=None)
        try:
            error += residuals[0]  * weight
        except IndexError:
            error += 0.0
    return error


def double_list(l):
    if USE_DOUBLE:
        return [val for val in l for i in range(2)]
    else:
        return l


# clustering
def Bottom_Up(emotion_all, emotion_pos, emotion_neg, bullet_sum, seg_k, starting, ending):
    T = emotion_all.copy()
    P = emotion_neg.copy()
    N = emotion_pos.copy()
    B = bullet_sum.copy()

    if CUT_END:
        seg_k = seg_k - 1
        T = T[:ending]
        P = P[:ending]
        N = N[:ending]
        B = B[:ending]

    if CUT_HEAD:
        seg_k = seg_k - 1
        T = T[starting:]
        P = P[starting:]
        N = N[starting:]
        B = B[starting:]


    if USE_DOUBLE:
        T = double_list(T)
        P = double_list(P)
        N = double_list(N)
        B = double_list(B)
    else:
        if len(T) % 2!=0:
            T.append(T[-1])
            P.append(P[-1])
            N.append(N[-1])
            B.append(B[-1])

    Seg_TS = []
    seg = []
    merge_cost = []
    all_res = []
    Z_print, Z_index = [], []
    for i in range(0, len(T), 2):
        Seg_TS += [T[i:i + 2]]
        seg.append((i, i + 1))
    for i in range(0, len(Seg_TS) - 1):
        merge_cost.insert(i, calculate_error(T, P, N, B, (seg[i][0], seg[i + 1][1])))
        Z_index.append(i)
    Z_index.append(len(Seg_TS) - 1)
    all_res.append(Seg_TS.copy())
    new_index = len(Seg_TS)
    while len(merge_cost) >= 1:
        # print(merge_cost)
        index = merge_cost.index(min(merge_cost))
        Seg_TS[index] = Seg_TS[index] + Seg_TS[index + 1]
        seg[index] = (seg[index][0], seg[index + 1][1])

        # Z_print.append(np.array([Z_index[index], Z_index[index + 1], float(len(Seg_TS[index])/2 - 1), len(Seg_TS[index])]))
        Z_print.append(np.array([Z_index[index], Z_index[index + 1], np.log(1+merge_cost[index]), len(Seg_TS[index])]))
        Z_index[index] = new_index

        del Seg_TS[index + 1]
        del seg[index + 1]
        del Z_index[index + 1]

        if index > 1:
            merge_cost[index - 1] = calculate_error(T, P, N, B, (seg[index - 1][0], seg[index][1]))
        if index + 1 < len(merge_cost):
            merge_cost[index] = calculate_error(T, P, N, B, (seg[index][0], seg[index + 1][1]))
            del merge_cost[index + 1]
        else:
            del merge_cost[index]
        all_res.append(Seg_TS.copy())
        new_index += 1

        if len(merge_cost) == seg_k - 1:
            return_TS = Seg_TS.copy()

    if CUT_HEAD:
        return_TS = [double_list(emotion_all[:starting])] + return_TS
    if CUT_END:
        return_TS.append(double_list(emotion_all[ending:]))
    return return_TS, all_res, np.array(Z_print)






def generate_tables():
    if NUM_SEPT == 3:
        feature = {'Season':[],'Episode':[],'dummy':[],'viewcount':[],'textcomment':[],'emotioncomment':[],
                    'pre_viewcount':[],'pre_textbullet':[],'pre_textcomment':[],'pre_emojibullet':[],'pre_emojicomment':[],'pre_emotioncomment':[],
                    'pre_all_positive_peak':[],'pre_all_positive_duration':[],'pre_all_negative_peak':[],'pre_all_negative_duration':[],
                    'pre_pos_k':[],'pre_pos_b':[],'pre_neg_k':[],'pre_neg_b':[],
                    'pre_pos_peak':[],'pre_pos_max':[],'pre_neg_peak':[],'pre_neg_max':[],'pre_pos_neg':[],
                    'pre_pos_peak_end':[],'pre_neg_peak_end':[],
                    '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
                    '0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],'0_pos_peak':[],'0_pos_max':[],'0_neg_peak':[],'0_neg_max':[],'0_pos_neg':[],
                    '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
                    '1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],'1_pos_peak':[],'1_pos_max':[],'1_neg_peak':[],'1_neg_max':[],'1_pos_neg':[],
                    '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
                    '2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[],'2_pos_peak':[],'2_pos_max':[],'2_neg_peak':[],'2_neg_max':[],'2_pos_neg':[],}
    elif NUM_SEPT == 4:
        feature = {'Season':[],'Episode':[],'dummy':[],'viewcount':[],'textcomment':[],'emotioncomment':[],
                    'pre_viewcount':[],'pre_textbullet':[],'pre_textcomment':[],'pre_emojibullet':[],'pre_emojicomment':[],'pre_emotioncomment':[],
                    'pre_all_positive_peak':[],'pre_all_positive_duration':[],'pre_all_negative_peak':[],'pre_all_negative_duration':[],
                    'pre_pos_k':[],'pre_pos_b':[],'pre_neg_k':[],'pre_neg_b':[],
                    'pre_pos_peak':[],'pre_pos_max':[],'pre_neg_peak':[],'pre_neg_max':[],'pre_pos_neg':[],
                    'pre_pos_peak_end':[],'pre_neg_peak_end':[],
                    '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
                    '0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],'0_pos_peak':[],'0_pos_max':[],'0_neg_peak':[],'0_neg_max':[],'0_pos_neg':[],
                    '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
                    '1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],'1_pos_peak':[],'1_pos_max':[],'1_neg_peak':[],'1_neg_max':[],'1_pos_neg':[],
                    '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
                    '2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[],'2_pos_peak':[],'2_pos_max':[],'2_neg_peak':[],'2_neg_max':[],'2_pos_neg':[],
                    '3_all_positive_peak':[],'3_all_positive_duration':[],'3_all_negative_peak':[],'3_all_negative_duration':[],
                    '3_pos_k':[],'3_pos_b':[],'3_neg_k':[],'3_neg_b':[],'3_pos_peak':[],'3_pos_max':[],'3_neg_peak':[],'3_neg_max':[],'3_pos_neg':[],}
    elif NUM_SEPT == 5:
        feature = {'Season':[],'Episode':[],'dummy':[],'viewcount':[],'textcomment':[],'emotioncomment':[],
                    'pre_viewcount':[],'pre_textbullet':[],'pre_textcomment':[],'pre_emojibullet':[],'pre_emojicomment':[],'pre_emotioncomment':[],
                    'pre_all_positive_peak':[],'pre_all_positive_duration':[],'pre_all_negative_peak':[],'pre_all_negative_duration':[],
                    'pre_pos_k':[],'pre_pos_b':[],'pre_neg_k':[],'pre_neg_b':[],
                    'pre_pos_peak':[],'pre_pos_max':[],'pre_neg_peak':[],'pre_neg_max':[],'pre_pos_neg':[],'pre_pos_div_neg':[],
                    'pre_pos_peak_end':[],'pre_neg_peak_end':[],
                    '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
                    '0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],'0_pos_peak':[],'0_pos_max':[],'0_neg_peak':[],'0_neg_max':[],'0_pos_neg':[],'0_pos_div_neg':[],
                    '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
                    '1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],'1_pos_peak':[],'1_pos_max':[],'1_neg_peak':[],'1_neg_max':[],'1_pos_neg':[],'1_pos_div_neg':[],
                    '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
                    '2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[],'2_pos_peak':[],'2_pos_max':[],'2_neg_peak':[],'2_neg_max':[],'2_pos_neg':[],'2_pos_div_neg':[],
                    '3_all_positive_peak':[],'3_all_positive_duration':[],'3_all_negative_peak':[],'3_all_negative_duration':[],
                    '3_pos_k':[],'3_pos_b':[],'3_neg_k':[],'3_neg_b':[],'3_pos_peak':[],'3_pos_max':[],'3_neg_peak':[],'3_neg_max':[],'3_pos_neg':[],'3_pos_div_neg':[],
                    '4_all_positive_peak':[],'4_all_positive_duration':[],'4_all_negative_peak':[],'4_all_negative_duration':[],
                    '4_pos_k':[],'4_pos_b':[],'4_neg_k':[],'4_neg_b':[],'4_pos_peak':[],'4_pos_max':[],'4_neg_peak':[],'4_neg_max':[],'4_pos_neg':[],'4_pos_div_neg':[],}
    elif NUM_SEPT == 6:
        feature = {'Season':[],'Episode':[],'dummy':[],'viewcount':[],'textcomment':[],'emotioncomment':[],
                    'pre_viewcount':[],'pre_textbullet':[],'pre_textcomment':[],'pre_emojibullet':[],'pre_emojicomment':[],'pre_emotioncomment':[],
                    'pre_all_positive_peak':[],'pre_all_positive_duration':[],'pre_all_negative_peak':[],'pre_all_negative_duration':[],
                    'pre_pos_k':[],'pre_pos_b':[],'pre_neg_k':[],'pre_neg_b':[],
                    'pre_pos_peak':[],'pre_pos_max':[],'pre_neg_peak':[],'pre_neg_max':[],'pre_pos_neg':[],
                    'pre_pos_peak_end':[],'pre_neg_peak_end':[],
                    '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
                    '0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],'0_pos_peak':[],'0_pos_max':[],'0_neg_peak':[],'0_neg_max':[],'0_pos_neg':[],
                    '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
                    '1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],'1_pos_peak':[],'1_pos_max':[],'1_neg_peak':[],'1_neg_max':[],'1_pos_neg':[],
                    '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
                    '2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[],'2_pos_peak':[],'2_pos_max':[],'2_neg_peak':[],'2_neg_max':[],'2_pos_neg':[],
                    '3_all_positive_peak':[],'3_all_positive_duration':[],'3_all_negative_peak':[],'3_all_negative_duration':[],
                    '3_pos_k':[],'3_pos_b':[],'3_neg_k':[],'3_neg_b':[],'3_pos_peak':[],'3_pos_max':[],'3_neg_peak':[],'3_neg_max':[],'3_pos_neg':[],
                    '4_all_positive_peak':[],'4_all_positive_duration':[],'4_all_negative_peak':[],'4_all_negative_duration':[],
                    '4_pos_k':[],'4_pos_b':[],'4_neg_k':[],'4_neg_b':[],'4_pos_peak':[],'4_pos_max':[],'4_neg_peak':[],'4_neg_max':[],'4_pos_neg':[],
                    '5_all_positive_peak':[],'5_all_positive_duration':[],'5_all_negative_peak':[],'5_all_negative_duration':[],
                    '5_pos_k':[],'5_pos_b':[],'5_neg_k':[],'5_neg_b':[],'5_pos_peak':[],'5_pos_max':[],'5_neg_peak':[],'5_neg_max':[],'5_pos_neg':[]}


    for season in range(1, 7):
        for total_episode in range(1,NUM_EPISODE[season-1]):

            if season < 7:
                emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, num_bullet, num_emoji_bullet = emotion_wave(season, total_episode)
            else:
                cur_episode1 = total_episode*2-1
                cur_episode2 = total_episode*2
                emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, num_bullet, num_emoji_bullet = emotion_wave(season, cur_episode1)
                emotion_all_percent2, emotion_all_percent_abs2, emotion_negative2, emotion_positive2, num_bullet2, num_emoji_bullet2 = emotion_wave(season, cur_episode2)
                
                end_time1 = 124
                for i in range(len(emotion_all_percent_abs)):
                    if emotion_all_percent_abs[123-i] != 0:
                        end_time1 = 124-i
                        break
                end_time2 = 124
                for i in range(len(emotion_all_percent_abs2)):
                    if emotion_all_percent_abs[123-i] != 0:
                        end_time2 = 124-i
                        break
                
                emotion_all_percent = emotion_all_percent[:end_time1] + emotion_all_percent2[:end_time2]
                emotion_all_percent_abs = emotion_all_percent_abs[:end_time1] + emotion_all_percent_abs2[:end_time2]
                emotion_negative = emotion_negative[:end_time1] + emotion_negative2[:end_time2]
                emotion_positive = emotion_positive[:end_time1] + emotion_positive2[:end_time2]
                num_bullet += num_bullet2
                num_emoji_bullet += num_emoji_bullet2

            result = get_peak(emotion_all_percent)
            all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency, pos_peak_end, neg_peak_end = get_feature(result)
            pos_k, pos_b, pos_peak, pos_max, pos_position = get_peak_2(emotion_positive,'_pos_slope')
            neg_k, neg_b, neg_peak, neg_max, neg_position = get_peak_2(emotion_negative,'_neg_slope')
            if pos_position > neg_position:
                position = -1
            elif pos_position < neg_position:
                position = 1
            else:
                position = 0


            pos_div_neg = []
            for i in range(len(emotion_positive)):
                if emotion_negative[i] == 0:
                    pos_div_neg.append(0)
                else:
                    pos_div_neg.append(emotion_positive[i]/emotion_negative[i])


            if season < 7:
                path_hot = "season" + str(season) + "/Hot/Hot_S" + str(season) + "E" + str(total_episode) + ".json"
                path_bullet = "season" + str(season) + "/Bullet_content_time_uid_emotion/Bullet_S" + str(season) + "E" + str(total_episode) + ".json"
                path_comment = "season" + str(season) + "/Comment_content_uid_emotion/Comment_S" + str(season) + "E" + str(total_episode) + ".json"
                viewcount = cal_view(path_hot)
            else:
                path_bullet = "season" + str(season) + "/Bullet_content_time_uid_emotion/Bullet_S" + str(season) + "E" + str(cur_episode1) + ".json"
                path_comment = "season" + str(season) + "/Comment_content_uid_emotion/Comment_S" + str(season) + "E" + str(cur_episode1) + ".json"
                path_bullet2 = "season" + str(season) + "/Bullet_content_time_uid_emotion/Bullet_S" + str(season) + "E" + str(cur_episode2) + ".json"
                path_comment2 = "season" + str(season) + "/Comment_content_uid_emotion/Comment_S" + str(season) + "E" + str(cur_episode2) + ".json"
                viewcount = 0


            num_bullet, num_emoji_bullet, _ = cal_emotion_comment(path_bullet)
            text_comment, emoji_comment, emotion_comment = cal_emotion_comment(path_comment)
            if season == 7:
                num_bullet2, num_emoji_bullet2, _ = cal_emotion_comment(path_bullet2)
                text_comment2, emoji_comment2, emotion_comment2 = cal_emotion_comment(path_comment2)
                num_bullet += num_bullet2
                num_emoji_bullet += num_emoji_bullet2
                text_comment += text_comment2
                emoji_comment += emoji_comment2
                emotion_comment += emotion_comment2

            
            end_time = len(emotion_all_percent_abs)
            for i in range(len(emotion_all_percent_abs)):
                if emotion_all_percent_abs[len(emotion_all_percent_abs)-1-i] != 0:
                    end_time = len(emotion_all_percent_abs)-i
                    break
            starting = OFFSET_TIME[season - 1]
            ending = end_time - OFFSET_TIME[season - 1]
            div = 1

            if USE_AVERAGE:
                div = end_time
            if USE_LOG:
                if viewcount > 0:
                    viewcount = np.log10(viewcount)
                if num_bullet > 0:
                    num_bullet = np.log10(num_bullet)
                if num_emoji_bullet > 0:
                    num_emoji_bullet = np.log10(num_emoji_bullet)
                if text_comment > 0:
                    text_comment = np.log10(text_comment)
                if emoji_comment > 0:
                    emoji_comment = np.log10(emoji_comment)

            feature['Season'].append(season)
            feature['Episode'].append(total_episode)
            if season <= 3:
                feature['dummy'].append(0)
            else:
                feature['dummy'].append(1)


            feature['viewcount'].append(viewcount)
            feature['textcomment'].append(text_comment)
            feature['emotioncomment'].append(emotion_comment)
            if season==1 and total_episode==1:
                feature['pre_viewcount'].append(0)
                feature['pre_textbullet'].append(0)
                feature['pre_textcomment'].append(0)
                feature['pre_emojibullet'].append(0)
                feature['pre_emojicomment'].append(0)
                feature['pre_emotioncomment'].append(0)
                feature['pre_all_positive_peak'].append(0)
                feature['pre_all_positive_duration'].append(0)
                feature['pre_all_negative_peak'].append(0)
                feature['pre_all_negative_duration'].append(0)
                feature['pre_pos_k'].append(0)
                feature['pre_pos_b'].append(0)
                feature['pre_neg_k'].append(0)
                feature['pre_neg_b'].append(0)
                feature['pre_pos_peak'].append(0)
                feature['pre_pos_max'].append(0)
                feature['pre_neg_peak'].append(0)
                feature['pre_neg_max'].append(0)
                feature['pre_pos_peak_end'].append(0)
                feature['pre_neg_peak_end'].append(0)
                feature['pre_pos_neg'].append(0)
                feature['pre_pos_div_neg'].append(0)
            if not(season==6 and total_episode==23):
                feature['pre_viewcount'].append(viewcount)
                feature['pre_textbullet'].append(num_bullet)
                feature['pre_textcomment'].append(text_comment)
                feature['pre_emojibullet'].append(num_emoji_bullet)
                feature['pre_emojicomment'].append(emoji_comment)
                feature['pre_emotioncomment'].append(emotion_comment)
                feature['pre_all_positive_peak'].append(all_positive_peak / div)
                feature['pre_all_positive_duration'].append(all_positive_duration / div)
                feature['pre_all_negative_peak'].append(all_negative_peak / div)
                feature['pre_all_negative_duration'].append(all_negative_duration / div)
                feature['pre_pos_k'].append(pos_k)
                feature['pre_pos_b'].append(pos_b)
                feature['pre_neg_k'].append(neg_k)
                feature['pre_neg_b'].append(neg_b)
                feature['pre_pos_peak'].append(pos_peak)
                feature['pre_pos_max'].append(pos_max)
                feature['pre_neg_peak'].append(neg_peak)
                feature['pre_neg_max'].append(neg_max)
                feature['pre_pos_peak_end'].append(pos_peak_end)
                feature['pre_neg_peak_end'].append(neg_peak_end)
                feature['pre_pos_neg'].append(position)
                feature['pre_pos_div_neg'].append(np.mean(pos_div_neg))

            
            if AVG_SEPT:
                if NUM_SEPT == 3:
                    sept = [0, starting, ending, end_time]
                elif NUM_SEPT == 4:
                    sept = [0, starting, starting + int((ending - starting)/2), ending, end_time]
                elif NUM_SEPT == 5:
                    sept = [0, starting, starting + int((ending - starting)/3), starting + int((ending - starting)/3*2), ending, end_time]
                elif NUM_SEPT == 6:
                    sept = [0, starting, starting + int((ending - starting)/4), starting + int((ending - starting)/4*2), starting + int((ending - starting)/4*3), ending, end_time]
            else:
                sept, all_res, Z_print = Bottom_Up(emotion_all_percent[:end_time], emotion_positive[:end_time], emotion_negative[:end_time], emotion_all_percent_abs[:end_time], NUM_SEPT, starting, ending)
                # if season==1 and total_episode==1:
                #     del font_manager.weight_dict['roman']
                #     font_manager._rebuild()
                #     rcParams['font.family'] = 'Times New Roman'
                #     rcParams["mathtext.fontset"] = 'stix'
                #     FONT_SIZE = 16
                #     parameters = {'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
                #               'legend.fontsize': FONT_SIZE,
                #               'xtick.labelsize': FONT_SIZE,'ytick.labelsize':FONT_SIZE}
                #     plt.rcParams.update(parameters)

                #     fig, ax = plt.subplots()
                #     dn = dendrogram(Z_print, color_threshold=9.5)
                #     plt.xlabel('Moment', fontsize=FONT_SIZE)
                #     plt.ylabel('Total Residual (log)', fontsize=FONT_SIZE)
                #     plt.hlines(9.5, xmin=0, xmax=280, color='gray', ls='--')
                #     ax.set_xticks(np.arange(0, 27, 2)*10+5)
                #     ax.set_xticklabels(np.arange(0, 27, 2)+5)
                #     plt.xticks(rotation=0)
                #     plt.tick_params(labelsize=FONT_SIZE)
                #     fig.set_size_inches(7, 5)
                #     plt.tight_layout()
                #     plt.savefig('figures/tree_s' + str(season) + 'e' + str(total_episode) + '.pdf')
                #     plt.show()
                #     return
                # if season==6 and total_episode==27:
                #     del font_manager.weight_dict['roman']
                #     font_manager._rebuild()
                #     rcParams['font.family'] = 'Times New Roman'
                #     rcParams["mathtext.fontset"] = 'stix'
                #     FONT_SIZE = 16
                #     parameters = {'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
                #               'legend.fontsize': FONT_SIZE,
                #               'xtick.labelsize': FONT_SIZE,'ytick.labelsize':FONT_SIZE}
                #     plt.rcParams.update(parameters)

                #     fig, ax = plt.subplots()
                #     dn = dendrogram(Z_print, color_threshold=16.5)
                #     plt.xlabel('Moment', fontsize=FONT_SIZE)
                #     plt.ylabel('Total Residual (log)', fontsize=FONT_SIZE)
                #     plt.hlines(16.5, xmin=0, xmax=640, color='gray', ls='--')
                #     ax.set_xticks(np.arange(0, 61, 4)*10+5)
                #     ax.set_xticklabels(np.arange(0, 61, 4)+10)
                #     plt.xticks(rotation=0)
                #     plt.tick_params(labelsize=FONT_SIZE)
                #     fig.set_size_inches(7, 5)
                #     plt.tight_layout()
                #     plt.savefig('figures/tree_s' + str(season) + 'e' + str(total_episode) + '.pdf')
                #     plt.show()
                #     return
                
                parts = [len(s) for s in sept]
                sept = [0]
                count = 0
                for i in range(len(parts)):
                    count += parts[i]
                    sept.append(count)
                if USE_DOUBLE:
                    for i in range(len(sept)):
                        sept[i] = int(sept[i] / 2)
                else:
                    if len(emotion_all_percent[starting:ending]) % 2 == 1:
                        sept[-2] -= 1
                        sept[-1] -= 1
                # print(season, episode, sept, parts)


            for sept_index in range(NUM_SEPT):
                if USE_AVERAGE:
                    div = sept[sept_index+1]-sept[sept_index]
                cur_emotion_all_percent = emotion_all_percent[sept[sept_index]:sept[sept_index+1]]
                cur_emotion_negative = emotion_negative[sept[sept_index]:sept[sept_index+1]]
                cur_emotion_positive = emotion_positive[sept[sept_index]:sept[sept_index+1]]
                cur_result = get_peak(cur_emotion_all_percent)
                all_positive_peak, all_positive_duration, all_negative_peak, all_negative_duration, all_frequency, pos_peak_end, neg_peak_end = get_feature(cur_result)
                pos_k,pos_b,pos_peak,pos_max,pos_position = get_peak_2(cur_emotion_positive,'_pos_slope')
                neg_k,neg_b,neg_peak,neg_max,neg_position = get_peak_2(cur_emotion_negative,'_neg_slope')

                pos_div_neg = []
                for i in range(len(cur_emotion_positive)):
                    if cur_emotion_negative[i] == 0:
                        pos_div_neg.append(0)
                    else:
                        pos_div_neg.append(cur_emotion_positive[i]/cur_emotion_negative[i])

                feature[str(sept_index) + '_all_positive_peak'].append(all_positive_peak / div)
                feature[str(sept_index) + '_all_positive_duration'].append(all_positive_duration / div)
                feature[str(sept_index) + '_all_negative_peak'].append(all_negative_peak / div)
                feature[str(sept_index) + '_all_negative_duration'].append(all_negative_duration / div)
                feature[str(sept_index) + '_pos_k'].append(pos_k)
                feature[str(sept_index) + '_pos_b'].append(pos_b)
                feature[str(sept_index) + '_neg_k'].append(neg_k)
                feature[str(sept_index) + '_neg_b'].append(neg_b)
                feature[str(sept_index) + '_pos_peak'].append(pos_peak)
                feature[str(sept_index) + '_pos_max'].append(pos_max)
                feature[str(sept_index) + '_neg_peak'].append(neg_peak)
                feature[str(sept_index) + '_neg_max'].append(neg_max)
                if pos_position > neg_position:
                    position = -1
                elif pos_position < neg_position:
                    position = 1
                else:
                    position = 0
                feature[str(sept_index) + '_pos_neg'].append(position)
                feature[str(sept_index) + '_pos_div_neg'].append(np.mean(pos_div_neg))

    # for key in feature.keys():
    #     print(key, len(feature[key]))
    feature_pd = pd.DataFrame.from_dict(feature)
    feature_pd.to_excel("feature_" + str(SELECT_PERCENT) + "_" + str(NUM_SEPT) + ".xlsx",encoding='utf-8',index=False, engine='xlsxwriter')


def evaluate(var):
    feature_pd = pd.read_excel("feature_" + str(SELECT_PERCENT) + ".xlsx",encoding='utf-8',index=False)
    x = feature_pd[var].values
    res, a_rsq = [], []
    for pred_var in PREDICT_VAR:
        y = feature_pd[pred_var].values
        y_pred = []
        for test in range(STARTING_TEST, 70):
            model = LinearRegression()
            model = model.fit(x[:test], y[:test])
            pred = model.predict([x[test]])
            y_pred.append(pred[0]-y[test])
        res.append(np.array(y_pred))

        model = LinearRegression()
        model = model.fit(x, y)
        a_rsq.append( 1 - (1 - model.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1) )
    return np.array(res), np.array(a_rsq)    # 3*num_pred, 3


def evaluate_percents():
    total_rmse, total_ar = [], []
    for epoch in range(30):
        cur_rmse, cur_ar = [], []
        for percent in PERCENTS:
            SELECT_PERCENT = percent
            generate_tables()
            res, ar = evaluate(STAT_VAR+NO_PEAK_VAR)
            cur_rmse.append(res)
            cur_ar.append(ar)
        total_rmse.append(np.array(cur_rmse))
        total_ar.append(np.array(cur_ar))
        print(epoch)

    total_rmse = np.array(total_rmse)    # epochs*num_per*3*num_pred
    total_ar = np.array(total_ar)   # epochs*num_per*3

    print(total_rmse.shape)
    rmse = np.sqrt(np.mean(np.square(total_rmse), axis=0))
    adjusted_r_square = np.mean(total_ar, axis=0)
    print(rmse.shape)

    for i in range(len(PERCENTS)):
        np.savetxt('rmse_' + str(PERCENTS[i]) + '.csv', rmse[i].T, delimiter=',')   # num_pred*3
    np.savetxt('adjust_r' + '.csv', adjusted_r_square, delimiter=',')   # num_per*3




def evaluate_adj_rsquare():
    s = time.time()
    SELECT_PERCENT = 1.0
    # generate_tables()
    feature_pd = pd.read_excel("feature_" + str(SELECT_PERCENT) + ".xlsx",encoding='utf-8',index=False)
    x = feature_pd[STAT_VAR+NO_PEAK_VAR+PEAK_END_VAR].values
    res, a_rsq = [], []
    for pred_var in PREDICT_VAR:
        y = feature_pd[pred_var].values
        model = LinearRegression()
        model = model.fit(x, y)
        a_rsq.append( 1 - (1 - model.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1) )

        # pred = model.predict(x)
        # print(score(pred, y, x.shape[1]))
    a_rsq = np.array(a_rsq)
    print(a_rsq)




# The input shape should be [samples, time steps, features]
def create_dataset(dataset, label, step=1):
    X, Y = [], []
    for i in range(len(dataset)-step+1):
        X.append(dataset[i:i+step, :])
        Y.append(label[i+step-1])
    return np.array(X).reshape(len(X), step, -1), np.array(Y)

def evaluate_train(var):
    feature_pd = pd.read_excel("feature_" + str(SELECT_PERCENT) + ".xlsx",encoding='utf-8',index=False)
    x = feature_pd[var].values
    pred_var = PREDICT_VAR[0]
    y = feature_pd[pred_var].values
    model = LinearRegression()
    model = model.fit(x, y)
    pred = model.predict(x[STARTING_TEST:])
    y_pred = np.abs(pred-y[STARTING_TEST:])
    return y_pred

def evaluate_train_LSTM(var):
    step = 5
    feature_pd = pd.read_excel("feature_" + str(SELECT_PERCENT) + ".xlsx",encoding='utf-8',index=False)
    x = feature_pd[var].values
    pred_var = PREDICT_VAR[0]
    y = feature_pd[pred_var].values
    split_x, split_y = create_dataset(x, y, step=step)

    model = Sequential()
    model.add(LSTM(32, activation='relu', return_sequences=False, input_shape = (split_x.shape[1], split_x.shape[2])))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.01))
    es = EarlyStopping(monitor= 'val_loss', patience=20)
    model.fit(split_x, split_y, epochs=2000, verbose=0, shuffle=False, callbacks=[es])
    pred = model.predict(split_x[STARTING_TEST-step+1:])
    y_pred = np.abs(pred[:,0]-split_y[STARTING_TEST-step+1:])
    return y_pred




def evaluate_all():
    SELECT_PERCENT = 1.0
    # generate_tables()
    res = evaluate_train_LSTM(STAT_VAR+NO_PEAK_VAR)
    print(res, np.mean(res))
    np.savetxt('rmse_6_only_LSTM.csv', np.abs(res.T), delimiter=',')

# s = time.time()
# evaluate_all()
# print("Total Time:", time.time()-s)
    



def emotion_bullet(season, episode=24, number_sample=30):
    # 导入数据
    with open("season" + str(season) + "/Bullet_content_time_uid_emotion/Bullet_S" + str(season) + "E" + str(episode) + ".json",'r',encoding='utf-8') as f:
        data_emotion = json.load(f)
    #画第一个pos和neg图
    divide = 60 
    data_percent = []
    for i in range(int(7414/divide + 1)):
        data_percent.append({"neg_num": 0,'neu_num':0 ,'pos_num':0 ,'second': i })


    data_emotion = sample(data_emotion, number_sample)
    list_bullet, list_sentiment = [], []

    for item_data in data_emotion:
        try: 
            m = int(int(item_data[1])/divide)
            item = item_data[3]
            if item['sentiment'] == 0:
                list_sentiment.append(-1)
                list_bullet.append(item_data[0])
            elif item['sentiment'] == 2:
                list_sentiment.append(1)
                list_bullet.append(item_data[0])
        except Exception as e:
            print("Exception:"+str(e))   
    return list_bullet, list_sentiment


if __name__ == '__main__':
    generate_tables()

    # list_bullet, list_sentiment = [], []
    # for season in range(1, 7):
    #     for total_episode in range(1,NUM_EPISODE[season-1]):
    #         cur_bullet, cur_sentiment = emotion_bullet(season, episode=total_episode, number_sample=10)
    #         list_bullet += cur_bullet
    #         list_sentiment += cur_sentiment

    # df_save_bullets = pd.DataFrame()
    # df_save_bullets['Bullet'] = list_bullet
    # df_save_bullets['Sentiment'] = list_sentiment
    # df_save_bullets.to_excel("Sample_Live_Comments.xlsx",encoding='utf-8',index=False, engine='xlsxwriter')