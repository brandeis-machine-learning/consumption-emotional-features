import urllib3,urllib
import json
import ast
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pylab
from scipy.signal import find_peaks,find_peaks_cwt
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
from random import sample
import warnings
warnings.filterwarnings("ignore")

np.random.seed(9) # 0.2=9, 0.15=6, 0.1=9
NUM_EPISODE = [25, 23, 19, 25, 25, 28]
NUM_SEPT = 5
OFFSET_TIME = [5, 5, 5, 10, 10, 10]
CUT_HEAD = True
CUT_END = True
USE_AVERAGE = False
USE_BALANCE = True
USE_DOUBLE = True
SELECT_PERCENT = 0.1


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


def emotion_wave(season, episode=24):
    # 导入数据
    with open("season" + str(season) + "/Bullet_content_time_uid_emotion/Bullet_S" + str(season) + "E" + str(episode) + ".json",'r',encoding='utf-8') as f:
        data_emotion = json.load(f)
    #画第一个pos和neg图
    divide = 60 
    data_percent = []
    for i in range(int(7414/divide + 1)):
        data_percent.append({"neg_num": 0,'neu_num':0 ,'pos_num':0 ,'second': i })

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
    for item in result:
        #pos peak and freq
        if item != pre and item == 1:
            all_positive_peak += 1
            if pre_emotion == -1:
                frequency += 1
            pre_emotion = item
        #neg peak and freq
        if item != pre and item == -1:
            all_negative_peak += 1
            if pre_emotion == 1:
                frequency += 1
            pre_emotion = item
        #pos duration
        if item == pre and item == 1:
            all_positive_duration += 1
        #neg duration
        if item == pre and item == -1:
            all_negative_duration += 1
        pre = item
    # print("pos peak:%d dur:%d neg peak:%d dur:%d freq:%d" 
    #       % (all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,frequency))
    return all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,frequency


def get_peak_2(time_series,title="",episode=24):
    indices = find_peaks(time_series,prominence=10)[0]
    try:
        x = np.array(indices)
        y = np.array([time_series[j] for j in indices])
        z1 = np.polyfit(x, y, 1)
    except Exception:
        return 0,0
    return z1[0],z1[1]


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
            error += np.log2(residuals[0])
        except IndexError:
            error += 0.0
    return error * weight


def double_list(l):
    if USE_DOUBLE:
        return [val for val in l for i in range(2)]
    else:
        return l


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

        Z_print.append(np.array([Z_index[index], Z_index[index + 1], float(len(Seg_TS[index])/2 - 1), len(Seg_TS[index])]))
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



if NUM_SEPT == 3:
    feature = {'time':[],'textbullet':[],'emojibullet':[],'wave_mean':[],'wave_std':[],'range':[],
               'all_positive_peak':[],'all_positive_duration':[],'all_negative_peak':[],'all_negative_duration':[],
               'all_frequency':[],'abs_k':[],'abs_b':[],'pos_k':[],'pos_b':[],'neg_k':[],'neg_b':[],
              '0_time':[],'0_wave_mean':[],'0_wave_std':[],'0_range':[],
               '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
               '0_all_frequency':[],'0_abs_k':[],'0_abs_b':[],'0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],
              '1_time':[],'1_wave_mean':[],'1_wave_std':[],'1_range':[],
               '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
               '1_all_frequency':[],'1_abs_k':[],'1_abs_b':[],'1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],
              '2_time':[],'2_wave_mean':[],'2_wave_std':[],'2_range':[],
               '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
               '2_all_frequency':[],'2_abs_k':[],'2_abs_b':[],'2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[]}
elif NUM_SEPT == 4:
    feature = {'time':[],'textbullet':[],'emojibullet':[],'wave_mean':[],'wave_std':[],'range':[],
               'all_positive_peak':[],'all_positive_duration':[],'all_negative_peak':[],'all_negative_duration':[],
               'all_frequency':[],'abs_k':[],'abs_b':[],'pos_k':[],'pos_b':[],'neg_k':[],'neg_b':[],
              '0_time':[],'0_wave_mean':[],'0_wave_std':[],'0_range':[],
               '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
               '0_all_frequency':[],'0_abs_k':[],'0_abs_b':[],'0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],
              '1_time':[],'1_wave_mean':[],'1_wave_std':[],'1_range':[],
               '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
               '1_all_frequency':[],'1_abs_k':[],'1_abs_b':[],'1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],
              '2_time':[],'2_wave_mean':[],'2_wave_std':[],'2_range':[],
               '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
               '2_all_frequency':[],'2_abs_k':[],'2_abs_b':[],'2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[],
              '3_time':[],'3_wave_mean':[],'3_wave_std':[],'3_range':[],
               '3_all_positive_peak':[],'3_all_positive_duration':[],'3_all_negative_peak':[],'3_all_negative_duration':[],
               '3_all_frequency':[],'3_abs_k':[],'3_abs_b':[],'3_pos_k':[],'3_pos_b':[],'3_neg_k':[],'3_neg_b':[]}
elif NUM_SEPT == 5:
    feature = {'time':[],'textbullet':[],'emojibullet':[],'wave_mean':[],'wave_std':[],'range':[],
               'all_positive_peak':[],'all_positive_duration':[],'all_negative_peak':[],'all_negative_duration':[],
               'all_frequency':[],'abs_k':[],'abs_b':[],'pos_k':[],'pos_b':[],'neg_k':[],'neg_b':[],
              '0_time':[],'0_wave_mean':[],'0_wave_std':[],'0_range':[],
               '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
               '0_all_frequency':[],'0_abs_k':[],'0_abs_b':[],'0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],
              '1_time':[],'1_wave_mean':[],'1_wave_std':[],'1_range':[],
               '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
               '1_all_frequency':[],'1_abs_k':[],'1_abs_b':[],'1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],
              '2_time':[],'2_wave_mean':[],'2_wave_std':[],'2_range':[],
               '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
               '2_all_frequency':[],'2_abs_k':[],'2_abs_b':[],'2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[],
              '3_time':[],'3_wave_mean':[],'3_wave_std':[],'3_range':[],
               '3_all_positive_peak':[],'3_all_positive_duration':[],'3_all_negative_peak':[],'3_all_negative_duration':[],
               '3_all_frequency':[],'3_abs_k':[],'3_abs_b':[],'3_pos_k':[],'3_pos_b':[],'3_neg_k':[],'3_neg_b':[],
              '4_time':[],'4_wave_mean':[],'4_wave_std':[],'4_range':[],
               '4_all_positive_peak':[],'4_all_positive_duration':[],'4_all_negative_peak':[],'4_all_negative_duration':[],
               '4_all_frequency':[],'4_abs_k':[],'4_abs_b':[],'4_pos_k':[],'4_pos_b':[],'4_neg_k':[],'4_neg_b':[]}
elif NUM_SEPT == 6:
    feature = {'time':[],'textbullet':[],'emojibullet':[],'wave_mean':[],'wave_std':[],'range':[],
               'all_positive_peak':[],'all_positive_duration':[],'all_negative_peak':[],'all_negative_duration':[],
               'all_frequency':[],'abs_k':[],'abs_b':[],'pos_k':[],'pos_b':[],'neg_k':[],'neg_b':[],
              '0_time':[],'0_wave_mean':[],'0_wave_std':[],'0_range':[],
               '0_all_positive_peak':[],'0_all_positive_duration':[],'0_all_negative_peak':[],'0_all_negative_duration':[],
               '0_all_frequency':[],'0_abs_k':[],'0_abs_b':[],'0_pos_k':[],'0_pos_b':[],'0_neg_k':[],'0_neg_b':[],
              '1_time':[],'1_wave_mean':[],'1_wave_std':[],'1_range':[],
               '1_all_positive_peak':[],'1_all_positive_duration':[],'1_all_negative_peak':[],'1_all_negative_duration':[],
               '1_all_frequency':[],'1_abs_k':[],'1_abs_b':[],'1_pos_k':[],'1_pos_b':[],'1_neg_k':[],'1_neg_b':[],
              '2_time':[],'2_wave_mean':[],'2_wave_std':[],'2_range':[],
               '2_all_positive_peak':[],'2_all_positive_duration':[],'2_all_negative_peak':[],'2_all_negative_duration':[],
               '2_all_frequency':[],'2_abs_k':[],'2_abs_b':[],'2_pos_k':[],'2_pos_b':[],'2_neg_k':[],'2_neg_b':[],
              '3_time':[],'3_wave_mean':[],'3_wave_std':[],'3_range':[],
               '3_all_positive_peak':[],'3_all_positive_duration':[],'3_all_negative_peak':[],'3_all_negative_duration':[],
               '3_all_frequency':[],'3_abs_k':[],'3_abs_b':[],'3_pos_k':[],'3_pos_b':[],'3_neg_k':[],'3_neg_b':[],
              '4_time':[],'4_wave_mean':[],'4_wave_std':[],'4_range':[],
               '4_all_positive_peak':[],'4_all_positive_duration':[],'4_all_negative_peak':[],'4_all_negative_duration':[],
               '4_all_frequency':[],'4_abs_k':[],'4_abs_b':[],'4_pos_k':[],'4_pos_b':[],'4_neg_k':[],'4_neg_b':[],
              '5_time':[],'5_wave_mean':[],'5_wave_std':[],'5_range':[],
               '5_all_positive_peak':[],'5_all_positive_duration':[],'5_all_negative_peak':[],'5_all_negative_duration':[],
               '5_all_frequency':[],'5_abs_k':[],'5_abs_b':[],'5_pos_k':[],'5_pos_b':[],'5_neg_k':[],'5_neg_b':[],}

for season in range(1, 7):
    for episode in range(1,NUM_EPISODE[season-1]):
        emotion_all_percent, emotion_all_percent_abs, emotion_negative, emotion_positive, num_bullet, num_emoji_bullet = emotion_wave(season, episode)
        result = get_peak(emotion_all_percent,episode=episode)
        all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency = get_feature(result)
        abs_k,abs_b = get_peak_2(emotion_all_percent_abs,'_all_abs_slope',episode)
        pos_k,pos_b = get_peak_2(emotion_positive,'_pos_slope',episode)
        neg_k,neg_b = get_peak_2(emotion_negative,'_neg_slope',episode)        
        
        end_time = 124
        for i in range(len(emotion_all_percent_abs)):
            if emotion_all_percent_abs[123-i] != 0:
                end_time = 124-i
                break
        starting = OFFSET_TIME[season - 1]
        ending = end_time - OFFSET_TIME[season - 1]
        div = 1

        if USE_AVERAGE:
            div = end_time

        feature['textbullet'].append(np.log10(num_bullet))
        feature['emojibullet'].append(np.log10(num_emoji_bullet))
        feature['wave_mean'].append(np.mean(emotion_all_percent))
        feature['wave_std'].append(np.std(emotion_all_percent))
        feature['all_positive_peak'].append(all_positive_peak / div)
        feature['all_positive_duration'].append(all_positive_duration / div)
        feature['all_negative_peak'].append(all_negative_peak / div)
        feature['all_negative_duration'].append(all_negative_duration / div)
        feature['all_frequency'].append(all_frequency / div)
        feature['abs_k'].append(abs_k)
        feature['abs_b'].append(abs_b)
        feature['pos_k'].append(pos_k)
        feature['pos_b'].append(pos_b)
        feature['neg_k'].append(neg_k)
        feature['neg_b'].append(neg_b)
        feature['range'].append(np.max(emotion_all_percent_abs))
        
        # if NUM_SEPT == 3:
        #     sept = [0, starting, ending, end_time]
        # elif NUM_SEPT == 4:
        #     sept = [0, starting, starting + int((ending - starting)/2), ending, end_time]
        # elif NUM_SEPT == 5:
        #     sept = [0, starting, starting + int((ending - starting)/3), starting + int((ending - starting)/3*2), ending, end_time]
        # elif NUM_SEPT == 6:
        #     sept = [0, starting, starting + int((ending - starting)/4), starting + int((ending - starting)/4*2), starting + int((ending - starting)/4*3), ending, end_time]

        # if season > 1:
        #     break

        sept, all_res, Z_print = Bottom_Up(emotion_all_percent[:end_time], emotion_positive[:end_time], emotion_negative[:end_time], emotion_all_percent_abs[:end_time], NUM_SEPT, starting, ending)
        
        # if season==1 and episode==1:
        #     fig = plt.figure()
        #     dn = dendrogram(Z_print)
        #     plt.xlabel('Time')
        #     plt.ylabel('Duration of Each Part')
        #     plt.show()


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
            
            
        feature['time'].append(end_time)
        feature['0_time'].append(sept[1]-sept[0])
        feature['1_time'].append(sept[2]-sept[1])
        feature['2_time'].append(sept[3]-sept[2])

        emotion_all_percent_0 = emotion_all_percent[sept[0]:sept[1]]
        emotion_all_percent_1 = emotion_all_percent[sept[1]:sept[2]]
        emotion_all_percent_2 = emotion_all_percent[sept[2]:sept[3]]
        emotion_all_percent_abs_0 = emotion_all_percent_abs[sept[0]:sept[1]]
        emotion_all_percent_abs_1 = emotion_all_percent_abs[sept[1]:sept[2]]
        emotion_all_percent_abs_2 = emotion_all_percent_abs[sept[2]:sept[3]]
        emotion_negative_0 = emotion_negative[sept[0]:sept[1]]
        emotion_negative_1 = emotion_negative[sept[1]:sept[2]]
        emotion_negative_2 = emotion_negative[sept[2]:sept[3]]
        emotion_positive_0 = emotion_positive[sept[0]:sept[1]]
        emotion_positive_1 = emotion_positive[sept[1]:sept[2]]
        emotion_positive_2 = emotion_positive[sept[2]:sept[3]]

        if NUM_SEPT >= 4:
            feature['3_time'].append(sept[4]-sept[3])
            emotion_all_percent_3 = emotion_all_percent[sept[3]:sept[4]]
            emotion_all_percent_abs_3 = emotion_all_percent_abs[sept[3]:sept[4]]
            emotion_negative_3 = emotion_negative[sept[3]:sept[4]]
            emotion_positive_3 = emotion_positive[sept[3]:sept[4]]
        if NUM_SEPT >= 5:
            feature['4_time'].append(sept[5]-sept[4])
            emotion_all_percent_4 = emotion_all_percent[sept[4]:sept[5]]
            emotion_all_percent_abs_4 = emotion_all_percent_abs[sept[4]:sept[5]]
            emotion_negative_4 = emotion_negative[sept[4]:sept[5]]
            emotion_positive_4 = emotion_positive[sept[4]:sept[5]]
        if NUM_SEPT >= 6:
            feature['5_time'].append(sept[6]-sept[5])
            emotion_all_percent_5 = emotion_all_percent[sept[5]:sept[6]]
            emotion_all_percent_abs_5 = emotion_all_percent_abs[sept[5]:sept[6]]
            emotion_negative_5 = emotion_negative[sept[5]:sept[6]]
            emotion_positive_5 = emotion_positive[sept[5]:sept[6]]
        
        
        if USE_AVERAGE:
            div = sept[1]-sept[0]
        emotion_all_percent = emotion_all_percent_0
        emotion_all_percent_abs = emotion_all_percent_abs_0
        emotion_negative = emotion_negative_0
        emotion_positive = emotion_positive_0
        result = get_peak(emotion_all_percent,episode=i)
        all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency = get_feature(result)
        abs_k,abs_b = get_peak_2(emotion_all_percent_abs,'_all_abs_slope',i)
        pos_k,pos_b = get_peak_2(emotion_positive,'_pos_slope',i)
        neg_k,neg_b = get_peak_2(emotion_negative,'_neg_slope',i)
        feature['0_wave_mean'].append(np.mean(emotion_all_percent))
        feature['0_wave_std'].append(np.std(emotion_all_percent))
        feature['0_all_positive_peak'].append(all_positive_peak / div)
        feature['0_all_positive_duration'].append(all_positive_duration / div)
        feature['0_all_negative_peak'].append(all_negative_peak / div)
        feature['0_all_negative_duration'].append(all_negative_duration / div)
        feature['0_all_frequency'].append(all_frequency / div)
        feature['0_abs_k'].append(abs_k)
        feature['0_abs_b'].append(abs_b)
        feature['0_pos_k'].append(pos_k)
        feature['0_pos_b'].append(pos_b)
        feature['0_neg_k'].append(neg_k)
        feature['0_neg_b'].append(neg_b)
        feature['0_range'].append(np.max(emotion_all_percent_abs_0))

        if USE_AVERAGE:
            div = sept[2]-sept[1]
        emotion_all_percent = emotion_all_percent_1
        emotion_all_percent_abs = emotion_all_percent_abs_1
        emotion_negative = emotion_negative_1
        emotion_positive = emotion_positive_1
        result = get_peak(emotion_all_percent,episode=i)
        all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency = get_feature(result)
        abs_k,abs_b = get_peak_2(emotion_all_percent_abs,'_all_abs_slope',i)
        pos_k,pos_b = get_peak_2(emotion_positive,'_pos_slope',i)
        neg_k,neg_b = get_peak_2(emotion_negative,'_neg_slope',i)
        feature['1_wave_mean'].append(np.mean(emotion_all_percent))
        feature['1_wave_std'].append(np.std(emotion_all_percent))
        feature['1_all_positive_peak'].append(all_positive_peak / div)
        feature['1_all_positive_duration'].append(all_positive_duration / div)
        feature['1_all_negative_peak'].append(all_negative_peak / div)
        feature['1_all_negative_duration'].append(all_negative_duration / div)
        feature['1_all_frequency'].append(all_frequency / div)
        feature['1_abs_k'].append(abs_k)
        feature['1_abs_b'].append(abs_b)
        feature['1_pos_k'].append(pos_k)
        feature['1_pos_b'].append(pos_b)
        feature['1_neg_k'].append(neg_k)
        feature['1_neg_b'].append(neg_b)
        feature['1_range'].append(np.max(emotion_all_percent_abs_1))

        if USE_AVERAGE:
            div = sept[3]-sept[2]
        emotion_all_percent = emotion_all_percent_2
        emotion_all_percent_abs = emotion_all_percent_abs_2
        emotion_negative = emotion_negative_2
        emotion_positive = emotion_positive_2
        result = get_peak(emotion_all_percent,episode=i)
        all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency = get_feature(result)
        abs_k,abs_b = get_peak_2(emotion_all_percent_abs,'_all_abs_slope',i)
        pos_k,pos_b = get_peak_2(emotion_positive,'_pos_slope',i)
        neg_k,neg_b = get_peak_2(emotion_negative,'_neg_slope',i)
        feature['2_wave_mean'].append(np.mean(emotion_all_percent))
        feature['2_wave_std'].append(np.std(emotion_all_percent))
        feature['2_all_positive_peak'].append(all_positive_peak / div)
        feature['2_all_positive_duration'].append(all_positive_duration / div)
        feature['2_all_negative_peak'].append(all_negative_peak / div)
        feature['2_all_negative_duration'].append(all_negative_duration / div)
        feature['2_all_frequency'].append(all_frequency / div)
        feature['2_abs_k'].append(abs_k)
        feature['2_abs_b'].append(abs_b)
        feature['2_pos_k'].append(pos_k)
        feature['2_pos_b'].append(pos_b)
        feature['2_neg_k'].append(neg_k)
        feature['2_neg_b'].append(neg_b)
        feature['2_range'].append(np.max(emotion_all_percent_abs_2))

        if NUM_SEPT >= 4:
            if USE_AVERAGE:
                div = sept[4]-sept[3]
            emotion_all_percent = emotion_all_percent_3
            emotion_all_percent_abs = emotion_all_percent_abs_3
            emotion_negative = emotion_negative_3
            emotion_positive = emotion_positive_3
            result = get_peak(emotion_all_percent,episode=i)
            all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency = get_feature(result)
            abs_k,abs_b = get_peak_2(emotion_all_percent_abs,'_all_abs_slope',i)
            pos_k,pos_b = get_peak_2(emotion_positive,'_pos_slope',i)
            neg_k,neg_b = get_peak_2(emotion_negative,'_neg_slope',i)
            feature['3_wave_mean'].append(np.mean(emotion_all_percent))
            feature['3_wave_std'].append(np.std(emotion_all_percent))
            feature['3_all_positive_peak'].append(all_positive_peak / div)
            feature['3_all_positive_duration'].append(all_positive_duration / div)
            feature['3_all_negative_peak'].append(all_negative_peak / div)
            feature['3_all_negative_duration'].append(all_negative_duration / div)
            feature['3_all_frequency'].append(all_frequency / div)
            feature['3_abs_k'].append(abs_k)
            feature['3_abs_b'].append(abs_b)
            feature['3_pos_k'].append(pos_k)
            feature['3_pos_b'].append(pos_b)
            feature['3_neg_k'].append(neg_k)
            feature['3_neg_b'].append(neg_b)
            feature['3_range'].append(np.max(emotion_all_percent_abs_3))
        if NUM_SEPT >= 5:
            if USE_AVERAGE:
                div = sept[5]-sept[4]
            emotion_all_percent = emotion_all_percent_4
            emotion_all_percent_abs = emotion_all_percent_abs_4
            emotion_negative = emotion_negative_4
            emotion_positive = emotion_positive_4
            result = get_peak(emotion_all_percent,episode=i)
            all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency = get_feature(result)
            abs_k,abs_b = get_peak_2(emotion_all_percent_abs,'_all_abs_slope',i)
            pos_k,pos_b = get_peak_2(emotion_positive,'_pos_slope',i)
            neg_k,neg_b = get_peak_2(emotion_negative,'_neg_slope',i)
            feature['4_wave_mean'].append(np.mean(emotion_all_percent))
            feature['4_wave_std'].append(np.std(emotion_all_percent))
            feature['4_all_positive_peak'].append(all_positive_peak / div)
            feature['4_all_positive_duration'].append(all_positive_duration / div)
            feature['4_all_negative_peak'].append(all_negative_peak / div)
            feature['4_all_negative_duration'].append(all_negative_duration / div)
            feature['4_all_frequency'].append(all_frequency / div)
            feature['4_abs_k'].append(abs_k)
            feature['4_abs_b'].append(abs_b)
            feature['4_pos_k'].append(pos_k)
            feature['4_pos_b'].append(pos_b)
            feature['4_neg_k'].append(neg_k)
            feature['4_neg_b'].append(neg_b)
            feature['4_range'].append(np.max(emotion_all_percent_abs_4))
        if NUM_SEPT >= 6:
            if USE_AVERAGE:
                div = sept[6]-sept[5]
            emotion_all_percent = emotion_all_percent_5
            emotion_all_percent_abs = emotion_all_percent_abs_5
            emotion_negative = emotion_negative_5
            emotion_positive = emotion_positive_5
            result = get_peak(emotion_all_percent,episode=i)
            all_positive_peak,all_positive_duration,all_negative_peak,all_negative_duration,all_frequency = get_feature(result)
            abs_k,abs_b = get_peak_2(emotion_all_percent_abs,'_all_abs_slope',i)
            pos_k,pos_b = get_peak_2(emotion_positive,'_pos_slope',i)
            neg_k,neg_b = get_peak_2(emotion_negative,'_neg_slope',i)
            feature['5_wave_mean'].append(np.mean(emotion_all_percent))
            feature['5_wave_std'].append(np.std(emotion_all_percent))
            feature['5_all_positive_peak'].append(all_positive_peak / div)
            feature['5_all_positive_duration'].append(all_positive_duration / div)
            feature['5_all_negative_peak'].append(all_negative_peak / div)
            feature['5_all_negative_duration'].append(all_negative_duration / div)
            feature['5_all_frequency'].append(all_frequency / div)
            feature['5_abs_k'].append(abs_k)
            feature['5_abs_b'].append(abs_b)
            feature['5_pos_k'].append(pos_k)
            feature['5_pos_b'].append(pos_b)
            feature['5_neg_k'].append(neg_k)
            feature['5_neg_b'].append(neg_b)
            feature['5_range'].append(np.max(emotion_all_percent_abs_5))


feature_pd = pd.DataFrame.from_dict(feature)
feature_pd.to_excel("feature_" + str(SELECT_PERCENT) + ".xlsx",encoding='utf-8',index=True, engine='xlsxwriter')