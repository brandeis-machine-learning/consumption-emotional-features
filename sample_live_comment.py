import pandas as pd
import numpy as np

samples = pd.read_csv('Sample_Live_Comments.csv', index_col=None, encoding='latin-1')

ml = samples['Sentiment'].values
hm = samples['Human'].values

correct, total = 0, 0
for i in range(len(ml)):
    total += 1
    if hm[i] != 0:
        # total += 1
        if hm[i] == ml[i]:
            correct += 1
    else:
        correct += 1

print(total, correct, correct/total)