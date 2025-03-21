import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import torch.optim as optim
import time
import os
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"]="0"
USE_CUDA = True

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



class MyTrans(nn.Module):
  # Transformer model for IMDB binary classification
  # n_tokens: num distinct words == vocabulary size
  # embed_dim: num values for each word/token is embed_dim
  # n_heads: num attention heads, needed by Tr_EncoderLayer
  # n_hid : num hidden nodes in NN part of Tr_EncoderLayer
  # n_eclayers: num Tr_EncoderLayer layers in Tr_Encoder
  # drop_p is used by PositionalEncoding AND Tr_EncoderLayer

  def __init__(self, n_tokens, embed_dim, n_heads, n_hid, \
    n_eclayers, pred_dim=3, drop_p=0.5):
    super(MyTrans, self).__init__()
    self.embed_dim = embed_dim

    self.embedder = nn.Linear(n_tokens, embed_dim)#nn.Embedding(n_tokens, embed_dim)
    self.pos_encoder = PositionalEncoding(embed_dim, drop_p)
    enc_layer = nn.TransformerEncoderLayer(embed_dim, \
      n_heads, n_hid, drop_p)
    self.transformer_encoder = \
      nn.TransformerEncoder(enc_layer, n_eclayers)
    # map 4 embed vals to 3 classes
    self.to_logits = nn.Linear(embed_dim, pred_dim)  

    self.embedder.weight.data.uniform_(-0.01, 0.01)
    self.to_logits.weight.data.uniform_(-0.01, 0.01)
    self.to_logits.bias.data.zero_()

  def forward(self, src, src_mask=None):
    src = src.permute(1,0,2)
    z = self.embedder(src) * math.sqrt(self.embed_dim) 
    z = self.pos_encoder(z)

    oupt = self.transformer_encoder(z)
    oupt = oupt.max(dim=0)[0]  # [3,4] == [bat, 'one word']
    oupt = self.to_logits(oupt)    # [3,2] == [bat, class]
    return oupt








# The input shape should be [samples, time steps, features]
def create_dataset(dataset, label, step=1):
    X, Y = [], []
    for i in range(len(dataset)-step+1):
        X.append(dataset[i:i+step, :])
        Y.append(label[i+step-1])
    return np.array(X).reshape(len(X), step, -1), np.array(Y)


def evaluate_all(df_feature, feat, pred_var):
    step = 10
    STARTING_TEST = 112

    x = df_feature[feat].values
    y = df_feature[pred_var].values

    split_x, split_y = create_dataset(x, y, step=step)
    split_x = torch.as_tensor(torch.from_numpy(split_x), dtype=torch.float32)
    split_y = torch.as_tensor(torch.from_numpy(split_y), dtype=torch.float32)
    y_pred = []
    for test in range(STARTING_TEST, len(x)):
        model = MyTrans(len(feat), 256, 256, 2, 2, pred_dim=1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        if USE_CUDA:
            model = model.cuda()
            split_x = split_x.cuda()
            split_y = split_y.cuda()
        model.train()
        for e in range(100):
            pred = model(split_x[:test-step])
            loss = torch.sum((pred - split_y[:test-step])**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = model(split_x[test-step:test-step+1])
        y_pred.append(torch.abs(pred[0][0]-split_y[test-step]).cpu().detach().numpy())
    rmse = np.sqrt(np.mean(np.square(np.array(y_pred))))
    return np.array(y_pred), rmse



def evaluate_lstm(df_feature, feat, pred_var):
    step = 10
    STARTING_TEST = 112

    x = df_feature[feat].values
    y = df_feature[pred_var].values

    split_x, split_y = create_dataset(x, y, step=step)
    y_pred = []
    for test in range(STARTING_TEST, len(x)):
        model = Sequential()
        model.add(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False, input_shape = (split_x.shape[1], split_x.shape[2])))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.0003))
        es = EarlyStopping(monitor= 'loss', patience=10)
        model.fit(split_x[:test-step], split_y[:test-step], epochs=5000, verbose=0, shuffle=False, callbacks=[es])
        pred = model.predict(split_x[test-step:test-step+1])
        y_pred.append(np.abs(pred[-1][-1]-split_y[test-step]))
    rmse = np.sqrt(np.mean(np.square(np.array(y_pred))))
    return np.array(y_pred), rmse




start_time = time.time()
df_feature = pd.read_csv("feature_0.2.csv", index_col=None)
labels = ['CAR', 'CAR+DUM', 'SIG', 'ALL']
features = [['pre_emotioncomment'],
            ['pre_emotioncomment','dummy'],
            ['survey','pre_emotioncomment','1_pos_k','2_minus','3_all_positive_duration','3_neg_k']]
features.append(df_feature.columns[4:])
pred_var = 'emotioncomment'


for i in range(2,len(features)):
    cur_time = time.time()
    cur_rmse = []
    for seed in range(10):
        np.random.seed(seed)
        res_cur, rmse_cur = evaluate_lstm(df_feature, features[i], pred_var)
        cur_rmse.append(res_cur)
    np.savetxt('prediction/result_' + pred_var + '_' + labels[i] + '.csv', np.array(cur_rmse), delimiter=',')
    print(labels[i], 'rmse:', np.mean(cur_rmse, axis=0), time.time()-cur_time)
print("Total Time:", time.time()-start_time)