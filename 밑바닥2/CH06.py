#!/usr/bin/env python
# coding: utf-8

# # 게이트가 추가된 RNN
# 
# 

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

N = 2 # 미니배치 크기
H = 3 # 은닉 상태 벡터의 차원 수
T = 20 # 시계열 데이터의 길이

dh = np.ones((N,H))
np.random.seed(3) # 재현할 수 있도록 난수의 시드 고정
Wh = np.random.rand(H, H)

norm_list = []
for t in range(T):
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)
    
    
print(norm_list)


plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.rc('font', family='Malgun Gothic') # 폰트
plt.xlabel('시간 크기(time step)')
plt.ylabel('노름(norm)')
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

N = 2 # 미니배치 크기
H = 3 # 은닉 상태 벡터의 차원 수
T = 20 # 시계열 데이터의 길이

dh = np.ones((N,H))
np.random.seed(3) # 재현할 수 있도록 난수의 시드 고정
Wh = np.random.rand(H, H) * 0.5

norm_list = []
for t in range(T):
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)
    
    
print(norm_list)


plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.rc('font', family='Malgun Gothic') # 폰트
plt.xlabel('시간 크기(time step)')
plt.ylabel('노름(norm)')
plt.show()


# In[11]:


# 기울기 클리핑
 

dW1 = np.random.rand(3,3) * 10
dW2 = np.random.rand(3,3) * 10
grads = [dW1,dW2]
max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
            
            


# In[12]:


# LSTM 클래스

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx,Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    def forward(self, x, h_prev, c_prev): # 순전파
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        
        # slice
        
        f = A[:, : H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        
        c_next = f*c_prev + g * i
        h_next = o*np.tanh(c_next)
        
        self.cache = (x, h+prev, c_prev, i, f, g, o ,c_next)
        
        return h_next, c_next
        
        
        
        


# In[18]:


# LSTM 계층을 사용하는 Rnnlm 클래스

import sys
sys.path.append('..')
from time_layers import * 
import pickle

class Rnnlm:
    def __init__(self, vocab_size = 10000, wordvec_size = 100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # 가중치 초기화
        embed_W = (rn(V, D)/ 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) /np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()
        
        
    def save_params(self, file_name = 'Rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
            
    def load_params(self, file_name = 'Rnnlm.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)
        


# In[19]:


# PTB 데이터셋 학습

import sys
sys.path.append('..')
from optimizer import SGD
from trainer import RnnlmTrainer
from util import eval_perplexity
import pbt


# 하이퍼파라미터 설정
batch_size = 20
wordvec_size = 100
hidden_size = 100 # RNN의 은닉 상태 벡터의 원소 수
time_size = 35 # RNN을 펼치는 크기
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = pbt.load_data('train')
corpus_test, _, _ = pbt.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# 모델 생성
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 기울기 클리핑을 적용해 학습
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0,500))
# 20번재 반복마다 퍼블렉서티를 평가
# 데이터가 크기 때문에 모든 에폭에서 평가하지 않고, 20번 반복될 때마다 평가

# 테스트 데이터로 평가
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('테스트 퍼블렉서티 : ', ppl_test)
# 테스트 데이터를 이용해 퍼블렉서티 평가
# 모델 상태 LSTM의 은닉 상태와 기억 셀를 재설정해 평가를 수행


# 매개변수 저장
model.save_params()


# In[22]:


# 기울기 구하기
model.forward(...)
model.backward(...)
params, grads = model.params, model.grads

# 기울기 클리핑
if max_grad is not None:
    clip_grads(grads, max_grad)

# 매개변수 갱신
optimizer.update(params, grads)


# In[42]:


# BetterRnnlm 클래스

import sys
sys.path.append('..')
from time_layers import *
from np import *
from base_model import BaseModel

class BetterRnnlm(BaseModel):
    def __init__(self, vocab_size = 10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
    
        embed_W = (rn(V, D)/ 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H)/np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H)/np.sqrt(H)).astype('f')
        lstm_b1 =  np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H)/np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H)/np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4*H).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        
        # 세 가지 개선
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b) # 가중치 공유
            
            
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2],self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def predict(self, xs, train_flag = False):
        for layer in self.drop_layers:
            layer.train_flg = train_flag
            
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
        


# In[47]:


import sys
sys.path.append('..')
import config

from optimizer import SGD
from trainer import RnnlmTrainer
from util import eval_perplexity
import pbt

config.GPU = True
# 하이퍼파라미터 설정
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = pbt.load_data('train')
corpus_val, _, _ = pbt.load_data('val')
corpus_test, _, _ = pbt.load_data('test')

if config.GPU:
    corpus = to_gpu(corpus)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
                time_size=time_size, max_grad=max_grad)

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('검증 퍼플렉서티: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)


# 테스트 데이터로 평가
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('테스트 퍼플렉서티: ', ppl_test)


# In[ ]:




